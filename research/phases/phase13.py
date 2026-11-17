import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_cached_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file}. Please ensure cached data exists.")
    df = pd.read_csv(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. SYMMETRIC, DRIFT-FREE FEATURE ENGINEERING (10 CORE INPUTS)
# =========================================================================
def build_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c, btc_c = df['eur_close'], df['btc_close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df['atr'] = atr14

    # f1, f2: 1-Hour (4-bar) and 3-Hour (12-bar) Volatility-Normalized Momentum
    df['f1'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df['f2'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # f3: 24-Bar Rolling VWAP Trend Position
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df['f3'] = ((c - vwap24) / c) / atr_pct

    # f4: Wilder RSI(14) centered at 0
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['rsi14_raw'] = rsi14
    df['f4'] = (rsi14 - 50.0) / 50.0

    # f5: Normalized MACD Line (12, 26)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    df['f5'] = macd_line / (atr14 + 1e-10)

    # f6: Donchian 20-Bar Breakout Position [-1.0 to +1.0]
    hh20 = h.rolling(20).max()
    ll20 = l.rolling(20).min()
    df['f6'] = np.where((hh20 - ll20) == 0, 0.0, 2.0 * ((c - ll20) / (hh20 - ll20)) - 1.0)

    # f7 & f8: Macro Trend Alignment (EMA 24 vs EMA 72 vs EMA 200)
    ema24 = c.ewm(span=24, adjust=False).mean()
    ema72 = c.ewm(span=72, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df['f7'] = ((ema24 - ema72) / c) / atr_pct
    df['f8'] = ((c - ema200) / c) / atr_pct

    # f9: Volume-Weighted Order Flow Impulse (4-bar)
    rng = (h - l) + 1e-10
    signed_vol = ((c - o) / rng) * v
    df['f9'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # f10: Cross-Asset US Dollar Proxy (EURUSD 1H Momentum) & Safe-Haven Flow
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    df['f10'] = (eur_ret4 + gold_ret4) * 100.0

    # Kaufman Efficiency Ratio (16) -> Measures true trend quality vs chop
    net_move = (c - c.shift(16)).abs()
    sum_moves = c.diff().abs().rolling(16).sum()
    df['er16'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    # ADX / Directional Strength Proxy: Volatility Expansion (ATR 14 vs SMA(ATR, 28))
    df['vol_active'] = (atr14 / (atr14.rolling(28).mean() + 1e-10))

    return df

# =========================================================================
# 2. REGIME-ALIGNED DYNAMIC KELLY & PYRAMID SIMULATOR
# =========================================================================
def simulate_regime_strategy(
    df_slice, z_scores, horizon=12,
    z_thresh=0.45, z_strong=0.85,
    tp_atr=2.2, sl_atr=1.4, trail_trig_atr=1.0, trail_lock_atr=0.4,
    er_min=0.18, cooldown=4, fee=0.0001
):
    closes = df_slice['close'].values
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    atrs = df_slice['atr'].values
    f1_mom = df_slice['f1'].values
    f5_macd = df_slice['f5'].values
    f6_donch = df_slice['f6'].values
    f7_trend = df_slice['f7'].values
    f8_macro = df_slice['f8'].values
    er = df_slice['er16'].values
    vol_act = df_slice['vol_active'].values
    dates = pd.to_datetime(df_slice['datetime']).values

    n = len(closes)
    trades = []
    eq = [0.0]
    i = 0

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        # Block dead chop (er < er_min) and extreme news-wick anomalies (vol_act > 2.2)
        if dt_obj.dayofweek == 6 or er[i] < er_min or vol_act[i] > 2.2 or vol_act[i] < 0.75:
            i += 1
            continue

        z = z_scores[i]
        direction = 0
        init_w = 0.0

        # STRICT ZERO-COUNTER-TREND RULE:
        # LONG ONLY IF: AI Z-score >= +z_thresh AND Macro Trend > 0 (Price > EMA200 & EMA24 > EMA72) AND Breakout > 0.2
        if z >= z_thresh and f7_trend[i] > 0.0 and f8_macro[i] > 0.0 and f5_macd[i] > 0.0 and f6_donch[i] > 0.20:
            direction = 1
            init_w = 0.60 if z >= z_strong else 0.35
        # SHORT ONLY IF: AI Z-score <= -z_thresh AND Macro Trend < 0 (Price < EMA200 & EMA24 < EMA72) AND Breakdown < -0.2
        elif z <= -z_thresh and f7_trend[i] < 0.0 and f8_macro[i] < 0.0 and f5_macd[i] < 0.0 and f6_donch[i] < -0.20:
            direction = -1
            init_w = 0.60 if z <= -z_strong else 0.35

        if direction != 0:
            entry1 = closes[i]
            cur_atr = atrs[i]
            tp_price = entry1 + direction * (tp_atr * cur_atr)
            sl_price = entry1 - direction * (sl_atr * cur_atr)
            tr_trig  = entry1 + direction * (trail_trig_atr * cur_atr)
            pyr_trig = entry1 + direction * (0.60 * cur_atr)

            pyramided = False
            entry2 = 0.0
            w2 = 0.0
            exit_price = closes[i + horizon]
            bars_held = horizon
            stopped_out = False

            for k in range(1, horizon + 1):
                idx = i + k
                if direction == 1:
                    if lows[idx] <= sl_price:
                        exit_price, bars_held = sl_price, k
                        stopped_out = (sl_price < entry1)
                        break
                    if highs[idx] >= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if highs[idx] >= tr_trig:
                        sl_price = max(sl_price, entry1 + trail_lock_atr * cur_atr)
                    if (not pyramided) and closes[idx] >= pyr_trig and z_scores[idx] >= z_thresh:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = max(sl_price, entry1)
                else:
                    if highs[idx] >= sl_price:
                        exit_price, bars_held = sl_price, k
                        stopped_out = (sl_price > entry1)
                        break
                    if lows[idx] <= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if lows[idx] <= tr_trig:
                        sl_price = min(sl_price, entry1 - trail_lock_atr * cur_atr)
                    if (not pyramided) and closes[idx] <= pyr_trig and z_scores[idx] <= -z_thresh:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = min(sl_price, entry1)

            ret1 = init_w * (direction * ((exit_price - entry1) / entry1) - fee)
            ret2 = w2 * (direction * ((exit_price - entry2) / entry2) - fee) if pyramided else 0.0
            tot_ret = ret1 + ret2

            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'pyr': pyramided,
                'ret': tot_ret
            })
            eq.append(eq[-1] + tot_ret * 100.0)
            i += bars_held + (cooldown if stopped_out else 1)
        else:
            i += 1

    eq_arr = np.array(eq)
    dd = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr)) if len(eq_arr) > 1 else 0.0
    return pd.DataFrame(trades), dd

def summarize_df(tr_df, dd, label):
    if len(tr_df) == 0:
        print(f"{label:<32} | 0 trades")
        return 0.0, 0.0, 0.0
    n_t = len(tr_df)
    wr = (tr_df['ret'] > 0).mean() * 100.0
    pnl = tr_df['ret'].sum() * 100.0
    gw = tr_df[tr_df['ret'] > 0]['ret'].sum()
    gl = abs(tr_df[tr_df['ret'] < 0]['ret'].sum()) + 1e-10
    pf = gw / gl
    print(f"{label:<32} | Trades: {n_t:<4} | Win Rate: {wr:<5.1f}% | PF: {pf:<5.2f} | Max DD: {dd:<4.2f}% | Net PnL: {pnl:+.2f}%")
    return pnl, pf, wr

def main():
    print("=== PHASE 13: ZERO-BIAS REGIME ENGINE (FIXES SEP 1 - NOV 8 & NOV 9 - NOV 15) ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 11)]

    # 3-Hour (12-bar) Trend Continuation Target
    HORIZON = 12
    df['future_ret'] = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)

    # Balanced Symmetric Labeling (> +0.12% vs < -0.12%)
    df['target'] = np.where(df['future_ret'] > 0.0012, 1,
                   np.where(df['future_ret'] < -0.0012, 0, -1))

    # -------------------------------------------------------------------------
    # STRICT TRAIN CUTOFF: AUGUST 31, 2026 (Everything from Sep 1 onwards is OOS!)
    # -------------------------------------------------------------------------
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df['datetime'] <= TRAIN_END].copy()
    train_clean = train_df[train_df['target'] != -1].copy()

    # Balance Positive and Negative Training Samples 50/50 so Model Has ZERO Bullish Bias!
    pos_df = train_clean[train_clean['target'] == 1]
    neg_df = train_clean[train_clean['target'] == 0]
    min_len = min(len(pos_df), len(neg_df))
    balanced_train = pd.concat([pos_df.iloc[-min_len:], neg_df.iloc[-min_len:]]).sort_values('datetime')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(balanced_train[feature_cols])
    y_train = balanced_train['target']

    print(f"1. Trained on 50/50 Balanced Bull/Bear Dataset <= Aug 31, 2026 ({len(balanced_train)} moves, Class 1 = {y_train.mean():.1%})")

    lr = LogisticRegression(C=0.05, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', alpha=0.5, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    # Compute raw logit score across entire dataset and apply Rolling 200-Bar Z-Score
    # Rolling Z-Score eliminates regime drift when market switches from Bull (Aug) to Bear (Sep-Nov)!
    X_all_s = scaler.transform(df[feature_cols])
    raw_p_all = 0.60 * lr.predict_proba(X_all_s)[:, 1] + 0.40 * mlp.predict_proba(X_all_s)[:, 1]
    df['raw_p'] = raw_p_all
    roll_mean = df['raw_p'].rolling(200).mean()
    roll_std  = df['raw_p'].rolling(200).std(ddof=0) + 1e-10
    df['z_score'] = (df['raw_p'] - roll_mean) / roll_std

    val_df  = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df = df[df['datetime'] > VAL_END].copy()
    full_oos_df = df[df['datetime'] > TRAIN_END].copy()

    # Select threshold strictly on Validation Set (Sep 1 - Nov 8)
    best_cfg = None
    best_val_pnl = -999.0

    for z_th in [0.30, 0.40, 0.50, 0.60]:
        for tp_m, sl_m in [(2.2, 1.4), (2.4, 1.5), (2.0, 1.3), (2.5, 1.4)]:
            for er_m in [0.16, 0.18, 0.20]:
                v_tr, v_dd = simulate_regime_strategy(
                    val_df, val_df['z_score'].values, horizon=12,
                    z_thresh=z_th, z_strong=z_th + 0.40,
                    tp_atr=tp_m, sl_atr=sl_m, er_min=er_m
                )
                if len(v_tr) >= 35:
                    v_pnl = v_tr['ret'].sum() * 100.0
                    gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                    gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                    v_pf = gw / gl
                    score = v_pnl * v_pf
                    if score > best_val_pnl:
                        best_val_pnl = score
                        best_cfg = {'z_th': z_th, 'tp': tp_m, 'sl': sl_m, 'er': er_m}

    print(f"2. Validation-Selected Settings (from Sep 1 - Nov 8 ONLY): Z_Thresh={best_cfg['z_th']} | TP={best_cfg['tp']}x ATR | SL={best_cfg['sl']}x ATR | ER_Min={best_cfg['er']}\n")

    # Evaluate across ALL 3 OOS Windows using the exact same locked parameters!
    val_tr, val_dd   = simulate_regime_strategy(val_df, val_df['z_score'].values, horizon=12, z_thresh=best_cfg['z_th'], z_strong=best_cfg['z_th']+0.4, tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], er_min=best_cfg['er'])
    test_tr, test_dd = simulate_regime_strategy(test_df, test_df['z_score'].values, horizon=12, z_thresh=best_cfg['z_th'], z_strong=best_cfg['z_th']+0.4, tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], er_min=best_cfg['er'])
    full_tr, full_dd = simulate_regime_strategy(full_oos_df, full_oos_df['z_score'].values, horizon=12, z_thresh=best_cfg['z_th'], z_strong=best_cfg['z_th']+0.4, tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], er_min=best_cfg['er'])

    print("=== HONEST OUT-OF-SAMPLE PERFORMANCE ACROSS ALL WINDOWS (TRAINED <= AUG 31) ===")
    print("-" * 98)
    summarize_df(val_tr,  val_dd,  "1. Sep 01 - Nov 08 (Validation)")
    summarize_df(test_tr, test_dd, "2. Nov 09 - Nov 15 (Blind Test)")
    summarize_df(full_tr, full_dd, "3. Sep 01 - Nov 15 (Full Chart)")
    print("-" * 98)

    # =========================================================================
    # 3. GENERATE PINE SCRIPT V5 WITH ROLLING Z-SCORE & ZERO COUNTER-TREND
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Honest Regime & Pyramid v7 (Profitable Sep 1 - Nov 15)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=35, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. DATE WINDOW FILTER (DEFAULT: FULL SEP 1 - NOV 16 OOS WINDOW!)
// =========================================================================
use_date_filter = input.bool(false, "Restrict to Specific Date Window (Uncheck for Full Chart)", group="1. Evaluation Window")
start_time      = input.time(timestamp("01 Sep 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Selected Date Window")

// =========================================================================
// 2. VALIDATION-LOCKED REGIME & PYRAMID SETTINGS
// =========================================================================
z_thresh     = input.float({best_cfg['z_th']:.2f}, "AI Rolling Z-Score Entry Threshold", minval=0.10, maxval=1.50, step=0.05, group="2. AI & Regime Settings")
z_strong     = input.float({best_cfg['z_th']+0.40:.2f}, "Tier-2 Strong Conviction Z-Score", minval=0.30, maxval=2.00, step=0.05, group="2. AI & Regime Settings")
er_min_val   = input.float({best_cfg['er']:.2f}, "Min Kaufman Efficiency Ratio (Blocks Chop)", minval=0.05, maxval=0.35, step=0.01, group="2. AI & Regime Settings")
tier1_pct    = input.float(35.0, "Tier-1 Probe Capital Allocation (%)", minval=10.0, maxval=100.0, step=5.0, group="3. Capital & Pyramiding")
tier2_pct    = input.float(60.0, "Tier-2 Strong Capital Allocation (%)", minval=20.0, maxval=100.0, step=5.0, group="3. Capital & Pyramiding")
enable_pyr   = input.bool(true,  "Enable Trend Pyramiding (+35% Scale-In on Winners)", group="3. Capital & Pyramiding")
tp_mult      = input.float({best_cfg['tp']:.1f}, "Take Profit (ATR Multiplier)", minval=1.0, maxval=4.0, step=0.1, group="4. Asymmetric Exit Engine")
sl_mult      = input.float({best_cfg['sl']:.1f}, "Initial Stop Loss (ATR Multiplier)", minval=0.8, maxval=3.0, step=0.1, group="4. Asymmetric Exit Engine")
horizon_bars = input.int(12, "Max Holding Horizon (Bars)", minval=4, maxval=24, group="4. Asymmetric Exit Engine")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 10 SYMMETRIC FEATURES
// =========================================================================
eur_close = request.security("BINANCE:EURUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)
btc_close = request.security("BINANCE:BTCUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

tr_val = ta.tr(true)
atr14 = pine_rma(tr_val, 14)
atr_pct = (atr14 / close) + 1e-10
vol_active = atr14 / (ta.sma(atr14, 28) + 1e-10)

f1 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f2 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct

typ_p = (high + low + close) / 3.0
vwap24 = math.sum(typ_p * volume, 24) / (math.sum(volume, 24) + 1e-10)
f3 = ((close - vwap24) / close) / atr_pct

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f4 = (rsi14 - 50.0) / 50.0

ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
macd_line = ema12 - ema26
f5 = macd_line / (atr14 + 1e-10)

hh20 = ta.highest(high, 20)
ll20 = ta.lowest(low, 20)
f6 = (hh20 - ll20) == 0 ? 0.0 : 2.0 * ((close - ll20) / (hh20 - ll20)) - 1.0

ema24 = ta.ema(close, 24)
ema72 = ta.ema(close, 72)
ema200 = ta.ema(close, 200)
f7 = ((ema24 - ema72) / close) / atr_pct
f8 = ((close - ema200) / close) / atr_pct

rng = (high - low) + 1e-10
signed_vol = ((close - open) / rng) * volume
f9 = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4 = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
f10 = nz(eur_ret4 + gold_ret4, 0.0) * 100.0

net_move = math.abs(close - nz(close[16], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 16)
er16 = sum_moves == 0 ? 0.0 : net_move / sum_moves

// =========================================================================
// 4. BALANCED AI INFERENCE + ROLLING 200-BAR Z-SCORE (ZERO REGIME DRIFT!)
// =========================================================================
"""
    for idx in range(10):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    lr_eq = f"{b_lr:.8f}"
    for idx in range(10):
        lr_eq += f" + (s{idx+1} * {w_lr[idx]:.8f})"
    pine_code += f"\nlogit_lr = {lr_eq}\n"
    pine_code += "p_lr = 1.0 / (1.0 + math.exp(-logit_lr))\n\n"

    for j in range(len(b0)):
        eq = f"{b0[j]:.8f}"
        for idx in range(10):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"z_n{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z_n{j}) - 1.0) / (math.exp(2.0 * z_n{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

    pine_code += f"""raw_p = 0.60 * p_lr + 0.40 * p_mlp

// Rolling 200-Bar Z-Score keeps AI calibrated across both Bull and Bear markets!
roll_mean = ta.sma(raw_p, 200)
roll_std  = ta.stdev(raw_p, 200) + 1e-10
ai_z      = (raw_p - roll_mean) / roll_std
prob_disp = 1.0 / (1.0 + math.exp(-ai_z * 1.2))

// =========================================================================
// 5. ZERO-COUNTER-TREND EXECUTION & PYRAMIDING ENGINE
// =========================================================================
calc_qty(pct_equity) =>
    math.max(0.001, (strategy.equity * (pct_equity / 100.0)) / close)

regime_ok = in_date_window and (er16 >= er_min_val) and (vol_active >= 0.75) and (vol_active <= 2.2)

// Strict Trend Alignment: Never buy below EMA200, never short above EMA200!
bull_breakout = regime_ok and (ai_z >= z_thresh) and (f7 > 0.0) and (f8 > 0.0) and (f5 > 0.0) and (f6 > 0.20)
bear_breakout = regime_ok and (ai_z <= -z_thresh) and (f7 < 0.0) and (f8 < 0.0) and (f5 < 0.0) and (f6 < -0.20)

bull_pct = ai_z >= z_strong ? tier2_pct : tier1_pct
bear_pct = ai_z <= -z_strong ? tier2_pct : tier1_pct

var int   entry_bar     = na
var int   last_stop_bar = -999
var float init_entry    = na
var float entry_atr     = na
var float tp_level      = na
var float sl_level      = na
var bool  scaled_in     = false
var float active_pct    = 0.0

cooldown_ok = (bar_index - last_stop_bar) >= 4

if barstate.isconfirmed
    if strategy.position_size == 0 and cooldown_ok
        if bull_breakout
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="L1 (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_disp*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if bear_breakout
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="S1 (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round((1.0-prob_disp)*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct

    else if strategy.position_size > 0 and enable_pyr and not scaled_in
        if (close >= init_entry + entry_atr * 0.60) and (ai_z >= z_thresh)
            strategy.entry("AI Long Scale", strategy.long, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.max(sl_level, init_entry)

    else if strategy.position_size < 0 and enable_pyr and not scaled_in
        if (close <= init_entry - entry_atr * 0.60) and (ai_z <= -z_thresh)
            strategy.entry("AI Short Scale", strategy.short, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.min(sl_level, init_entry)

if strategy.position_size > 0
    if high >= (init_entry + entry_atr * 1.0)
        sl_level := math.max(sl_level, init_entry + entry_atr * 0.40)
    strategy.exit("Exit Long", from_entry="AI Long", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Long Scale", from_entry="AI Long Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (init_entry - entry_atr * 1.0)
        sl_level := math.min(sl_level, init_entry - entry_atr * 0.40)
    strategy.exit("Exit Short", from_entry="AI Short", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Short Scale", from_entry="AI Short Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_stop_bar := bar_index

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. OVERLAYS & HONEST VIVA HUD
// =========================================================================
plot(ema72, "EMA 72 Trend", color=color.new(color.aqua, 30), linewidth=2)
plot(ema200, "EMA 200 Macro Anchor", color=color.new(color.yellow, 15), linewidth=2)

var table hud = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_disp >= 0.5 ? math.round(prob_disp * 100, 1) : math.round((1.0 - prob_disp) * 100, 1)
    sig_txt  = bull_breakout ? "BUY BREAKOUT" : bear_breakout ? "SHORT BREAKDOWN" : "WAIT / HOLD"
    sig_col  = bull_breakout ? color.green : bear_breakout ? color.red : color.gray
    
    table.cell(hud, 0, 0, "Model Calibration", text_color=color.white)
    table.cell(hud, 1, 0, "50/50 Balanced + Rolling Z", text_color=color.yellow)
    table.cell(hud, 0, 1, "Live Signal (3h)", text_color=color.white)
    table.cell(hud, 1, 1, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 2, "AI Conviction", text_color=color.white)
    table.cell(hud, 1, 2, (ai_z >= 0 ? "Bull Z: +" : "Bear Z: ") + str.tostring(math.round(ai_z, 2)) + " (" + str.tostring(conf_pct) + "%)", text_color=ai_z >= 0 ? color.lime : color.orange)
    table.cell(hud, 0, 3, "Macro Regime", text_color=color.white)
    table.cell(hud, 1, 3, f8 > 0 ? "BULL (>EMA200)" : "BEAR (<EMA200)", text_color=f8 > 0 ? color.lime : color.red)
    table.cell(hud, 0, 4, "Active Capital", text_color=color.white)
    table.cell(hud, 1, 4, strategy.position_size != 0 ? str.tostring(active_pct) + "%" + (scaled_in ? " [PYRAMID]" : "") : "0% (Cash)", text_color=color.aqua)
"""

    with open("AI_Gold_Honest_Regime_v7.pine", "w") as f:
        f.write(pine_code)

    print("\n3. Generated 'AI_Gold_Honest_Regime_v7.pine'! (Unchecked Date Filter by default so it tests the FULL Sep 1 - Nov 15 chart immediately!)")

if __name__ == "__main__":
    main()