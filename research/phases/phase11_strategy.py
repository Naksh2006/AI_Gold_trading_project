import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_cached_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file}. Run phase9 first to cache data.")
    df = pd.read_csv(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. EXACT PINE-PARITY INSTITUTIONAL FEATURES (12 INPUTS)
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

    # News Shock Detector
    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df['shock_ratio'] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    # f1, f2, f3: Multi-Horizon Volatility-Adjusted Returns (30m, 1h, 3h)
    df['f1'] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df['f2'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df['f3'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # f4: Rolling 24-Bar (6h) Institutional VWAP Distance
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df['f4'] = ((c - vwap24) / c) / atr_pct

    # f5: Wilder RSI(14) centered
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['rsi14_raw'] = rsi14
    df['f5'] = (rsi14 - 50.0) / 50.0

    # f6: Normalized MACD Histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['f6'] = macd_hist / (atr14 + 1e-10)

    # f7: Donchian / Stochastic 16-Bar Price Position [-0.5, +0.5]
    hh16 = h.rolling(16).max()
    ll16 = l.rolling(16).min()
    df['f7'] = np.where((hh16 - ll16) == 0, 0.0, ((c - ll16) / (hh16 - ll16)) - 0.5)

    # f8, f9: Multi-Scale EMA Trend Hierarchy (EMA 20, EMA 60, EMA 180)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    ema180 = c.ewm(span=180, adjust=False).mean()
    df['f8'] = ((ema20 - ema60) / c) / atr_pct
    df['f9'] = ((ema60 - ema180) / c) / atr_pct

    # f10: Order Flow VSA (4-bar Institutional Buying/Selling Pressure)
    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df['f10'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # f11, f12: Cross-Asset USD Inverse (EURUSDT) & Liquidity Beta (BTCUSDT)
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f11'] = eur_ret4 * 100.0
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Kaufman Efficiency Ratio (12)
    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df['er12'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df

# =========================================================================
# 2. REALISTIC WALK-FORWARD STRATEGY SIMULATOR
# =========================================================================
def simulate_strategy(df_slice, probs, horizon=8, up_th=0.58, tp_atr=1.6, sl_atr=1.8, trail_atr=0.9, er_min=0.15, cooldown=2, fee=0.0001):
    closes = df_slice['close'].values
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    atrs = df_slice['atr'].values
    vwap_d = df_slice['f4'].values
    macd_h = df_slice['f6'].values
    ema_fast = df_slice['f8'].values
    er = df_slice['er12'].values
    shocks = df_slice['shock_ratio'].values
    dates = pd.to_datetime(df_slice['datetime']).values

    dn_th = 1.0 - up_th
    n = len(closes)
    trades = []
    i = 0

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        # Block Sunday low-liquidity session, flat chop, and extreme news-wick bars
        if dt_obj.dayofweek == 6 or er[i] < er_min or shocks[i] > 2.3:
            i += 1
            continue

        p = probs[i]
        direction = 0

        # Long: High AI confidence + positive momentum + not excessively overbought
        if p >= up_th and ema_fast[i] > -0.4 and macd_h[i] > -0.2 and vwap_d[i] < 2.2:
            direction = 1
        # Short: High Bearish AI confidence + negative momentum + not excessively oversold
        elif p <= dn_th and ema_fast[i] < 0.4 and macd_h[i] < 0.2 and vwap_d[i] > -2.2:
            direction = -1

        if direction != 0:
            entry = closes[i]
            cur_atr = atrs[i]
            tp_price = entry + direction * (tp_atr * cur_atr)
            sl_price = entry - direction * (sl_atr * cur_atr)
            trail_trig = entry + direction * (trail_atr * cur_atr)

            exit_price = closes[i + horizon]
            bars_held = horizon
            hit_loss = False

            for k in range(1, horizon + 1):
                if direction == 1:
                    if highs[i + k] >= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if lows[i + k] <= sl_price:
                        exit_price, bars_held = sl_price, k
                        hit_loss = (sl_price < entry)
                        break
                    if highs[i + k] >= trail_trig:
                        sl_price = max(sl_price, entry + 0.35 * cur_atr)
                else:
                    if lows[i + k] <= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if highs[i + k] >= sl_price:
                        exit_price, bars_held = sl_price, k
                        hit_loss = (sl_price > entry)
                        break
                    if lows[i + k] <= trail_trig:
                        sl_price = min(sl_price, entry - 0.35 * cur_atr)

            net_ret = direction * ((exit_price - entry) / entry) - fee
            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'ret': net_ret
            })
            i += bars_held + (cooldown if hit_loss else 1)
        else:
            i += 1

    return pd.DataFrame(trades)

def evaluate_baselines(df_test, horizon=8, fee=0.0001):
    closes = df_test['close'].values
    rsi = df_test['rsi14_raw'].values
    n = len(closes)

    b1_rets = [((closes[i + horizon] - closes[i]) / closes[i]) - fee for i in range(0, n - horizon - 1, horizon)]
    b2_rets = []
    i = 0
    while i < n - horizon - 1:
        if rsi[i] < 30.0:
            b2_rets.append(((closes[i + horizon] - closes[i]) / closes[i]) - fee)
            i += horizon
        elif rsi[i] > 70.0:
            b2_rets.append(((closes[i] - closes[i + horizon]) / closes[i]) - fee)
            i += horizon
        else:
            i += 1

    b1_rets = np.array(b1_rets)
    b2_rets = np.array(b2_rets) if len(b2_rets) > 0 else np.array([0.0])
    return {
        'Always_Up': {'trades': len(b1_rets), 'wr': (b1_rets > 0).mean() * 100, 'pnl': b1_rets.sum() * 100},
        'Basic_RSI': {'trades': len(b2_rets), 'wr': (b2_rets > 0).mean() * 100, 'pnl': b2_rets.sum() * 100}
    }

def main():
    print("=== PHASE 11: HONEST WALK-FORWARD ENGINE (PROFITABLE VALIDATION -> BLIND TEST) ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 13)]

    # Clean Directional Return Target over 6 bars (90m intraday momentum continuation)
    TARGET_H = 6
    df['future_ret'] = (df['close'].shift(-TARGET_H) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)

    # Symmetric target labeling on decisive moves (> 0.08%)
    df['target'] = np.where(df['future_ret'] > 0.0008, 1,
                   np.where(df['future_ret'] < -0.0008, 0, -1))

    # -------------------------------------------------------------------------
    # STRICT 3-WAY CHRONOLOGICAL SPLIT (ZERO LEAKAGE!)
    # -------------------------------------------------------------------------
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df['datetime'] <= TRAIN_END].copy()
    val_df   = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df  = df[df['datetime'] > VAL_END].copy()

    print(f"1. [TRAIN SET]      : {train_df['datetime'].min().date()} -> {train_df['datetime'].max().date()} ({len(train_df)} bars)")
    print(f"2. [VALIDATION SET] : {val_df['datetime'].min().date()} -> {val_df['datetime'].max().date()} ({len(val_df)} bars) <-- ALL TUNING HERE")
    print(f"3. [BLIND TEST SET] : {test_df['datetime'].min().date()} -> {test_df['datetime'].max().date()} ({len(test_df)} bars) <-- LOCKED VAULT\n")

    # Use recent 12 months inside Train Set so weights reflect modern Gold volatility regime
    train_recent = train_df.iloc[-35000:].copy()
    train_clean = train_recent[(train_recent['target'] != -1) & (train_recent['shock_ratio'] <= 2.3)].copy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_clean[feature_cols])
    y_train = train_clean['target']

    lr = LogisticRegression(C=0.1, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', alpha=0.25, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    train_raw_p = 0.55 * lr.predict_proba(X_train_s)[:, 1] + 0.45 * mlp.predict_proba(X_train_s)[:, 1]
    mean_p, std_p = float(np.mean(train_raw_p)), float(np.std(train_raw_p)) + 1e-10

    # -------------------------------------------------------------------------
    # STEP 2: OPTIMIZE STRICTLY ON VALIDATION SET (SEP 1 - NOV 8)
    # -------------------------------------------------------------------------
    X_val_s = scaler.transform(val_df[feature_cols])
    val_raw_p = 0.55 * lr.predict_proba(X_val_s)[:, 1] + 0.45 * mlp.predict_proba(X_val_s)[:, 1]
    val_probs = 1.0 / (1.0 + np.exp(-((val_raw_p - mean_p) / (std_p * 0.75))))

    best_val_score = -9999.0
    best_cfg = None

    for h_bars in [6, 8, 10]:
        for th in [0.56, 0.58, 0.60, 0.62, 0.65, 0.68]:
            for tp_m, sl_m, tr_m in [(1.5, 1.8, 0.9), (1.6, 1.6, 0.9), (1.8, 1.5, 1.0), (1.4, 2.0, 0.8)]:
                for er_min in [0.14, 0.18]:
                    v_tr = simulate_strategy(
                        val_df, val_probs, horizon=h_bars, up_th=th,
                        tp_atr=tp_m, sl_atr=sl_m, trail_atr=tr_m, er_min=er_min, cooldown=3
                    )
                    # Require 45 to 220 trades over 2.2 months (~1 to 3 trades/day, no over-trading!)
                    if 45 <= len(v_tr) <= 220:
                        v_pnl = v_tr['ret'].sum() * 100.0
                        v_wr = (v_tr['ret'] > 0).mean() * 100.0
                        gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                        gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                        v_pf = gw / gl
                        if v_pnl > 0 and v_wr >= 52.0:
                            score = v_pnl * (v_pf ** 1.5) * (v_wr / 50.0)
                            if score > best_val_score:
                                best_val_score = score
                                best_cfg = {
                                    'h': h_bars, 'th': th, 'tp': tp_m, 'sl': sl_m, 'tr': tr_m, 'er': er_min,
                                    'val_pnl': v_pnl, 'val_wr': v_wr, 'val_pf': v_pf, 'val_trades': len(v_tr)
                                }

    # Fallback if strict filter wasn't hit
    if best_cfg is None:
        for th in [0.56, 0.58, 0.60, 0.62, 0.65]:
            v_tr = simulate_strategy(val_df, val_probs, horizon=8, up_th=th, tp_atr=1.6, sl_atr=1.6, trail_atr=0.9, er_min=0.15, cooldown=3)
            if len(v_tr) >= 30:
                v_pnl = v_tr['ret'].sum() * 100.0
                v_wr = (v_tr['ret'] > 0).mean() * 100.0
                gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                v_pf = gw / gl
                if v_pnl > best_val_score:
                    best_val_score = v_pnl
                    best_cfg = {'h': 8, 'th': th, 'tp': 1.6, 'sl': 1.6, 'tr': 0.9, 'er': 0.15, 'val_pnl': v_pnl, 'val_wr': v_wr, 'val_pf': v_pf, 'val_trades': len(v_tr)}

    print("--- STEP 2: LOCKED OPTIMAL PARAMETERS FROM VALIDATION SET (SEP 1 - NOV 8) ---")
    print(f"   Confidence Thresh : >= {best_cfg['th']*100:.0f}% | Horizon: {best_cfg['h']} bars | TP: {best_cfg['tp']}x ATR | SL: {best_cfg['sl']}x ATR | Trail: +{best_cfg['tr']}x ATR")
    print(f"   Validation Result : {best_cfg['val_trades']} trades | Win Rate: {best_cfg['val_wr']:.2f}% | Profit Factor: {best_cfg['val_pf']:.2f} | Net PnL: {best_cfg['val_pnl']:+.2f}%\n")

    # -------------------------------------------------------------------------
    # STEP 3: BLIND OUT-OF-SAMPLE TEST (NOV 9 - NOV 15) + DAY-BY-DAY TABLE
    # -------------------------------------------------------------------------
    X_test_s = scaler.transform(test_df[feature_cols])
    test_raw_p = 0.55 * lr.predict_proba(X_test_s)[:, 1] + 0.45 * mlp.predict_proba(X_test_s)[:, 1]
    test_probs = 1.0 / (1.0 + np.exp(-((test_raw_p - mean_p) / (std_p * 0.75))))

    blind_trades = simulate_strategy(
        test_df, test_probs, horizon=best_cfg['h'], up_th=best_cfg['th'],
        tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], trail_atr=best_cfg['tr'], er_min=best_cfg['er'], cooldown=3
    )

    baselines = evaluate_baselines(test_df, horizon=best_cfg['h'])

    if len(blind_trades) > 0:
        print("=== STEP 3A: BLIND TEST DAY-BY-DAY BREAKDOWN (NOV 9 – NOV 15, 2026) ===")
        print("-" * 70)
        print(f"{'Date':<14} | {'Trades':<8} | {'Winners':<9} | {'Win Rate':<10} | {'Daily Net PnL'}")
        print("-" * 70)
        for dt, grp in blind_trades.groupby('date'):
            n_t = len(grp)
            n_w = (grp['ret'] > 0).sum()
            wr = (n_w / n_t) * 100.0
            d_pnl = grp['ret'].sum() * 100.0
            print(f"{dt:<14} | {n_t:<8} | {n_w:<9} | {wr:<9.1f}% | {d_pnl:+.2f}%")
        print("-" * 70)

        t_pnl = blind_trades['ret'].sum() * 100.0
        t_wr = (blind_trades['ret'] > 0).mean() * 100.0
        t_gw = blind_trades[blind_trades['ret'] > 0]['ret'].sum()
        t_gl = abs(blind_trades[blind_trades['ret'] < 0]['ret'].sum()) + 1e-10
        t_pf = t_gw / t_gl

        print("\n=== STEP 3B: BLIND TEST SUMMARY VS. REQUIRED RUBRIC BASELINES ===")
        print("-" * 76)
        print(f"{'Strategy / Benchmark':<32} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL (After Fees)'}")
        print("-" * 76)
        print(f"{'1. Baseline: Always Predict UP':<32} | {baselines['Always_Up']['trades']:<8} | {baselines['Always_Up']['wr']:<9.1f}% | {baselines['Always_Up']['pnl']:+.2f}%")
        print(f"{'2. Baseline: Basic RSI (30/70)':<32} | {baselines['Basic_RSI']['trades']:<8} | {baselines['Basic_RSI']['wr']:<9.1f}% | {baselines['Basic_RSI']['pnl']:+.2f}%")
        print(f"{'3. OUR HONEST AI ENGINE':<32} | {len(blind_trades):<8} | {t_wr:<9.1f}% | {t_pnl:+.2f}% (PF: {t_pf:.2f})")
        print("-" * 76)

    # -------------------------------------------------------------------------
    # STEP 4: COMPILE FINAL PINE SCRIPT (AI_Gold_Viva_Winner.pine)
    # -------------------------------------------------------------------------
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Final Honest Walk-Forward (Trained <= Aug 31, Val <= Nov 8)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. VIVA EVALUATION DATE-RANGE PICKER
// =========================================================================
use_date_filter = input.bool(true, "Enable Custom Date Range Filter", group="1. Viva Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Test Start Date (UTC)", group="1. Viva Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "Test End Date (UTC)", group="1. Viva Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session (Validated Sep-Nov 8)", group="1. Viva Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Viva Test Window")

// =========================================================================
// 2. VALIDATION-LOCKED PARAMETERS (Tuned Strictly on Sep 1 - Nov 8, 2026)
// =========================================================================
conf_thresh  = input.float({best_cfg['th']:.2f}, "AI Confidence Threshold", minval=0.50, maxval=0.75, step=0.01, group="2. Validation-Locked Settings")
er_min_val   = input.float({best_cfg['er']:.2f}, "Min Efficiency Ratio (Chop Filter)", minval=0.05, maxval=0.35, step=0.01, group="2. Validation-Locked Settings")
tp_mult      = input.float({best_cfg['tp']:.1f}, "Take Profit (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="2. Validation-Locked Settings")
sl_mult      = input.float({best_cfg['sl']:.1f}, "Stop Loss (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="2. Validation-Locked Settings")
trail_trig_m = input.float({best_cfg['tr']:.1f}, "Profit Ratchet Trigger (ATR)", minval=0.4, maxval=2.5, step=0.1, group="2. Validation-Locked Settings")
horizon_bars = input.int({best_cfg['h']}, "Holding Horizon (Bars)", minval=3, maxval=24, group="2. Validation-Locked Settings")
cooldown_n   = input.int(3, "Post-Stop Cooldown Bars", minval=0, maxval=10, group="2. Validation-Locked Settings")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 12 STATIONARY FEATURES
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

bar_shock = (high - low) / (atr14 + 1e-10)
jump_shock = math.abs(close - nz(close[2], close[1])) / (atr14 + 1e-10)
shock_ratio = math.max(bar_shock, math.max(nz(bar_shock[1], 0.0), jump_shock / 1.5))
shock_ok = shock_ratio <= 2.3

f1 = ((close - nz(close[2], close[1])) / nz(close[2], close[1])) / atr_pct
f2 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f3 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct

typ_p = (high + low + close) / 3.0
vwap24 = math.sum(typ_p * volume, 24) / (math.sum(volume, 24) + 1e-10)
f4 = ((close - vwap24) / close) / atr_pct

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f5 = (rsi14 - 50.0) / 50.0

ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
macd_line = ema12 - ema26
macd_hist = macd_line - ta.ema(macd_line, 9)
f6 = macd_hist / (atr14 + 1e-10)

hh16 = ta.highest(high, 16)
ll16 = ta.lowest(low, 16)
f7 = (hh16 - ll16) == 0 ? 0.0 : ((close - ll16) / (hh16 - ll16)) - 0.5

ema20 = ta.ema(close, 20)
ema60 = ta.ema(close, 60)
ema180 = ta.ema(close, 180)
f8 = ((ema20 - ema60) / close) / atr_pct
f9 = ((ema60 - ema180) / close) / atr_pct

rng = (high - low) + 1e-10
signed_vol = ((close - low) - (high - close)) / rng * volume
f10 = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4 = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4 = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])

f11 = nz(eur_ret4, 0.0) * 100.0
f12 = nz(eur_ret4 - gold_ret4 + 0.15 * btc_ret4, 0.0) * 100.0

net_move = math.abs(close - nz(close[12], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 12)
er12 = sum_moves == 0 ? 0.0 : net_move / sum_moves

// =========================================================================
// 4. FROZEN AI WEIGHTS (TRAINED STRICTLY <= AUG 31, 2026)
// =========================================================================
"""
    for idx in range(12):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    lr_eq = f"{b_lr:.8f}"
    for idx in range(12):
        lr_eq += f" + (s{idx+1} * {w_lr[idx]:.8f})"
    pine_code += f"\nlogit_lr = {lr_eq}\n"
    pine_code += "p_lr = 1.0 / (1.0 + math.exp(-logit_lr))\n\n"

    for j in range(len(b0)):
        eq = f"{b0[j]:.8f}"
        for idx in range(12):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"z{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z{j}) - 1.0) / (math.exp(2.0 * z{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

    pine_code += f"""raw_p = 0.55 * p_lr + 0.45 * p_mlp
z_prob = (raw_p - {mean_p:.8f}) / ({std_p * 0.75:.8f})
prob_up = 1.0 / (1.0 + math.exp(-z_prob))

// =========================================================================
// 5. EXECUTION & PROFIT-RATCHET ENGINE
// =========================================================================
dn_thresh = 1.0 - conf_thresh

is_up   = in_date_window and shock_ok and (er12 >= er_min_val) and (prob_up >= conf_thresh) and (f8 > -0.4) and (f6 > -0.2) and (f4 < 2.2)
is_down = in_date_window and shock_ok and (er12 >= er_min_val) and (prob_up <= dn_thresh) and (f8 < 0.4) and (f6 < 0.2) and (f4 > -2.2)

var int entry_bar = na
var int last_loss_bar = -999
var float entry_p = na
var float entry_atr = na
var float tp_level = na
var float sl_level = na

cooldown_ok = (bar_index - last_loss_bar) >= cooldown_n

if barstate.isconfirmed and strategy.position_size == 0 and cooldown_ok
    if is_up
        strategy.entry("AI Long", strategy.long, comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
        entry_bar := bar_index
        entry_p   := close
        entry_atr := atr14
        tp_level  := close + (atr14 * tp_mult)
        sl_level  := close - (atr14 * sl_mult)
    else if is_down
        strategy.entry("AI Short", strategy.short, comment="DN " + str.tostring(math.round((1.0 - prob_up) * 100)) + "%")
        entry_bar := bar_index
        entry_p   := close
        entry_atr := atr14
        tp_level  := close - (atr14 * tp_mult)
        sl_level  := close + (atr14 * sl_mult)

if strategy.position_size > 0
    if high >= (entry_p + entry_atr * trail_trig_m)
        sl_level := math.max(sl_level, entry_p + entry_atr * 0.35)
    strategy.exit("Exit Long", "AI Long", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (entry_p - entry_atr * trail_trig_m)
        sl_level := math.min(sl_level, entry_p - entry_atr * 0.35)
    strategy.exit("Exit Short", "AI Short", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_loss_bar := bar_index

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. OVERLAYS & VIVA HUD
// =========================================================================
plot(vwap24, "6H Institutional VWAP", color=color.new(color.fuchsia, 25), linewidth=2)
plot(ema60, "EMA 60 Core Trend", color=color.new(color.yellow, 25), linewidth=2)

var table hud = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    sig_txt = not shock_ok ? "VETO (NEWS SHOCK)" : is_up ? "BUY / LONG" : is_down ? "SELL / SHORT" : "WAIT / HOLD"
    sig_col = not shock_ok ? color.orange : is_up ? color.green : is_down ? color.red : color.gray
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round((1.0 - prob_up) * 100, 1)
    
    table.cell(hud, 0, 0, "Validation Protocol", text_color=color.white)
    table.cell(hud, 1, 0, "3-Way Split (Zero Leakage)", text_color=color.yellow)
    table.cell(hud, 0, 1, "Live Signal", text_color=color.white)
    table.cell(hud, 1, 1, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 2, "AI Confidence", text_color=color.white)
    table.cell(hud, 1, 2, str.tostring(conf_pct) + "% (Min " + str.tostring(math.round(conf_thresh*100)) + "%)", text_color=color.white)
    table.cell(hud, 0, 3, "News Shock Shield", text_color=color.white)
    table.cell(hud, 1, 3, shock_ok ? "SAFE (" + str.tostring(math.round(shock_ratio, 1)) + "x ATR)" : "SHOCK VETO", text_color=shock_ok ? color.lime : color.red)
    table.cell(hud, 0, 4, "Regime Efficiency", text_color=color.white)
    table.cell(hud, 1, 4, er12 >= er_min_val ? "TRENDING" : "CHOP FILTERED", text_color=er12 >= er_min_val ? color.lime : color.orange)
"""

    with open("AI_Gold_Viva_Winner.pine", "w") as f:
        f.write(pine_code)

    print("\n4. Saved 'AI_Gold_Viva_Winner.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()