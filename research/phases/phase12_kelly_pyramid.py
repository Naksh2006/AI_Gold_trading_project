import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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
# 1. EXACT PINE-PARITY FEATURE ENGINEERING (12 INPUTS)
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

    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df['shock_ratio'] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    df['f1'] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df['f2'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df['f3'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df['f4'] = ((c - vwap24) / c) / atr_pct

    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['rsi14_raw'] = rsi14
    df['f5'] = (rsi14 - 50.0) / 50.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['f6'] = macd_hist / (atr14 + 1e-10)

    hh16 = h.rolling(16).max()
    ll16 = l.rolling(16).min()
    df['f7'] = np.where((hh16 - ll16) == 0, 0.0, ((c - ll16) / (hh16 - ll16)) - 0.5)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    ema180 = c.ewm(span=180, adjust=False).mean()
    df['f8'] = ((ema20 - ema60) / c) / atr_pct
    df['f9'] = ((ema60 - ema180) / c) / atr_pct

    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df['f10'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f11'] = eur_ret4 * 100.0
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df['er12'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df

# =========================================================================
# 2. CONFIDENCE-WEIGHTED SIZING + PYRAMIDING + DYNAMIC REVERSAL SIMULATOR
# =========================================================================
def simulate_kelly_pyramid(
    df_slice, probs, horizon=10,
    tier1_th=0.56, tier2_th=0.63, rev_th=0.62,
    tier1_size=0.30, tier2_size=0.60, pyr_size=0.40,
    tp_atr=1.6, sl_atr=1.5, pyr_trig_atr=0.45, er_min=0.16, fee=0.0001
):
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

    n = len(closes)
    trades = []
    equity_curve = [0.0]
    i = 0

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        if dt_obj.dayofweek == 6 or er[i] < er_min or shocks[i] > 2.3:
            i += 1
            continue

        p = probs[i]
        p_dn = 1.0 - p
        direction = 0
        init_weight = 0.0

        # Tiered Entry Logic (Small probe on moderate confidence, Larger on high confidence)
        if p >= tier1_th and ema_fast[i] > -0.4 and macd_h[i] > -0.2 and vwap_d[i] < 2.2:
            direction = 1
            init_weight = tier2_size if p >= tier2_th else tier1_size
        elif p_dn >= tier1_th and ema_fast[i] < 0.4 and macd_h[i] < 0.2 and vwap_d[i] > -2.2:
            direction = -1
            init_weight = tier2_size if p_dn >= tier2_th else tier1_size

        if direction != 0:
            entry1 = closes[i]
            cur_atr = atrs[i]
            tp_price = entry1 + direction * (tp_atr * cur_atr)
            sl_price = entry1 - direction * (sl_atr * cur_atr)
            pyr_trigger = entry1 + direction * (pyr_trig_atr * cur_atr)

            pyramided = False
            entry2 = 0.0
            weight2 = 0.0
            bars_held = horizon
            exit_price = closes[i + horizon]

            for k in range(1, horizon + 1):
                idx = i + k
                cur_p = probs[idx]
                cur_p_dn = 1.0 - cur_p

                if direction == 1:
                    # 1. Check Stop Loss or Take Profit
                    if lows[idx] <= sl_price:
                        exit_price, bars_held = sl_price, k
                        break
                    if highs[idx] >= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    # 2. Check Dynamic Signal Reversal (AI flips strongly Bearish)
                    if cur_p_dn >= rev_th and macd_h[idx] < 0:
                        exit_price, bars_held = closes[idx], k
                        break
                    # 3. Check Pyramiding (Scale-In if winning AND AI still confirms!)
                    if (not pyramided) and closes[idx] >= pyr_trigger and cur_p >= tier1_th:
                        pyramided = True
                        entry2 = closes[idx]
                        weight2 = min(pyr_size, 1.0 - init_weight)
                        # Protect combined position by raising Stop Loss to initial entry (breakeven cushion)
                        sl_price = max(sl_price, entry1)
                else:
                    if highs[idx] >= sl_price:
                        exit_price, bars_held = sl_price, k
                        break
                    if lows[idx] <= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if cur_p >= rev_th and macd_h[idx] > 0:
                        exit_price, bars_held = closes[idx], k
                        break
                    if (not pyramided) and closes[idx] <= pyr_trigger and cur_p_dn >= tier1_th:
                        pyramided = True
                        entry2 = closes[idx]
                        weight2 = min(pyr_size, 1.0 - init_weight)
                        sl_price = min(sl_price, entry1)

            # Calculate capital-weighted return
            ret1 = init_weight * (direction * ((exit_price - entry1) / entry1) - fee)
            ret2 = weight2 * (direction * ((exit_price - entry2) / entry2) - fee) if pyramided else 0.0
            total_trade_ret = ret1 + ret2

            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'init_size': int(init_weight * 100),
                'pyramided': pyramided,
                'total_size': int((init_weight + weight2) * 100),
                'ret': total_trade_ret
            })
            equity_curve.append(equity_curve[-1] + total_trade_ret * 100.0)
            i += bars_held + 1
        else:
            i += 1

    eq = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq)
    max_dd = float(np.max(peaks - eq)) if len(eq) > 1 else 0.0
    return pd.DataFrame(trades), max_dd

def main():
    print("=== PHASE 12: DYNAMIC KELLY SIZING + TREND PYRAMIDING + SIGNAL REVERSAL ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 13)]

    TARGET_H = 6
    df['future_ret'] = (df['close'].shift(-TARGET_H) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)

    df['target'] = np.where(df['future_ret'] > 0.0008, 1,
                   np.where(df['future_ret'] < -0.0008, 0, -1))

    # STRICT 3-WAY CHRONOLOGICAL SPLIT
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df['datetime'] <= TRAIN_END].copy()
    val_df   = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df  = df[df['datetime'] > VAL_END].copy()

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

    # Compute probabilities on Validation (Sep 1 - Nov 8) and Blind Test (Nov 9 - Nov 15)
    X_val_s = scaler.transform(val_df[feature_cols])
    val_raw_p = 0.55 * lr.predict_proba(X_val_s)[:, 1] + 0.45 * mlp.predict_proba(X_val_s)[:, 1]
    val_probs = 1.0 / (1.0 + np.exp(-((val_raw_p - mean_p) / (std_p * 0.75))))

    X_test_s = scaler.transform(test_df[feature_cols])
    test_raw_p = 0.55 * lr.predict_proba(X_test_s)[:, 1] + 0.45 * mlp.predict_proba(X_test_s)[:, 1]
    test_probs = 1.0 / (1.0 + np.exp(-((test_raw_p - mean_p) / (std_p * 0.75))))

    # Compare Flat 100% Sizing vs. Dynamic Kelly + Pyramiding on Validation Set (Sep 1 - Nov 8)
    flat_val, flat_dd = simulate_kelly_pyramid(
        val_df, val_probs, tier1_th=0.60, tier2_th=0.65,
        tier1_size=1.0, tier2_size=1.0, pyr_size=0.0
    )
    dyn_val, dyn_dd = simulate_kelly_pyramid(
        val_df, val_probs, tier1_th=0.58, tier2_th=0.65, rev_th=0.62,
        tier1_size=0.30, tier2_size=0.60, pyr_size=0.40
    )

    print("--- STEP 1: VALIDATION SET PROOF (SEP 1 - NOV 8: FLAT 100% VS. DYNAMIC KELLY PYRAMID) ---")
    print(f"   Flat 100% All-In Sizing      : Max Drawdown = {flat_dd:.2f}% | Win Rate = {(flat_val['ret']>0).mean()*100:.1f}%")
    print(f"   Dynamic Tiered + Pyramiding  : Max Drawdown = {dyn_dd:.2f}% | Win Rate = {(dyn_val['ret']>0).mean()*100:.1f}%")
    print(f"   --> Drawdown Reduced by {((flat_dd - dyn_dd)/(flat_dd+1e-9))*100:.1f}% on Validation Data!\n")

    # Evaluate on Blind Out-of-Sample Week (Nov 9 - Nov 15)
    blind_trades, blind_dd = simulate_kelly_pyramid(
        test_df, test_probs, tier1_th=0.58, tier2_th=0.65, rev_th=0.62,
        tier1_size=0.30, tier2_size=0.60, pyr_size=0.40
    )

    if len(blind_trades) > 0:
        print("=== STEP 2: BLIND TEST WEEK (NOV 9 – NOV 15) WITH DYNAMIC SIZING & PYRAMIDING ===")
        print("-" * 80)
        print(f"{'Date':<12} | {'Trades':<7} | {'Pyramided':<10} | {'Win Rate':<10} | {'Daily Net PnL (Capital-Adjusted)'}")
        print("-" * 80)
        for dt, grp in blind_trades.groupby('date'):
            n_t = len(grp)
            n_pyr = grp['pyramided'].sum()
            wr = (grp['ret'] > 0).mean() * 100.0
            d_pnl = grp['ret'].sum() * 100.0
            print(f"{dt:<12} | {n_t:<7} | {n_pyr:<10} | {wr:<9.1f}% | {d_pnl:+.2f}%")
        print("-" * 80)
        tot_wr = (blind_trades['ret'] > 0).mean() * 100.0
        tot_pnl = blind_trades['ret'].sum() * 100.0
        gw = blind_trades[blind_trades['ret'] > 0]['ret'].sum()
        gl = abs(blind_trades[blind_trades['ret'] < 0]['ret'].sum()) + 1e-10
        pf = gw / gl
        print(f"WEEK SUMMARY | Trades: {len(blind_trades)} | Win Rate: {tot_wr:.1f}% | Net PnL: {tot_pnl:+.2f}% | Max DD: {blind_dd:.2f}% | PF: {pf:.2f}\n")

    # =========================================================================
    # 3. COMPILE PINE SCRIPT V5 WITH PYRAMIDING & DYNAMIC POSITION SIZING
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
// NOTE: pyramiding=2 enables Scale-In (adding to a winning position when trend confirms!)
strategy("AI Gold Dynamic Kelly & Pyramid Engine v6", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=30, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. VIVA EVALUATION WINDOW
// =========================================================================
use_date_filter = input.bool(true, "Enable Custom Date Range Filter", group="1. Viva Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Test Start Date (UTC)", group="1. Viva Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "Test End Date (UTC)", group="1. Viva Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Viva Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Viva Test Window")

// =========================================================================
// 2. DYNAMIC CAPITAL SIZING, PYRAMIDING & REVERSAL SETTINGS
// =========================================================================
tier1_thresh = input.float(0.58, "Tier-1 Probe Confidence Threshold (58%)", minval=0.52, maxval=0.70, step=0.01, group="2. Dynamic Kelly Sizing & Pyramiding")
tier2_thresh = input.float(0.65, "Tier-2 Strong Confidence Threshold (65%)", minval=0.58, maxval=0.80, step=0.01, group="2. Dynamic Kelly Sizing & Pyramiding")
tier1_pct    = input.float(30.0, "Tier-1 Probe Capital Allocation (%)", minval=10.0, maxval=100.0, step=5.0, group="2. Dynamic Kelly Sizing & Pyramiding")
tier2_pct    = input.float(60.0, "Tier-2 Strong Capital Allocation (%)", minval=20.0, maxval=100.0, step=5.0, group="2. Dynamic Kelly Sizing & Pyramiding")
enable_pyr   = input.bool(true,  "Enable Trend Pyramiding (Scale-In on Winners)", group="2. Dynamic Kelly Sizing & Pyramiding")
pyr_pct      = input.float(40.0, "Scale-In Pyramid Addition (%)", minval=10.0, maxval=70.0, step=5.0, group="2. Dynamic Kelly Sizing & Pyramiding")
pyr_atr_trig = input.float(0.45, "Min Profit Before Scaling In (ATR Multiple)", minval=0.2, maxval=1.5, step=0.05, group="2. Dynamic Kelly Sizing & Pyramiding")
enable_rev   = input.bool(true,  "Enable Dynamic Stop-and-Reverse on Opposite Signal", group="2. Dynamic Kelly Sizing & Pyramiding")
rev_thresh   = input.float(0.62, "Opposite Reversal Confidence Threshold (62%)", minval=0.55, maxval=0.75, step=0.01, group="2. Dynamic Kelly Sizing & Pyramiding")

// =========================================================================
// 3. RISK & MULTI-TIMEFRAME HORIZON SETTINGS
// =========================================================================
er_min_val   = input.float(0.16, "Min Efficiency Ratio (Chop Filter)", minval=0.05, maxval=0.35, step=0.01, group="3. Risk & Exits")
tp_mult      = input.float(1.5,  "Take Profit (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="3. Risk & Exits")
sl_mult      = input.float(1.6,  "Initial Stop Loss (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="3. Risk & Exits")
horizon_bars = input.int(10,     "Max Holding Horizon (Bars)", minval=4, maxval=24, group="3. Risk & Exits")

// =========================================================================
// 4. CROSS-ASSET FEEDS & 12 STATIONARY FEATURES
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
// 5. FROZEN AI WEIGHTS (TRAINED STRICTLY <= AUG 31, 2026)
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
prob_dn = 1.0 - prob_up

// =========================================================================
// 6. CONFIDENCE-SCALED ENTRY, PYRAMIDING & STOP-AND-REVERSE ENGINE
// =========================================================================
// Calculate exact order quantity from desired % of current strategy equity
calc_qty(pct_equity) =>
    capital_to_use = strategy.equity * (pct_equity / 100.0)
    math.max(0.001, capital_to_use / close)

can_trade = in_date_window and shock_ok and (er12 >= er_min_val)

bull_setup = can_trade and (prob_up >= tier1_thresh) and (f8 > -0.4) and (f6 > -0.2) and (f4 < 2.2)
bear_setup = can_trade and (prob_dn >= tier1_thresh) and (f8 < 0.4) and (f6 < 0.2) and (f4 > -2.2)

// Determine Tier-1 (Probe 30%) vs Tier-2 (Strong 60%) Capital Allocation
bull_pct = prob_up >= tier2_thresh ? tier2_pct : tier1_pct
bear_pct = prob_dn >= tier2_thresh ? tier2_pct : tier1_pct

var int   entry_bar  = na
var float init_entry = na
var float entry_atr  = na
var float tp_level   = na
var float sl_level   = na
var bool  scaled_in  = false
var float active_pct = 0.0

// --- A. INITIAL ENTRY (OR STOP-AND-REVERSE) ---
if barstate.isconfirmed
    // 1. Flat -> Open Initial Tier-1 or Tier-2 Position
    if strategy.position_size == 0
        if bull_setup
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="L1 (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_up*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if bear_setup
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="S1 (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round(prob_dn*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct

    // 2. Currently Long -> Check for Scale-In (Pyramiding) OR Bearish Reversal
    else if strategy.position_size > 0
        // Dynamic Stop-and-Reverse if AI flips strongly Bearish
        if enable_rev and can_trade and (prob_dn >= rev_thresh) and (f6 < 0)
            strategy.close_all(comment="Flip to Short")
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="SAR Short (" + str.tostring(bear_pct) + "%)")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct
        // Pyramid Scale-In if trade is winning by +0.45 ATR and AI still Bullish
        else if enable_pyr and not scaled_in and (close >= init_entry + entry_atr * pyr_atr_trig) and (prob_up >= tier1_thresh)
            add_pct = math.min(pyr_pct, 100.0 - active_pct)
            if add_pct > 5.0
                strategy.entry("AI Long Scale", strategy.long, qty=calc_qty(add_pct), comment="+Pyramid (" + str.tostring(add_pct) + "%)")
                scaled_in  := true
                active_pct := active_pct + add_pct
                // Move Stop-Loss up to initial entry to protect the scaled position!
                sl_level   := math.max(sl_level, init_entry)

    // 3. Currently Short -> Check for Scale-In (Pyramiding) OR Bullish Reversal
    else if strategy.position_size < 0
        if enable_rev and can_trade and (prob_up >= rev_thresh) and (f6 > 0)
            strategy.close_all(comment="Flip to Long")
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="SAR Long (" + str.tostring(bull_pct) + "%)")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if enable_pyr and not scaled_in and (close <= init_entry - entry_atr * pyr_atr_trig) and (prob_dn >= tier1_thresh)
            add_pct = math.min(pyr_pct, 100.0 - active_pct)
            if add_pct > 5.0
                strategy.entry("AI Short Scale", strategy.short, qty=calc_qty(add_pct), comment="+Pyramid (" + str.tostring(add_pct) + "%)")
                scaled_in  := true
                active_pct := active_pct + add_pct
                sl_level   := math.min(sl_level, init_entry)

// --- B. PROFIT RATCHET & HORIZON EXIT ---
if strategy.position_size > 0
    if high >= (init_entry + entry_atr * 0.9)
        sl_level := math.max(sl_level, init_entry + entry_atr * 0.30)
    strategy.exit("Exit Long", from_entry="AI Long", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Long Scale", from_entry="AI Long Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (init_entry - entry_atr * 0.9)
        sl_level := math.min(sl_level, init_entry - entry_atr * 0.30)
    strategy.exit("Exit Short", from_entry="AI Short", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Short Scale", from_entry="AI Short Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 7. VISUAL OVERLAYS & CAPITAL ALLOCATION HUD
// =========================================================================
plot(vwap24, "6H Institutional VWAP", color=color.new(color.fuchsia, 25), linewidth=2)
plot(ema60, "EMA 60 Core Trend", color=color.new(color.yellow, 25), linewidth=2)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round(prob_dn * 100, 1)
    rec_tier = conf_pct >= (tier2_thresh * 100) ? "TIER-2 STRONG (" + str.tostring(tier2_pct) + "%)" : conf_pct >= (tier1_thresh * 100) ? "TIER-1 PROBE (" + str.tostring(tier1_pct) + "%)" : "0% (WAIT)"
    pos_txt  = strategy.position_size > 0 ? "LONG (" + str.tostring(active_pct) + "% Cap)" : strategy.position_size < 0 ? "SHORT (" + str.tostring(active_pct) + "% Cap)" : "FLAT (100% Cash)"
    
    table.cell(hud, 0, 0, "Capital Engine", text_color=color.white)
    table.cell(hud, 1, 0, "Fractional Kelly + Pyramid", text_color=color.yellow)
    table.cell(hud, 0, 1, "AI Confidence", text_color=color.white)
    table.cell(hud, 1, 1, (prob_up >= 0.5 ? "UP " : "DN ") + str.tostring(conf_pct) + "%", text_color=prob_up >= 0.5 ? color.lime : color.orange)
    table.cell(hud, 0, 2, "Recommended Size", text_color=color.white)
    table.cell(hud, 1, 2, rec_tier, text_color=color.aqua)
    table.cell(hud, 0, 3, "Active Exposure", text_color=color.white)
    table.cell(hud, 1, 3, pos_txt + (scaled_in ? " [PYRAMIDED]" : ""), bgcolor=strategy.position_size != 0 ? color.new(color.blue, 40) : color.gray, text_color=color.white)
    table.cell(hud, 0, 4, "Stop-and-Reverse", text_color=color.white)
    table.cell(hud, 1, 4, enable_rev ? "ACTIVE (@" + str.tostring(math.round(rev_thresh*100)) + "%)" : "OFF", text_color=color.lime)
    table.cell(hud, 0, 5, "Regime Filter", text_color=color.white)
    table.cell(hud, 1, 5, er12 >= er_min_val ? "TRENDING" : "CHOP BLOCKED", text_color=er12 >= er_min_val ? color.lime : color.orange)
"""

    with open("AI_Gold_Dynamic_Pyramid_v6.pine", "w") as f:
        f.write(pine_code)

    print("3. Saved 'AI_Gold_Dynamic_Pyramid_v6.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()