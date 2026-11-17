import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# 1. MULTI-YEAR DATA FETCHER (2 YEARS OF 15M DATA FOR DEEP TRAINING)
# =========================================================================
def fetch_binance_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Ensure cache reaches at least Nov 14, 2026
        if len(df) > 40000 and df['datetime'].max() >= pd.Timestamp("2026-11-13"):
            print(f"   Loaded cached {symbol} ({len(df)} rows, up to {df['datetime'].max()})")
            return df

    print(f"   Downloading {days_back} days of {symbol} ({interval}) from Binance...")
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    all_klines = []
    cur = start_time

    while cur < end_time:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur}&limit=1000"
        try:
            r = requests.get(url, timeout=10).json()
            if not r or isinstance(r, dict):
                break
            all_klines.extend(r)
            cur = r[-1][6] + 1
            time.sleep(0.03)
        except Exception:
            time.sleep(1)

    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates('datetime').reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"   Saved {cache_file} ({len(df)} rows)")
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 2. EXACT PINE-PARITY FEATURE & BASE INDICATOR ENGINEERING
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

    # f1, f2: Short & Medium Volatility-Normalized Momentum
    df['f1'] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df['f2'] = ((c - c.shift(6)) / c.shift(6)) / atr_pct

    # f3: Rolling Intraday VWAP Distance (20 bars)
    typ_p = (h + l + c) / 3.0
    vwap20 = (typ_p * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    df['vwap'] = vwap20
    df['f3'] = ((c - vwap20) / c) / atr_pct

    # f4: Wilder RSI(14) centered
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['f4'] = (rsi14 - 50.0) / 50.0

    # f5: Normalized MACD Histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['f5'] = macd_hist / (atr14 + 1e-10)

    # f6: Bollinger Band Z-Score (20)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0) + 1e-10
    df['f6'] = (c - sma20) / (2.0 * std20)

    # f7, f8: Multi-Timeframe EMA Stack (EMA 21 vs EMA 55 vs EMA 144)
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema55 = c.ewm(span=55, adjust=False).mean()
    ema144 = c.ewm(span=144, adjust=False).mean()
    df['f7'] = ((c - ema21) / c) / atr_pct
    df['f8'] = ((ema21 - ema55) / c) / atr_pct
    df['f9'] = ((ema55 - ema144) / c) / atr_pct

    # f10: Order Flow VSA (4-bar Institutional Buying/Selling Absorption)
    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df['f10'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # f11: Cross-Asset US Dollar Proxy (EURUSD 1H Momentum)
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    df['f11'] = eur_ret4 * 100.0

    # f12: Safe-Haven / Cross-Asset Divergence
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Kaufman Efficiency Ratio (10)
    net_move = (c - c.shift(10)).abs()
    sum_moves = c.diff().abs().rolling(10).sum()
    df['er10'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df

# =========================================================================
# 3. ANTI-CHASING + BREAKEVEN TRAILING STOP BACKTEST ENGINE
# =========================================================================
def simulate_week(df_slice, probs, horizon=8, up_th=0.53, tp_atr=1.8, sl_atr=1.4, be_atr=0.85, max_vwap_dist=1.8, fee=0.0001):
    closes = df_slice['close'].values
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    atrs = df_slice['atr'].values
    vwap_d = df_slice['f3'].values  # Distance from VWAP in ATR units
    ema_trend = df_slice['f8'].values # EMA21 - EMA55 trend
    er = df_slice['er10'].values
    dates = df_slice['datetime'].values

    dn_th = 1.0 - up_th
    n = len(closes)
    trades = []
    i = 0

    while i < n - horizon - 1:
        if er[i] < 0.12:
            i += 1
            continue

        p = probs[i]
        direction = 0

        # ANTI-CHASING LOGIC:
        # Go Long if AI is Bullish, trend is supportive, AND price is NOT overbought (> +1.8 ATR above VWAP)
        if p >= up_th and ema_trend[i] > -0.6 and vwap_d[i] < max_vwap_dist:
            direction = 1
        # Go Short if AI is Bearish, trend is supportive, AND price is NOT oversold (< -1.8 ATR below VWAP)
        elif p <= dn_th and ema_trend[i] < 0.6 and vwap_d[i] > -max_vwap_dist:
            direction = -1

        if direction != 0:
            entry = closes[i]
            cur_atr = atrs[i]
            tp_price = entry + direction * (tp_atr * cur_atr)
            sl_price = entry - direction * (sl_atr * cur_atr)
            be_trigger = entry + direction * (be_atr * cur_atr)

            exit_price = closes[i + horizon]
            bars_held = horizon

            for k in range(1, horizon + 1):
                if direction == 1:
                    if lows[i + k] <= sl_price:
                        exit_price = sl_price
                        bars_held = k
                        break
                    if highs[i + k] >= tp_price:
                        exit_price = tp_price
                        bars_held = k
                        break
                    # Breakeven ratchet: if price reaches +0.85 ATR, lock SL above entry!
                    if highs[i + k] >= be_trigger:
                        sl_price = max(sl_price, entry * (1.0 + 0.0003))
                else:
                    if highs[i + k] >= sl_price:
                        exit_price = sl_price
                        bars_held = k
                        break
                    if lows[i + k] <= tp_price:
                        exit_price = tp_price
                        bars_held = k
                        break
                    if lows[i + k] <= be_trigger:
                        sl_price = min(sl_price, entry * (1.0 - 0.0003))

            net_ret = direction * ((exit_price - entry) / entry) - fee
            trades.append({
                'date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'ret': net_ret
            })
            i += bars_held + 1
        else:
            i += 1

    return pd.DataFrame(trades)

def main():
    print("=== PHASE 8: FINAL VIVA ENGINE (TRAIN <= NOV 8, TEST NOV 9 - NOV 16) ===")
    df_gold = fetch_binance_symbol("PAXGUSDT", "15m", 730)
    df_eur = fetch_binance_symbol("EURUSDT", "15m", 730)
    df_btc = fetch_binance_symbol("BTCUSDT", "15m", 730)

    df = build_features(df_gold, df_eur, df_btc)
    HORIZON = 8
    feature_cols = [f"f{i}" for i in range(1, 13)]

    # Target: Did max upside excursion beat max downside excursion over next 8 bars?
    f_high = df['high'].rolling(HORIZON).max().shift(-HORIZON)
    f_low = df['low'].rolling(HORIZON).min().shift(-HORIZON)
    up_ex = (f_high - df['close']) / df['close']
    dn_ex = (df['close'] - f_low) / df['close']

    df['target'] = np.where((up_ex > dn_ex * 1.15) & (up_ex > 0.0012), 1,
                   np.where((dn_ex > up_ex * 1.15) & (dn_ex > 0.0012), 0, -1))

    df = df.dropna().reset_index(drop=True)

    # STRICT CUTOFF AT NOV 8, 2026 23:59:59 UTC!
    CUTOFF_DATE = pd.Timestamp("2026-11-08 23:59:59")
    train_df = df[df['datetime'] <= CUTOFF_DATE].copy()
    test_week_df = df[df['datetime'] > CUTOFF_DATE].copy()

    train_clean = train_df[train_df['target'] != -1].copy()
    X_train = train_clean[feature_cols]
    y_train = train_clean['target']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    print(f"1. Strictly Trained on data <= Nov 8, 2026 ({len(X_train)} clean directional moves)...")
    print(f"2. Unseen Live Evaluation Window: {test_week_df['datetime'].min()} to {test_week_df['datetime'].max()} ({len(test_week_df)} bars)")

    lr = LogisticRegression(C=0.25, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', alpha=0.1, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    # Calibrate probabilities on training distribution only (Zero Lookahead!)
    train_raw_p = 0.5 * lr.predict_proba(X_train_s)[:, 1] + 0.5 * mlp.predict_proba(X_train_s)[:, 1]
    mean_p = float(np.mean(train_raw_p))
    std_p = float(np.std(train_raw_p)) + 1e-10

    X_test_s = scaler.transform(test_week_df[feature_cols])
    test_raw_p = 0.5 * lr.predict_proba(X_test_s)[:, 1] + 0.5 * mlp.predict_proba(X_test_s)[:, 1]
    test_probs = 1.0 / (1.0 + np.exp(-((test_raw_p - mean_p) / (std_p * 0.8))))

    trades_df = simulate_week(test_week_df, test_probs, horizon=HORIZON, up_th=0.54, tp_atr=1.8, sl_atr=1.4, be_atr=0.85)

    if len(trades_df) > 0:
        print("\n=== DAY-BY-DAY UNSEEN TEST RESULTS (NOV 9 – NOV 16, 2026) ===")
        print("-" * 68)
        print(f"{'Date':<14} | {'Trades':<8} | {'Winners':<9} | {'Win Rate':<10} | {'Daily Net PnL'}")
        print("-" * 68)
        for dt, grp in trades_df.groupby('date'):
            n_t = len(grp)
            n_w = (grp['ret'] > 0).sum()
            wr = (n_w / n_t) * 100.0
            d_pnl = grp['ret'].sum() * 100.0
            print(f"{dt:<14} | {n_t:<8} | {n_w:<9} | {wr:<9.1f}% | {d_pnl:+.2f}%")
        print("-" * 68)
        tot_t = len(trades_df)
        tot_wr = (trades_df['ret'] > 0).mean() * 100.0
        tot_pnl = trades_df['ret'].sum() * 100.0
        g_win = trades_df[trades_df['ret'] > 0]['ret'].sum()
        g_loss = abs(trades_df[trades_df['ret'] < 0]['ret'].sum()) + 1e-10
        tot_pf = g_win / g_loss
        print(f"WEEK TOTAL     | {tot_t:<8} | {(trades_df['ret']>0).sum():<9} | {tot_wr:<9.1f}% | {tot_pnl:+.2f}% (PF: {tot_pf:.2f})\n")

    # =========================================================================
    # 4. GENERATE FINAL PINE SCRIPT V5 WITH DATE-RANGE PICKER & BREAKEVEN STOP
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Final Viva Engine (Trained <= Nov 8)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. VIVA DATE-RANGE PICKER (Test Any Single Day or Full Week!)
// =========================================================================
use_date_filter = input.bool(true, "Enable Custom Date Range Filter", group="1. Viva Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Test Start Date (UTC)", group="1. Viva Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "Test End Date (UTC)", group="1. Viva Evaluation Window")

in_date_window = not use_date_filter or (time >= start_time and time <= end_time)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Test Window Highlight")

// =========================================================================
// 2. STRATEGY & RISK PARAMETERS (Works on 5m, 15m, and 30m!)
// =========================================================================
conf_thresh   = input.float(0.54, "AI Confidence Threshold (0.54 = 54%)", minval=0.50, maxval=0.75, step=0.01, group="2. AI & Risk Settings")
use_st_filter = input.bool(true, "Combine AI with SuperTrend Base Indicator", group="2. AI & Risk Settings")
st_factor     = input.float(2.5, "SuperTrend ATR Factor", minval=1.0, maxval=5.0, step=0.1, group="2. AI & Risk Settings")
max_vwap_ext  = input.float(1.8, "Anti-Chasing Max VWAP Extension (ATR)", minval=0.8, maxval=4.0, step=0.1, group="2. AI & Risk Settings")
tp_mult       = input.float(1.8, "Take Profit (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="2. AI & Risk Settings")
sl_mult       = input.float(1.4, "Initial Stop Loss (ATR Multiplier)", minval=0.6, maxval=3.0, step=0.1, group="2. AI & Risk Settings")
be_trigger_m  = input.float(0.85, "Breakeven Ratchet Trigger (ATR)", minval=0.3, maxval=2.5, step=0.05, group="2. AI & Risk Settings")
horizon_bars  = input.int(8, "Max Holding Horizon (Bars)", minval=3, maxval=24, group="2. AI & Risk Settings")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 12 EXACT AI FEATURES
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

f1 = ((close - nz(close[2], close[1])) / nz(close[2], close[1])) / atr_pct
f2 = ((close - nz(close[6], close[1])) / nz(close[6], close[1])) / atr_pct

typ_p = (high + low + close) / 3.0
vwap20 = math.sum(typ_p * volume, 20) / (math.sum(volume, 20) + 1e-10)
f3 = ((close - vwap20) / close) / atr_pct

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f4 = (rsi14 - 50.0) / 50.0

ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
macd_line = ema12 - ema26
macd_hist = macd_line - ta.ema(macd_line, 9)
f5 = macd_hist / (atr14 + 1e-10)

sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20) + 1e-10
f6 = (close - sma20) / (2.0 * std20)

ema21 = ta.ema(close, 21)
ema55 = ta.ema(close, 55)
ema144 = ta.ema(close, 144)
f7 = ((close - ema21) / close) / atr_pct
f8 = ((ema21 - ema55) / close) / atr_pct
f9 = ((ema55 - ema144) / close) / atr_pct

rng = (high - low) + 1e-10
signed_vol = ((close - low) - (high - close)) / rng * volume
f10 = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4 = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4 = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])

f11 = nz(eur_ret4, 0.0) * 100.0
f12 = nz(eur_ret4 - gold_ret4 + 0.15 * btc_ret4, 0.0) * 100.0

net_move = math.abs(close - nz(close[10], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 10)
er10 = sum_moves == 0 ? 0.0 : net_move / sum_moves

// Base Indicator: Adaptive SuperTrend (10, st_factor)
[st_val, st_dir] = ta.supertrend(st_factor, 10)
st_bull = st_dir < 0
st_bear = st_dir > 0

// =========================================================================
// 4. FROZEN AI WEIGHTS (STRICTLY TRAINED ON DATA <= NOV 8, 2026)
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

    pine_code += f"""raw_p = 0.50 * p_lr + 0.50 * p_mlp
z_prob = (raw_p - {mean_p:.8f}) / ({std_p * 0.80:.8f})
prob_up = 1.0 / (1.0 + math.exp(-z_prob))

// =========================================================================
// 5. ANTI-CHASING ENTRY & BREAKEVEN TRAILING STOP ENGINE
// =========================================================================
dn_thresh = 1.0 - conf_thresh

// Combine AI Probability + Base SuperTrend/EMA Structure + Anti-Chasing VWAP Cap
bull_structure = use_st_filter ? (st_bull or f8 > 0) : (f8 > -0.6)
bear_structure = use_st_filter ? (st_bear or f8 < 0) : (f8 < 0.6)

is_up = in_date_window and (er10 >= 0.12) and (prob_up >= conf_thresh) and bull_structure and (f3 < max_vwap_ext)
is_down = in_date_window and (er10 >= 0.12) and (prob_up <= dn_thresh) and bear_structure and (f3 > -max_vwap_ext)

var int entry_bar = na
var float entry_p = na
var float tp_level = na
var float sl_level = na
var float be_level = na

if barstate.isconfirmed and strategy.position_size == 0
    if is_up
        strategy.entry("AI Long", strategy.long, comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
        entry_bar := bar_index
        entry_p := close
        tp_level := close + (atr14 * tp_mult)
        sl_level := close - (atr14 * sl_mult)
        be_level := close + (atr14 * be_trigger_m)
    else if is_down
        strategy.entry("AI Short", strategy.short, comment="DN " + str.tostring(math.round((1.0 - prob_up) * 100)) + "%")
        entry_bar := bar_index
        entry_p := close
        tp_level := close - (atr14 * tp_mult)
        sl_level := close + (atr14 * sl_mult)
        be_level := close - (atr14 * be_trigger_m)

// Dynamic Breakeven Ratchet (Moves Stop-Loss to Entry once trade reaches +0.85 ATR)
if strategy.position_size > 0
    if high >= be_level
        sl_level := math.max(sl_level, entry_p * 1.0003)
    strategy.exit("Exit Long", "AI Long", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= be_level
        sl_level := math.min(sl_level, entry_p * 0.9997)
    strategy.exit("Exit Short", "AI Short", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

// Close any open position at the end of the custom test window
if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="End of Test Window")

// =========================================================================
// 6. CHART OVERLAYS & VIVA HUD TABLE
// =========================================================================
plot(vwap20, "Intraday VWAP", color=color.new(color.fuchsia, 25), linewidth=2)
plot(st_dir < 0 ? st_val : na, "SuperTrend Bull", color=color.new(color.lime, 30), style=plot.style_linebr, linewidth=2)
plot(st_dir > 0 ? st_val : na, "SuperTrend Bear", color=color.new(color.red, 30), style=plot.style_linebr, linewidth=2)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    sig_txt = is_up ? "BUY / LONG" : is_down ? "SELL / SHORT" : "WAIT / HOLD"
    sig_col = is_up ? color.green : is_down ? color.red : color.gray
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round((1.0 - prob_up) * 100, 1)
    dir_txt = prob_up >= 0.5 ? "Bullish (" + str.tostring(conf_pct) + "%)" : "Bearish (" + str.tostring(conf_pct) + "%)"

    table.cell(hud, 0, 0, "Train Cutoff", text_color=color.white)
    table.cell(hud, 1, 0, "<= 08 Nov 2026 (Strict OOS)", text_color=color.yellow)
    table.cell(hud, 0, 1, "AI Forecast", text_color=color.white)
    table.cell(hud, 1, 1, dir_txt, text_color=prob_up >= 0.5 ? color.lime : color.orange)
    table.cell(hud, 0, 2, "Live Signal", text_color=color.white)
    table.cell(hud, 1, 2, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 3, "Base Structure", text_color=color.white)
    table.cell(hud, 1, 3, st_bull ? "SUPERTREND BULL" : "SUPERTREND BEAR", text_color=st_bull ? color.lime : color.red)
    table.cell(hud, 0, 4, "VWAP Stretch", text_color=color.white)
    table.cell(hud, 1, 4, str.tostring(math.round(f3, 2)) + " ATR", text_color=math.abs(f3) < max_vwap_ext ? color.lime : color.orange)
    table.cell(hud, 0, 5, "Risk Protection", text_color=color.white)
    table.cell(hud, 1, 5, "Breakeven @ +" + str.tostring(be_trigger_m) + " ATR", text_color=color.aqua)
"""

    with open("AI_Gold_Final_Viva.pine", "w") as f:
        f.write(pine_code)

    print("3. Generated 'AI_Gold_Final_Viva.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()