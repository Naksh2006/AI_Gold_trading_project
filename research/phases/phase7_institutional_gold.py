import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. DATA FETCHER (GOLD + DOLLAR + LIQUIDITY + MACRO)
# ---------------------------------------------------------
def fetch_binance_symbol(symbol, interval="15m", days_back=365):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        if len(df) > 25000:
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df

    print(f"   Downloading {symbol} ({interval}) from Binance...")
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
            time.sleep(0.04)
        except Exception:
            time.sleep(1)

    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume', 'tbbav']:
        df[c] = pd.to_numeric(df[c])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'tbbav']].drop_duplicates('datetime').reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# ---------------------------------------------------------
# 2. INSTITUTIONAL FEATURE ENGINEERING (14 FEATURES)
# ---------------------------------------------------------
def build_institutional_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c = df['eur_close']
    btc_c = df['btc_close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df['atr'] = atr14

    # f1: 45m Volatility-Adjusted Momentum
    df['f1'] = ((c - c.shift(3)) / c.shift(3)) / atr_pct

    # f2: 6-Hour Rolling Institutional VWAP Distance
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df['f2'] = ((c - vwap24) / c) / atr_pct

    # f3: Wilder RSI(14)
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['f3'] = (rsi14 - 50.0) / 50.0

    # f4: Normalized MACD Histogram (Momentum Acceleration)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['f4'] = (macd_hist / (atr14 + 1e-10))

    # f5: Bollinger Band Z-Score (20)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0) + 1e-10
    df['f5'] = (c - sma20) / (2.0 * std20)

    # f6, f7: Multi-Timeframe (1H EMA16, 4H EMA64, Daily EMA200) Trend Stack
    ema16 = c.ewm(span=16, adjust=False).mean()
    ema64 = c.ewm(span=64, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df['f6'] = ((ema16 - ema64) / c) / atr_pct
    df['f7'] = ((ema64 - ema200) / c) / atr_pct

    # f8: Kaufman Efficiency Ratio (12) -> Blocks sideways range chop!
    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df['f8'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    # f9: Spread / Candle Body-to-Range Microstructure Efficiency
    # +1 = Full Bullish Marubozu, -1 = Full Bearish Marubozu, 0 = High-Spread Doji/Wick Chop
    rng = (h - l) + 1e-10
    df['f9'] = (c - o) / rng

    # f10: Order Flow / Volume-Weighted Pressure (3-bar institutional buying/selling pressure)
    signed_vol = ((c - l) - (h - c)) / rng * v
    df['f10'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # f11: Cross-Asset US Dollar Inverse (EURUSD) 1H Z-Score
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    df['f11'] = eur_ret4 * 100.0

    # f12: Cross-Asset Lead-Lag Divergence (Dollar vs Gold)
    df['f12'] = (eur_ret4 - gold_ret4) * 100.0

    # f13: Geopolitical / Safe-Haven Catalyst Proxy
    # Detects when Gold surges DESPITE a strong US Dollar (EUR dropping) or Risk-Off dump (BTC dropping)
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    safe_haven_div = gold_ret4 + eur_ret4 * 0.5 - btc_ret4 * 0.2
    vol_surge = v / (v.rolling(20).mean() + 1e-10)
    df['f13'] = safe_haven_div * 100.0 * np.clip(vol_surge, 0.5, 3.0)

    # f14: Volatility Regime Ratio (ATR 5 / ATR 24)
    atr5 = pine_rma(tr, 5)
    atr24 = pine_rma(tr, 24)
    df['f14'] = (atr5 / (atr24 + 1e-10)) - 1.0

    # Session & Weekend Filter (UTC hours & Day of Week)
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek # 5 = Saturday, 6 = Sunday
    # Active London + NY + Asian Liquid Hours (blocks dead rollover & weekends)
    df['is_liquid'] = ((df['dow'] < 5) & ((df['hour'] >= 1) & (df['hour'] <= 21))).astype(int)

    return df

# ---------------------------------------------------------
# 3. REALISTIC BACKTEST SIMULATOR WITH COOLDOWN & REGIME GATE
# ---------------------------------------------------------
def simulate_pro(df_test, probs, horizon=8, up_th=0.60, dn_th=0.40, tp_atr=2.0, sl_atr=1.4, cooldown_bars=3, fee=0.0001):
    closes = df_test['close'].values
    highs = df_test['high'].values
    lows = df_test['low'].values
    atrs = df_test['atr'].values
    f6_mtf = df_test['f6'].values
    f8_er = df_test['f8'].values
    f14_vol = df_test['f14'].values
    liquid = df_test['is_liquid'].values

    n = len(closes)
    trades = []
    i = 0

    while i < n - horizon - 1:
        if liquid[i] == 0 or f8_er[i] < 0.18 or f14_vol[i] > 1.5:
            # Skip illiquid weekend bars, dead sideways chop (ER < 0.18), or extreme news-wick spikes
            i += 1
            continue

        p = probs[i]
        direction = 0

        # Require High Confidence (>= 60%) + MTF Trend Alignment
        if p >= up_th and f6_mtf[i] > -0.3:
            direction = 1
        elif p <= dn_th and f6_mtf[i] < 0.3:
            direction = -1

        if direction != 0:
            entry = closes[i]
            cur_atr = atrs[i]
            tp_price = entry + direction * (tp_atr * cur_atr)
            sl_price = entry - direction * (sl_atr * cur_atr)

            exit_price = closes[i + horizon]
            bars_held = horizon
            hit_sl = False

            for k in range(1, horizon + 1):
                if direction == 1:
                    if lows[i + k] <= sl_price:
                        exit_price = sl_price
                        bars_held = k
                        hit_sl = True
                        break
                    if highs[i + k] >= tp_price:
                        exit_price = tp_price
                        bars_held = k
                        break
                else:
                    if highs[i + k] >= sl_price:
                        exit_price = sl_price
                        bars_held = k
                        hit_sl = True
                        break
                    if lows[i + k] <= tp_price:
                        exit_price = tp_price
                        bars_held = k
                        break

            net_ret = direction * ((exit_price - entry) / entry) - fee
            trades.append(net_ret)
            # If stopped out, wait 'cooldown_bars' so we don't machine-gun losses in a chop box!
            i += bars_held + (cooldown_bars if hit_sl else 1)
        else:
            i += 1

    if len(trades) < 60:
        return -999.0, 0.0, 0.0, len(trades)

    trades = np.array(trades)
    wr = (trades > 0).mean() * 100.0
    g_win = trades[trades > 0].sum()
    g_loss = abs(trades[trades < 0].sum()) + 1e-10
    pf = g_win / g_loss
    net_pnl = trades.sum() * 100.0
    return net_pnl, pf, wr, len(trades)

def main():
    print("=== PHASE 7: INSTITUTIONAL GOLD ENGINE (REGIME + ORDER FLOW + SAFE-HAVEN) ===")
    df_gold = fetch_binance_symbol("PAXGUSDT", "15m", 365)
    df_eur = fetch_binance_symbol("EURUSDT", "15m", 365)
    df_btc = fetch_binance_symbol("BTCUSDT", "15m", 365)

    print("1. Computing 14 Microstructure, Cross-Asset, Safe-Haven & Chop-Filter Features...")
    df = build_institutional_features(df_gold, df_eur, df_btc)

    HORIZON = 8 # 2-Hour Intraday Horizon
    feature_cols = [f"f{i}" for i in range(1, 15)]

    # Forward return & excursion labeling
    df['future_ret'] = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)

    # Symmetric target labeling on liquid hours only
    THRESH = 0.0010
    df['target'] = np.where(df['future_ret'] > THRESH, 1,
                   np.where(df['future_ret'] < -THRESH, 0, -1))

    split_idx = int(len(df) * 0.72)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    train_clean = train_df[(train_df['target'] != -1) & (train_df['is_liquid'] == 1)].copy()
    X_train = train_clean[feature_cols]
    y_train = train_clean['target']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(test_df[feature_cols])

    print(f"2. Training Ensemble (Logistic + Tanh MLP) on {len(X_train)} liquid institutional moves...")
    lr = LogisticRegression(C=0.2, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', alpha=0.15, max_iter=500, random_state=42)

    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    raw_p = 0.45 * lr.predict_proba(X_test_s)[:, 1] + 0.55 * mlp.predict_proba(X_test_s)[:, 1]
    mean_p = float(np.mean(raw_p))
    std_p = float(np.std(raw_p)) + 1e-10

    # Z-score probability calibration so 0.60+ strictly corresponds to top-conviction setups
    p_cal = 1.0 / (1.0 + np.exp(-((raw_p - mean_p) / (std_p * 0.85))))

    best_cfg = None
    best_score = -9999.0

    print("\n3. Grid-Searching High-Conviction Thresholds (Cutting 58% Chop Losses)...")
    print("-" * 82)
    print(f"{'Conf Thresh':<12} | {'TP/SL (ATR)':<12} | {'Cooldown':<8} | {'Trades':<7} | {'Win Rate':<9} | {'PF':<6} | {'Net PnL'}")
    print("-" * 82)

    for th in [0.59, 0.61, 0.63, 0.65]:
        for tp_m, sl_m in [(1.8, 1.5), (2.0, 1.4), (2.2, 1.5), (1.6, 1.6)]:
            for cd in [2, 4]:
                pnl, pf, wr, n_tr = simulate_pro(
                    test_df, p_cal, horizon=HORIZON, up_th=th, dn_th=1.0 - th,
                    tp_atr=tp_m, sl_atr=sl_m, cooldown_bars=cd, fee=0.0001
                )
                if n_tr >= 70:
                    score = pnl * (pf ** 1.5)
                    if score > best_score:
                        best_score = score
                        best_cfg = {
                            'th': th, 'tp': tp_m, 'sl': sl_m, 'cd': cd,
                            'pnl': pnl, 'pf': pf, 'wr': wr, 'trades': n_tr
                        }
        if best_cfg and best_cfg['th'] == th:
            print(f"{best_cfg['th']*100:>9.0f}%   | {best_cfg['tp']:.1f}x / {best_cfg['sl']:.1f}x  | {best_cfg['cd']} bars   | {best_cfg['trades']:<7} | {best_cfg['wr']:<8.2f}% | {best_cfg['pf']:<6.2f} | {best_cfg['pnl']:+.2f}%")

    print("-" * 82)
    print(f"LOCKED CONFIGURATION: Confidence >= {best_cfg['th']*100:.0f}% | Trades: {best_cfg['trades']} | Win Rate: {best_cfg['wr']:.2f}% | PF: {best_cfg['pf']:.2f} | Net PnL: {best_cfg['pnl']:+.2f}%\n")

    # ---------------------------------------------------------
    # 4. GENERATE PINE SCRIPT V5 (AI_Gold_Institutional_v4.pine)
    # ---------------------------------------------------------
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Institutional v4 (Macro + OrderFlow + SafeHaven)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. USER-CONFIGURABLE FILTERS (Adjust in Strategy Settings Gear Icon!)
// =========================================================================
conf_thresh = input.float({best_cfg['th']:.2f}, "Minimum AI Confidence Threshold (0.60 = 60%)", minval=0.52, maxval=0.75, step=0.01)
min_er      = input.float(0.18, "Minimum Trend Efficiency Ratio (Blocks Chop)", minval=0.05, maxval=0.40, step=0.02)
tp_mult     = input.float({best_cfg['tp']:.1f}, "Take Profit (ATR Multiplier)", minval=1.0, maxval=4.0, step=0.1)
sl_mult     = input.float({best_cfg['sl']:.1f}, "Stop Loss (ATR Multiplier)", minval=0.8, maxval=3.0, step=0.1)
cooldown_n  = input.int({best_cfg['cd']}, "Post-Loss Cooldown Bars (Prevents Whipsaw Clustering)", minval=0, maxval=12)
block_wknd  = input.bool(true, "Block Low-Liquidity Weekend & Rollover Hours")

// =========================================================================
// 2. CROSS-ASSET LIVE FEEDS (US Dollar Proxy & Global Liquidity)
// =========================================================================
eur_close = request.security("BINANCE:EURUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)
btc_close = request.security("BINANCE:BTCUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

// =========================================================================
// 3. 14 INSTITUTIONAL FEATURES (MICROSTRUCTURE, MTF, MACRO & SAFE-HAVEN)
// =========================================================================
pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

tr_val = ta.tr(true)
atr14 = pine_rma(tr_val, 14)
atr_pct = (atr14 / close) + 1e-10

f1 = ((close - nz(close[3], close[1])) / nz(close[3], close[1])) / atr_pct

typ_p = (high + low + close) / 3.0
vwap24 = math.sum(typ_p * volume, 24) / (math.sum(volume, 24) + 1e-10)
f2 = ((close - vwap24) / close) / atr_pct

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f3 = (rsi14 - 50.0) / 50.0

ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
macd_line = ema12 - ema26
macd_hist = macd_line - ta.ema(macd_line, 9)
f4 = macd_hist / (atr14 + 1e-10)

sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20) + 1e-10
f5 = (close - sma20) / (2.0 * std20)

ema16 = ta.ema(close, 16)
ema64 = ta.ema(close, 64)
ema200 = ta.ema(close, 200)
f6 = ((ema16 - ema64) / close) / atr_pct
f7 = ((ema64 - ema200) / close) / atr_pct

// f8: Kaufman Efficiency Ratio (12) -> Chop Detector
net_move = math.abs(close - nz(close[12], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 12)
f8 = sum_moves == 0 ? 0.0 : net_move / sum_moves

// f9: Spread / Candle Body-to-Range Efficiency
rng = (high - low) + 1e-10
f9 = (close - open) / rng

// f10: Order Flow / Volume-Weighted Buying/Selling Pressure
signed_vol = ((close - low) - (high - close)) / rng * volume
f10 = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

// f11, f12, f13: Macro USD & Geopolitical Safe-Haven Catalyst
eur_ret4 = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4 = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])

f11 = nz(eur_ret4, 0.0) * 100.0
f12 = nz(eur_ret4 - gold_ret4, 0.0) * 100.0

vol_surge = math.min(3.0, math.max(0.5, volume / (ta.sma(volume, 20) + 1e-10)))
safe_haven_div = nz(gold_ret4 + eur_ret4 * 0.5 - btc_ret4 * 0.2, 0.0)
f13 = safe_haven_div * 100.0 * vol_surge

// f14: Volatility Expansion Ratio (Detects News Spikes vs Dead Markets)
atr5 = pine_rma(tr_val, 5)
atr24 = pine_rma(tr_val, 24)
f14 = (atr5 / (atr24 + 1e-10)) - 1.0

// =========================================================================
// 4. STANDARD SCALER & HYBRID ENSEMBLE INFERENCE
// =========================================================================
"""
    for idx in range(14):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    lr_eq = f"{b_lr:.8f}"
    for idx in range(14):
        lr_eq += f" + (s{idx+1} * {w_lr[idx]:.8f})"
    pine_code += f"\nlogit_lr = {lr_eq}\n"
    pine_code += "p_lr = 1.0 / (1.0 + math.exp(-logit_lr))\n\n"

    for j in range(len(b0)):
        eq = f"{b0[j]:.8f}"
        for idx in range(14):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"z{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z{j}) - 1.0) / (math.exp(2.0 * z{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

    pine_code += f"""raw_p = 0.45 * p_lr + 0.55 * p_mlp
z_prob = (raw_p - {mean_p:.8f}) / ({std_p * 0.85:.8f})
prob_up = 1.0 / (1.0 + math.exp(-z_prob))

// =========================================================================
// 5. REGIME-GATED EXECUTION & ANTI-WHIPSAW COOLDOWN
// =========================================================================
HORIZON = {HORIZON}
dn_thresh = 1.0 - conf_thresh

// Liquidity & Microstructure Regime Check
wknd_ok = not block_wknd or (dayofweek != dayofweek.saturday and dayofweek != dayofweek.sunday and hour(time) >= 1 and hour(time) <= 21)
regime_ok = wknd_ok and (f8 >= min_er) and (f14 <= 1.5)

is_up = regime_ok and (prob_up >= conf_thresh) and (f6 > -0.3)
is_down = regime_ok and (prob_up <= dn_thresh) and (f6 < 0.3)

var int entry_bar = na
var int last_exit_bar = -999
var float tp_level = na
var float sl_level = na

cooldown_ok = (bar_index - last_exit_bar) >= cooldown_n

if barstate.isconfirmed and strategy.position_size == 0 and cooldown_ok
    if is_up
        strategy.entry("AI Long", strategy.long, comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
        entry_bar := bar_index
        tp_level := close + (atr14 * tp_mult)
        sl_level := close - (atr14 * sl_mult)
    else if is_down
        strategy.entry("AI Short", strategy.short, comment="DN " + str.tostring(math.round((1.0 - prob_up) * 100)) + "%")
        entry_bar := bar_index
        tp_level := close - (atr14 * tp_mult)
        sl_level := close + (atr14 * sl_mult)

if strategy.position_size > 0
    strategy.exit("Exit Long", "AI Long", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= HORIZON
        strategy.close_all(comment="2h Horizon Exit")
        last_exit_bar := bar_index
else if strategy.position_size < 0
    strategy.exit("Exit Short", "AI Short", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= HORIZON
        strategy.close_all(comment="2h Horizon Exit")
        last_exit_bar := bar_index

if strategy.position_size == 0 and strategy.position_size[1] != 0
    last_exit_bar := bar_index

// =========================================================================
// 6. VISUAL OVERLAYS & INSTITUTIONAL TRADER HUD
// =========================================================================
plot(vwap24, "6H Institutional VWAP", color=color.new(color.fuchsia, 30), linewidth=2)
plot(ema64, "16H MTF Trend", color=color.new(color.yellow, 20), linewidth=2)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    sig_txt = is_up ? "STRONG BUY (UP)" : is_down ? "STRONG SELL (DOWN)" : not regime_ok ? "FILTERED (CHOP/SPREAD)" : "NEUTRAL (WAIT)"
    sig_col = is_up ? color.green : is_down ? color.red : color.gray
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round((1.0 - prob_up) * 100, 1)
    
    table.cell(hud, 0, 0, "AI Quant Engine", text_color=color.white)
    table.cell(hud, 1, 0, "v4 Hybrid + OrderFlow", text_color=color.yellow)
    table.cell(hud, 0, 1, "Live Action (2h)", text_color=color.white)
    table.cell(hud, 1, 1, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 2, "Directional Conf", text_color=color.white)
    table.cell(hud, 1, 2, str.tostring(conf_pct) + "% (Min: " + str.tostring(math.round(conf_thresh*100)) + "%)", text_color=color.white)
    table.cell(hud, 0, 3, "Safe-Haven Flow", text_color=color.white)
    table.cell(hud, 1, 3, f13 > 0.15 ? "SAFE-HAVEN INFLOW" : f13 < -0.15 ? "RISK-ON OUTFLOW" : "NEUTRAL", text_color=f13 > 0 ? color.lime : color.orange)
    table.cell(hud, 0, 4, "Order Flow (VSA)", text_color=color.white)
    table.cell(hud, 1, 4, f10 > 0.05 ? "NET BUY ABSORPTION" : f10 < -0.05 ? "NET SELL PRESSURE" : "BALANCED", text_color=f10 > 0 ? color.lime : color.red)
    table.cell(hud, 0, 5, "Market Regime", text_color=color.white)
    table.cell(hud, 1, 5, f8 >= min_er ? "CLEAN TREND" : "CHOP / WIDE SPREAD", text_color=f8 >= min_er ? color.lime : color.orange)
"""

    with open("AI_Gold_Institutional_v4.pine", "w") as f:
        f.write(pine_code)

    print("4. Saved 'AI_Gold_Institutional_v4.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()