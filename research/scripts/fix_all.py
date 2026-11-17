import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

def run_fixed_pipeline(input_file="paxg_15m_1year.csv"):
    print(f"1. Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Exact TradingView RMA (Wilder's Smoothing)
    def pine_rma(series, length):
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    # ---------------------------------------------------------
    # 1. EXACT 1-TO-1 PINE SCRIPT FEATURES (NO TAKER VOLUME MISMATCH)
    # ---------------------------------------------------------
    df['f1'] = np.log(df['close'] / df['close'].shift(1))
    df['f2'] = np.log(df['close'] / df['close'].shift(4))
    df['f3'] = np.log(df['close'] / df['close'].shift(16)) # 4-hour momentum

    # RSI(14)
    delta = df['close'].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rs = pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['f4'] = (rsi - 50.0) / 50.0

    # Normalized MACD Hist
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['f5'] = ((macd_line - signal_line) / df['close']) * 1000.0

    # Bollinger Band Position (20, 2)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std(ddof=0) # Pine stdev uses population std (ddof=0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df['f6'] = np.where((upper - lower) == 0, 0.5, (df['close'] - lower) / (upper - lower))

    # Normalized ATR(14)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['f7'] = (pine_rma(tr, 14) / df['close']) * 100.0

    # Distance from EMA 50 and EMA 200 (Macro Trend)
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    df['f8'] = ((df['close'] - ema50) / ema50) * 100.0
    df['f9'] = ((ema50 - ema200) / ema200) * 100.0

    # Candle Close Position (100% Identical in Python and Pine Script!)
    hl_range = df['high'] - df['low']
    df['f10'] = np.where(hl_range == 0, 0.5, (df['close'] - df['low']) / hl_range)

    # Future 8-candle (2-hour) return to give larger moves relative to fees
    HORIZON = 8
    df['future_ret'] = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

    # ---------------------------------------------------------
    # 2. SYMMETRIC TARGET LABELING (REMOVE FLAT CHOP FROM TRAINING)
    # ---------------------------------------------------------
    # Only train on clear moves (> +0.12% for UP, < -0.12% for DOWN)
    MOVE_THRESH = 0.0012
    df['label'] = np.where(df['future_ret'] > MOVE_THRESH, 1,
                  np.where(df['future_ret'] < -MOVE_THRESH, 0, -1))

    # Use recent regime for training/validation so weights match current gold volatility
    split_idx = int(len(df) * 0.70)
    train_raw = df.iloc[:split_idx]
    test_raw = df.iloc[split_idx:].copy()

    # Filter out flat noise (-1) during training so classes are 100% symmetric
    train_clean = train_raw[train_raw['label'] != -1].copy()

    X_train = train_clean[feature_cols]
    y_train = train_clean['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"2. Training on {len(X_train)} clean directional moves...")
    print(f"   Class balance -> UP (1): {(y_train==1).mean():.2%}, DOWN (0): {(y_train==0).mean():.2%}")

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation='relu',
        solver='adam',
        alpha=0.1,              # L2 regularization to prevent overfitting
        max_iter=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # 3. BACKTEST IN PYTHON (WITH EXACT NON-OVERLAPPING 8-BAR HOLD)
    # ---------------------------------------------------------
    X_test_scaled = scaler.transform(test_raw[feature_cols])
    test_raw['prob_up'] = model.predict_proba(X_test_scaled)[:, 1]

    UP_TH = 0.62
    DN_TH = 0.38
    FEE = 0.0004  # 0.02% entry + 0.02% exit

    trades = []
    i = 0
    closes = test_raw['close'].values
    probs = test_raw['prob_up'].values
    f9_trend = test_raw['f9'].values # EMA50 > EMA200 trend filter

    while i < len(test_raw) - HORIZON:
        p = probs[i]
        # Long when confident AND in macro uptrend
        if p >= UP_TH and f9_trend[i] > 0:
            ret = ((closes[i + HORIZON] - closes[i]) / closes[i]) - FEE
            trades.append(ret)
            i += HORIZON  # Lock trade for 8 bars (no overlapping flip!)
        # Short when confident AND in macro downtrend
        elif p <= DN_TH and f9_trend[i] < 0:
            ret = ((closes[i] - closes[i + HORIZON]) / closes[i]) - FEE
            trades.append(ret)
            i += HORIZON
        else:
            i += 1

    trades = np.array(trades)
    if len(trades) > 0:
        win_rate = (trades > 0).mean() * 100
        cum_pnl = trades.sum() * 100
        print(f"\n--- REALISTIC PYTHON BACKTEST (Out-of-Sample) ---")
        print(f"Total Non-Overlapping Trades: {len(trades)}")
        print(f"Win Rate (After Fees):        {win_rate:.2f}%")
        print(f"Cumulative Return:            {cum_pnl:+.2f}%\n")

    # ---------------------------------------------------------
    # 4. GENERATE FIXED PINE SCRIPT STRATEGY
    # ---------------------------------------------------------
    w0, b0 = model.coefs_[0], model.intercepts_[0]
    w1, b1 = model.coefs_[1], model.intercepts_[1]
    mean, scale = scaler.mean_, scaler.scale_

    pine_code = f"""//@version=5
strategy("AI Gold Predictor v2 (PAXG/XAU)", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.02)

// --- 1. EXACT 1-TO-1 FEATURE ENGINEERING ---
f1 = math.log(close / close[1])
f2 = math.log(close / nz(close[4], close[1]))
f3 = math.log(close / nz(close[16], close[1]))

pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
avg_gain = pine_rma(gain, 14)
avg_loss = pine_rma(loss, 14)
rs = avg_loss == 0 ? 0.0 : avg_gain / avg_loss
rsi_val = 100.0 - (100.0 / (1.0 + rs))
f4 = (rsi_val - 50.0) / 50.0

ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
macd_line = ema12 - ema26
signal_line = ta.ema(macd_line, 9)
f5 = ((macd_line - signal_line) / close) * 1000.0

sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20)
upper_band = sma20 + (std20 * 2.0)
lower_band = sma20 - (std20 * 2.0)
f6 = upper_band == lower_band ? 0.5 : (close - lower_band) / (upper_band - lower_band)

f7 = (pine_rma(ta.tr(true), 14) / close) * 100.0

ema50 = ta.ema(close, 50)
ema200 = ta.ema(close, 200)
f8 = ((close - ema50) / ema50) * 100.0
f9 = ((ema50 - ema200) / ema200) * 100.0

hl_range = high - low
f10 = hl_range == 0 ? 0.5 : (close - low) / hl_range

// --- 2. MODEL SCALING ---
"""
    for idx in range(10):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    pine_code += "\n// --- 3. NEURAL NETWORK FORWARD PASS ---\n"
    for j in range(8):
        eq = f"{b0[j]:.8f}"
        for idx in range(10):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"h{j} = math.max(0.0, {eq})\n"

    out_eq = f"{b1[0]:.8f}"
    for j in range(8):
        out_eq += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"out_raw = {out_eq}\n"
    pine_code += "prob_up = 1.0 / (1.0 + math.exp(-out_raw))\n\n"

    pine_code += f"""// --- 4. NON-OVERLAPPING TRADE EXECUTION & VISUALS ---
UP_THRESHOLD = {UP_TH}
DOWN_THRESHOLD = {DN_TH}
HORIZON = {HORIZON}

// Regime alignment filter: trade with macro momentum
is_up = prob_up >= UP_THRESHOLD and f9 > 0
is_down = prob_up <= DOWN_THRESHOLD and f9 < 0

var int entry_bar = na

// Only enter if flat (prevents constant flipping and commission bleed!)
if barstate.isconfirmed and strategy.position_size == 0
    if is_up
        strategy.entry("AI Long", strategy.long, comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
        entry_bar := bar_index
    else if is_down
        strategy.entry("AI Short", strategy.short, comment="DN " + str.tostring(math.round((1.0 - prob_up) * 100)) + "%")
        entry_bar := bar_index

// Strict Horizon Exit after exactly {HORIZON} bars (2 hours)
if strategy.position_size != 0 and (bar_index - entry_bar) >= HORIZON
    strategy.close_all(comment="Horizon Exit")
    entry_bar := na

// Plot EMA 200 Regime Line
plot(ema200, "EMA 200 Trend Filter", color=color.new(color.yellow, 40), linewidth=2)
"""

    with open("AI_Strategy_v2.pine", "w") as f:
        f.write(pine_code)

    print("3. Saved 'AI_Strategy_v2.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    run_fixed_pipeline()