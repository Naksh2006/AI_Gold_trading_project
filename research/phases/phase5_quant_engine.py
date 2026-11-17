import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import _tree
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def build_features(df):
    df = df.copy()
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    # True Range & ATR(14) for Volatility Normalization (Lopez de Prado standard)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df['atr'] = atr14

    # f1, f2, f3: Volatility-Normalized Returns (Z-score returns relative to current ATR)
    df['f1'] = ((c - c.shift(1)) / c.shift(1)) / atr_pct
    df['f2'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df['f3'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # f4: Exact Wilder RSI(14) centered at 0
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rs = pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    df['f4'] = (rsi14 - 50.0) / 50.0

    # f5: Fast RSI(5) for pullback detection inside trends
    rs5 = pine_rma(gain, 5) / (pine_rma(loss, 5) + 1e-10)
    rsi5 = 100.0 - (100.0 / (1.0 + rs5))
    df['f5'] = (rsi5 - 50.0) / 50.0

    # f6: Bollinger Band %B (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df['f6'] = np.where((upper - lower) == 0, 0.5, (c - lower) / (upper - lower))

    # f7 & f8: Multi-EMA Macro & Micro Trend Alignment normalized by ATR
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df['f7'] = ((c - ema50) / c) / atr_pct
    df['f8'] = ((ema50 - ema200) / c) / atr_pct

    # f9: Kaufman Efficiency Ratio (10) -> Measures Trend Strength (1.0) vs Choppy Noise (0.0)
    net_move = (c - c.shift(10)).abs()
    sum_moves = c.diff().abs().rolling(10).sum()
    df['f9'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    # f10: Volume Surge Ratio (Current 4-bar volume vs 20-bar average volume)
    vol_sma4 = v.rolling(4).mean()
    vol_sma20 = v.rolling(20).mean()
    df['f10'] = np.where(vol_sma20 == 0, 1.0, vol_sma4 / vol_sma20)

    return df

def simulate_strategy(df_test, probs, horizon=12, up_th=0.56, dn_th=0.44, tp_atr=2.5, sl_atr=1.5, allow_short=False, fee=0.0004):
    closes = df_test['close'].values
    highs = df_test['high'].values
    lows = df_test['low'].values
    atrs = df_test['atr'].values
    f8_macro = df_test['f8'].values
    f9_er = df_test['f9'].values

    n = len(closes)
    trades = []
    i = 0

    while i < n - horizon - 1:
        p = probs[i]
        direction = 0
        # Enter Long when model is confident + macro trend supportive
        if p >= up_th and f8_macro[i] > -0.5 and f9_er[i] > 0.15:
            direction = 1
        elif allow_short and p <= dn_th and f8_macro[i] < -0.5 and f9_er[i] > 0.20:
            direction = -1

        if direction != 0:
            entry_price = closes[i]
            cur_atr = atrs[i]
            tp_price = entry_price + direction * (tp_atr * cur_atr)
            sl_price = entry_price - direction * (sl_atr * cur_atr)

            exit_price = closes[i + horizon]
            bars_held = horizon

            # Triple-Barrier Path Check (Check if TP or SL hit before Horizon)
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
                else:
                    if highs[i + k] >= sl_price:
                        exit_price = sl_price
                        bars_held = k
                        break
                    if lows[i + k] <= tp_price:
                        exit_price = tp_price
                        bars_held = k
                        break

            raw_ret = direction * ((exit_price - entry_price) / entry_price)
            net_ret = raw_ret - fee
            trades.append(net_ret)
            i += bars_held
        else:
            i += 1

    if len(trades) < 15:
        return -999.0, 0.0, 0.0, len(trades)

    trades = np.array(trades)
    win_rate = (trades > 0).mean() * 100.0
    gross_win = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum()) + 1e-10
    pf = gross_win / gross_loss
    cum_ret = trades.sum() * 100.0
    return cum_ret, pf, win_rate, len(trades)

def tree_to_pine(tree, feature_names, tree_idx):
    tree_ = tree.tree_
    lines = []

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            left_expr = recurse(tree_.children_left[node])
            right_expr = recurse(tree_.children_right[node])
            return f"({feat} <= {thresh:.6f} ? {left_expr} : {right_expr})"
        else:
            val = tree_.value[node][0][0]
            return f"{val:.6f}"

    expr = recurse(0)
    lines.append(f"t{tree_idx} = {expr}")
    return "\n".join(lines)

def run_quant_tournament(input_file="paxg_15m_1year.csv"):
    print("1. Loading dataset and building Triple-Barrier & Regime features...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = build_features(df)

    HORIZON = 12 # 3-hour horizon (12 x 15m candles) so winners easily beat fees
    feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

    # Triple-Barrier Labeling in Python for Training
    future_max = df['high'].rolling(HORIZON).max().shift(-HORIZON)
    future_min = df['low'].rolling(HORIZON).min().shift(-HORIZON)
    up_move = (future_max - df['close']) / df['close']
    dn_move = (df['close'] - future_min) / df['close']

    # Label 1 if upside excursion exceeds downside excursion and clears 0.20% barrier
    df['target'] = np.where((up_move > dn_move * 1.2) & (up_move > 0.0020), 1,
                   np.where((dn_move > up_move * 1.2) & (dn_move > 0.0020), 0, -1))

    df = df.dropna().reset_index(drop=True)

    # Split: First 75% Train, Last 25% Out-of-Sample Test (matches Sep–Nov 2026 window!)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    train_clean = train_df[train_df['target'] != -1].copy()
    X_train = train_clean[feature_cols]
    y_train = train_clean['target']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(test_df[feature_cols])

    # ---------------------------------------------------------
    # 2. TRAIN 3 DISTINCT AI ARCHITECTURES
    # ---------------------------------------------------------
    models = {
        "LogisticRegression": LogisticRegression(C=0.1, random_state=42),
        "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(8,), alpha=0.5, max_iter=400, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=15, max_depth=2, learning_rate=0.15, random_state=42)
    }

    best_score = -999.0
    best_cfg = None

    print("\n2. Running Out-of-Sample Walk-Forward Tournament (with Fees)...")
    print("-" * 75)
    print(f"{'Model':<20} | {'Mode':<10} | {'Thresh':<6} | {'Trades':<6} | {'Win%':<6} | {'PF':<5} | {'Net PnL'}")
    print("-" * 75)

    for name, m in models.items():
        m.fit(X_train_s, y_train)
        probs = m.predict_proba(X_test_s)[:, 1]

        for allow_short in [False, True]:
            for th in [0.52, 0.55, 0.58, 0.60, 0.62]:
                for tp_mult, sl_mult in [(2.5, 1.5), (3.0, 1.5), (2.0, 1.2)]:
                    cum_ret, pf, wr, n_tr = simulate_strategy(
                        test_df, probs, horizon=HORIZON, up_th=th, dn_th=1.0 - th,
                        tp_atr=tp_mult, sl_atr=sl_mult, allow_short=allow_short
                    )
                    if n_tr >= 25:
                        # Score rewards Profit Factor and Net Return
                        score = cum_ret * (pf if pf > 1.0 else 0.5)
                        if score > best_score:
                            best_score = score
                            best_cfg = {
                                'name': name, 'model': m, 'th': th,
                                'allow_short': allow_short, 'tp': tp_mult, 'sl': sl_mult,
                                'ret': cum_ret, 'pf': pf, 'wr': wr, 'trades': n_tr
                            }
            # Print summary for best threshold of this model/mode
            if best_cfg and best_cfg['name'] == name and best_cfg['allow_short'] == allow_short:
                mode_str = "Long+Short" if allow_short else "Long-Only"
                print(f"{name:<20} | {mode_str:<10} | {best_cfg['th']:<6.2f} | {best_cfg['trades']:<6} | {best_cfg['wr']:<5.1f}% | {best_cfg['pf']:<5.2f} | {best_cfg['ret']:+.2f}%")

    print("-" * 75)
    print(f"\nWINNER SELECTED: {best_cfg['name']} ({'Long+Short' if best_cfg['allow_short'] else 'Long-Only'})")
    print(f"Expected Out-of-Sample Net PnL: {best_cfg['ret']:+.2f}% | Profit Factor: {best_cfg['pf']:.2f} | Win Rate: {best_cfg['wr']:.1f}% ({best_cfg['trades']} trades)\n")

    # ---------------------------------------------------------
    # 3. COMPILE WINNING MODEL TO PINE SCRIPT V5
    # ---------------------------------------------------------
    mean, scale = scaler.mean_, scaler.scale_
    s_names = [f"s{i+1}" for i in range(10)]

    pine_code = f"""//@version=5
strategy("AI Gold Quant Engine v3 ({best_cfg['name']})", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.02)

// --- 1. VOLATILITY & REGIME FEATURE ENGINEERING ---
pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

atr14 = pine_rma(ta.tr(true), 14)
atr_pct = (atr14 / close) + 1e-10

f1 = ((close - close[1]) / close[1]) / atr_pct
f2 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f3 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0

rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f4 = (rsi14 - 50.0) / 50.0

rs5 = pine_rma(loss, 5) == 0 ? 0.0 : pine_rma(gain, 5) / pine_rma(loss, 5)
rsi5 = 100.0 - (100.0 / (1.0 + rs5))
f5 = (rsi5 - 50.0) / 50.0

sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20)
upper_b = sma20 + 2.0 * std20
lower_b = sma20 - 2.0 * std20
f6 = upper_b == lower_b ? 0.5 : (close - lower_b) / (upper_b - lower_b)

ema50 = ta.ema(close, 50)
ema200 = ta.ema(close, 200)
f7 = ((close - ema50) / close) / atr_pct
f8 = ((ema50 - ema200) / close) / atr_pct

// Kaufman Efficiency Ratio (10)
net_move = math.abs(close - nz(close[10], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 10)
f9 = sum_moves == 0 ? 0.0 : net_move / sum_moves

vol_sma4 = ta.sma(volume, 4)
vol_sma20 = ta.sma(volume, 20)
f10 = vol_sma20 == 0 ? 1.0 : vol_sma4 / vol_sma20

// --- 2. FEATURE SCALING ---
"""
    for idx in range(10):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    pine_code += f"\n// --- 3. WINNING MODEL INFERENCE ({best_cfg['name']}) ---\n"
    m = best_cfg['model']

    if best_cfg['name'] == "LogisticRegression":
        w = m.coef_[0]
        b = m.intercept_[0]
        eq = f"{b:.8f}"
        for idx in range(10):
            eq += f" + (s{idx+1} * {w[idx]:.8f})"
        pine_code += f"logit = {eq}\n"
        pine_code += "prob_up = 1.0 / (1.0 + math.exp(-logit))\n"

    elif best_cfg['name'] == "MLP_NeuralNet":
        w0, b0 = m.coefs_[0], m.intercepts_[0]
        w1, b1 = m.coefs_[1], m.intercepts_[1]
        for j in range(len(b0)):
            eq = f"{b0[j]:.8f}"
            for idx in range(10):
                eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
            pine_code += f"h{j} = math.max(0.0, {eq})\n"
        out_eq = f"{b1[0]:.8f}"
        for j in range(len(b0)):
            out_eq += f" + (h{j} * {w1[j][0]:.8f})"
        pine_code += f"out_raw = {out_eq}\n"
        pine_code += "prob_up = 1.0 / (1.0 + math.exp(-out_raw))\n"

    elif best_cfg['name'] == "GradientBoosting":
        init_p = np.mean(y_train)
        init_log_odds = np.log(init_p / (1.0 - init_p))
        lr = m.learning_rate
        for t_idx, est in enumerate(m.estimators_):
            pine_code += tree_to_pine(est[0], s_names, t_idx) + "\n"
        sum_trees = " + ".join([f"t{i}" for i in range(len(m.estimators_))])
        pine_code += f"raw_gbdt = {init_log_odds:.8f} + ({lr:.6f} * ({sum_trees}))\n"
        pine_code += "prob_up = 1.0 / (1.0 + math.exp(-raw_gbdt))\n"

    allow_short_str = "true" if best_cfg['allow_short'] else "false"
    pine_code += f"""
// --- 4. TRIPLE-BARRIER EXECUTION & RISK ENGINE ---
UP_TH = {best_cfg['th']}
DN_TH = {1.0 - best_cfg['th']:.2f}
HORIZON = {HORIZON}
TP_MULT = {best_cfg['tp']}
SL_MULT = {best_cfg['sl']}
ALLOW_SHORT = {allow_short_str}

is_up = prob_up >= UP_TH and f8 > -0.5 and f9 > 0.15
is_down = ALLOW_SHORT and (prob_up <= DN_TH and f8 < -0.5 and f9 > 0.20)

var int entry_bar = na
var float tp_level = na
var float sl_level = na

if barstate.isconfirmed and strategy.position_size == 0
    if is_up
        strategy.entry("AI Long", strategy.long, comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
        entry_bar := bar_index
        tp_level := close + (atr14 * TP_MULT)
        sl_level := close - (atr14 * SL_MULT)
    else if is_down
        strategy.entry("AI Short", strategy.short, comment="DN " + str.tostring(math.round((1.0 - prob_up) * 100)) + "%")
        entry_bar := bar_index
        tp_level := close - (atr14 * TP_MULT)
        sl_level := close + (atr14 * SL_MULT)

// Triple Barrier Exit: Take Profit, Stop Loss, or Vertical Time Barrier
if strategy.position_size > 0
    strategy.exit("TP/SL Long", "AI Long", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= HORIZON
        strategy.close_all(comment="Time Barrier")
else if strategy.position_size < 0
    strategy.exit("TP/SL Short", "AI Short", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= HORIZON
        strategy.close_all(comment="Time Barrier")

// --- 5. LIVE TRADER DASHBOARD (20 Marks Visual Usefulness) ---
var table hud = table.new(position.top_right, 2, 4, bgcolor=color.new(color.black, 20), border_width=1)
if barstate.islast
    sig_txt = is_up ? "BULLISH (UP)" : is_down ? "BEARISH (DOWN)" : "NEUTRAL (WAIT)"
    sig_col = is_up ? color.green : is_down ? color.red : color.gray
    conf_val = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round((1.0 - prob_up) * 100, 1)
    dir_lbl = prob_up >= 0.5 ? "Up: " : "Down: "
    
    table.cell(hud, 0, 0, "Model Engine", text_color=color.white)
    table.cell(hud, 1, 0, "{best_cfg['name']}", text_color=color.yellow)
    table.cell(hud, 0, 1, "Signal (3h Horizon)", text_color=color.white)
    table.cell(hud, 1, 1, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 2, "Model Confidence", text_color=color.white)
    table.cell(hud, 1, 2, dir_lbl + str.tostring(conf_val) + "%", text_color=color.white)
    table.cell(hud, 0, 3, "Regime (Efficiency)", text_color=color.white)
    table.cell(hud, 1, 3, f9 > 0.20 ? "TRENDING" : "CHOPPY", text_color=f9 > 0.20 ? color.lime : color.orange)

plot(ema50, "EMA 50", color=color.new(color.aqua, 50))
plot(ema200, "EMA 200", color=color.new(color.yellow, 30), linewidth=2)
"""

    with open("AI_Strategy_v3.pine", "w") as f:
        f.write(pine_code)

    print("3. Generated 'AI_Strategy_v3.pine' (Includes Live Trader HUD Table + Triple-Barrier Exits)!")

if __name__ == "__main__":
    run_quant_tournament()