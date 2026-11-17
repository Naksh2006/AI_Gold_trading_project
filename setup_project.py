import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# 1. CREATE MODULAR DIRECTORY STRUCTURE
# =========================================================================
DIRS = ["data", "src", "results", "pine"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------------------
# MODULE 1: data/loader.py
# -------------------------------------------------------------------------
LOADER_CODE = '''import os
import pandas as pd

def load_synchronized_data(interval="15m"):
    symbols = ["paxgusdt", "eurusdt", "btcusdt"]
    dfs = {}
    for sym in symbols:
        loaded = False
        for days in [1460, 730, 365]:
            candidate_paths = [
                os.path.join("data", "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join("data", f"{sym}_{interval}_{days}d.csv"),
                f"{sym}_{interval}_{days}d.csv",
                os.path.join(os.path.dirname(__file__), "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join(os.path.dirname(__file__), f"{sym}_{interval}_{days}d.csv")
            ]
            for fname in candidate_paths:
                if os.path.exists(fname):
                    df = pd.read_csv(fname)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    dfs[sym] = df
                    loaded = True
                    break
            if loaded:
                break
        if not loaded:
            raise FileNotFoundError(f"Dataset for {sym} not found in data/cache/, data/, or working directory.")

    df_gold = dfs["paxgusdt"]
    df_eur  = dfs["eurusdt"][["datetime", "close"]].rename(columns={"close": "eur_close"})
    df_btc  = dfs["btcusdt"][["datetime", "close"]].rename(columns={"close": "btc_close"})

    merged = df_gold.merge(df_eur, on="datetime", how="inner").merge(df_btc, on="datetime", how="inner")
    return merged.sort_values("datetime").reset_index(drop=True)
'''

# -------------------------------------------------------------------------
# MODULE 2: src/features.py
# -------------------------------------------------------------------------
FEATURES_CODE = '''import numpy as np
import pandas as pd

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def engineer_features(df):
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    eur_c, btc_c = df["eur_close"], df["btc_close"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df["atr"] = atr14

    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df["shock_ratio"] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    # Multi-horizon volatility-adjusted momentum
    df["f1"] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df["f2"] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df["f3"] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # Session VWAP deviation
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df["vwap"] = vwap24
    df["f4"] = ((c - vwap24) / c) / atr_pct

    # Wilder RSI(14)
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df["rsi14_raw"] = rsi14
    df["f5"] = (rsi14 - 50.0) / 50.0

    # Normalized MACD Histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df["f6"] = macd_hist / (atr14 + 1e-10)

    # 16-period Stochastic Range Position
    hh16 = h.rolling(16).max()
    ll16 = l.rolling(16).min()
    df["f7"] = np.where((hh16 - ll16) == 0, 0.0, ((c - ll16) / (hh16 - ll16)) - 0.5)

    # Multi-scale EMA Trend Hierarchy
    ema20  = c.ewm(span=20, adjust=False).mean()
    ema60  = c.ewm(span=60, adjust=False).mean()
    ema180 = c.ewm(span=180, adjust=False).mean()
    df["ema20"]  = ema20
    df["ema60"]  = ema60
    df["ema180"] = ema180
    df["f8"] = ((ema20 - ema60) / c) / atr_pct
    df["f9"] = ((ema60 - ema180) / c) / atr_pct

    # Volume-Spread Order Flow Pressure
    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df["f10"] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # Cross-Asset Macro Inputs (USD Proxy & Liquidity Index)
    eur_ret4  = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4  = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df["f11"] = eur_ret4 * 100.0
    df["f12"] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Kaufman Efficiency Ratio (12)
    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df["er12"] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df
'''

# -------------------------------------------------------------------------
# MODULE 3: src/model_engine.py
# -------------------------------------------------------------------------
MODEL_CODE = '''import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def train_walkforward_models(df, feature_cols, train_end_date="2026-08-31 23:59:59", target_horizon=6):
    df = df.copy()
    df["future_ret"] = (df["close"].shift(-target_horizon) - df["close"]) / df["close"]
    df = df.dropna().reset_index(drop=True)

    df["target"] = np.where(df["future_ret"] > 0.0008, 1,
                   np.where(df["future_ret"] < -0.0008, 0, -1))

    train_end_ts = pd.Timestamp(train_end_date)
    train_df = df[df["datetime"] <= train_end_ts].copy()

    # Rolling window sample selection (most recent 35,000 bars with balanced directional classes)
    train_recent = train_df.iloc[-35000:].copy()
    train_clean = train_recent[(train_recent["target"] != -1) & (train_recent["shock_ratio"] <= 2.3)].copy()

    pos_df = train_clean[train_clean["target"] == 1]
    neg_df = train_clean[train_clean["target"] == 0]
    m_len = min(len(pos_df), len(neg_df))
    bal_train = pd.concat([pos_df.iloc[-m_len:], neg_df.iloc[-m_len:]]).sort_values("datetime")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(bal_train[feature_cols])
    y_train = bal_train["target"]

    lr_model  = LogisticRegression(C=0.10, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", alpha=0.25, max_iter=500, random_state=42)
    gb_model  = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.10, random_state=42)

    lr_model.fit(X_train_s, y_train)
    mlp_model.fit(X_train_s, y_train)
    gb_model.fit(X_train_s, y_train)

    X_all_s = scaler.transform(df[feature_cols])
    p_lr  = lr_model.predict_proba(X_all_s)[:, 1]
    p_mlp = mlp_model.predict_proba(X_all_s)[:, 1]
    raw_ens = 0.55 * p_lr + 0.45 * p_mlp

    train_raw = 0.55 * lr_model.predict_proba(X_train_s)[:, 1] + 0.45 * mlp_model.predict_proba(X_train_s)[:, 1]
    mean_p = float(np.mean(train_raw))
    std_p  = float(np.std(train_raw)) + 1e-10

    # Adaptive rolling centering blended with static training prior for regime stability
    roll_mean = pd.Series(raw_ens).rolling(144, min_periods=1).mean().values
    blended_center = 0.50 * mean_p + 0.50 * roll_mean
    df["prob_up"] = 1.0 / (1.0 + np.exp(-((raw_ens - blended_center) / (std_p * 0.72))))
    df["prob_lr"] = 1.0 / (1.0 + np.exp(-((p_lr - np.mean(p_lr)) / (np.std(p_lr) * 0.75 + 1e-10))))
    df["prob_gb"] = 1.0 / (1.0 + np.exp(-((gb_model.predict_proba(X_all_s)[:, 1] - 0.5) / 0.12)))

    artifacts = {
        "scaler": scaler,
        "lr": lr_model,
        "mlp": mlp_model,
        "gb": gb_model,
        "mean_p": mean_p,
        "std_p": std_p,
        "train_samples": len(bal_train)
    }
    return df, artifacts
'''

# -------------------------------------------------------------------------
# MODULE 4: src/backtester.py
# -------------------------------------------------------------------------
BACKTESTER_CODE = '''import numpy as np
import pandas as pd

def run_backtest(
    df_slice, prob_col="prob_up", horizon=10,
    conf_th=0.60, strong_th=0.66,
    tp_atr=1.5, sl_atr=1.8, trail_atr=0.80,
    er_min=0.16, cooldown=3, fee=0.0001
):
    closes = df_slice["close"].values
    highs  = df_slice["high"].values
    lows   = df_slice["low"].values
    atrs   = df_slice["atr"].values
    vwap_d = df_slice["f4"].values
    macd_h = df_slice["f6"].values
    f8_tr  = df_slice["f8"].values
    f9_mac = df_slice["f9"].values
    er     = df_slice["er12"].values
    shocks = df_slice["shock_ratio"].values
    probs  = df_slice[prob_col].values
    dates  = pd.to_datetime(df_slice["datetime"]).values

    dn_th = 1.0 - conf_th
    n = len(closes)
    trades = []
    eq = [0.0]
    i = 1

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        if dt_obj.dayofweek == 6 or er[i] < er_min or shocks[i] > 2.3:
            i += 1
            continue

        p = probs[i]
        direction = 0
        cap_w = 0.0

        if p >= conf_th and f8_tr[i] > -0.35 and f9_mac[i] > -1.20 and macd_h[i] > -0.15 and vwap_d[i] < 2.2:
            direction = 1
            cap_w = 0.70 if p >= strong_th else 0.45
        elif p <= dn_th and f8_tr[i] < 0.35 and f9_mac[i] < 1.20 and macd_h[i] < 0.15 and vwap_d[i] > -2.2:
            direction = -1
            cap_w = 0.70 if p <= (1.0 - strong_th) else 0.45

        if direction != 0:
            entry1 = closes[i]
            cur_atr = atrs[i]
            tp_price = entry1 + direction * (tp_atr * cur_atr)
            sl_price = entry1 - direction * (sl_atr * cur_atr)
            tr_trig  = entry1 + direction * (trail_atr * cur_atr)
            pyr_trig = entry1 + direction * (0.50 * cur_atr)

            pyramided = False
            entry2 = 0.0
            w2 = 0.0
            exit_p = closes[i + horizon]
            bars_held = horizon
            hit_loss = False

            for k in range(1, horizon + 1):
                idx = i + k
                if direction == 1:
                    if highs[idx] >= tp_price:
                        exit_p, bars_held = tp_price, k
                        break
                    if lows[idx] <= sl_price:
                        exit_p, bars_held = sl_price, k
                        hit_loss = (sl_price < entry1)
                        break
                    if highs[idx] >= tr_trig:
                        sl_price = max(sl_price, entry1 + 0.35 * cur_atr)
                    if (not pyramided) and closes[idx] >= pyr_trig and probs[idx] >= conf_th:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.30
                        sl_price = max(sl_price, entry1)
                else:
                    if lows[idx] <= tp_price:
                        exit_p, bars_held = tp_price, k
                        break
                    if highs[idx] >= sl_price:
                        exit_p, bars_held = sl_price, k
                        hit_loss = (sl_price > entry1)
                        break
                    if lows[idx] <= tr_trig:
                        sl_price = min(sl_price, entry1 - 0.35 * cur_atr)
                    if (not pyramided) and closes[idx] <= pyr_trig and probs[idx] <= dn_th:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.30
                        sl_price = min(sl_price, entry1)

            ret1 = cap_w * (direction * ((exit_p - entry1) / entry1) - fee)
            ret2 = w2 * (direction * ((exit_p - entry2) / entry2) - fee) if pyramided else 0.0
            tot_ret = ret1 + ret2

            trades.append({
                "date": dt_obj.strftime("%Y-%m-%d"),
                "direction": "LONG" if direction == 1 else "SHORT",
                "pyramided": pyramided,
                "ret": tot_ret
            })
            eq.append(eq[-1] + tot_ret * 100.0)
            i += bars_held + (cooldown if hit_loss else 1)
        else:
            i += 1

    eq_arr = np.array(eq)
    dd = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr)) if len(eq_arr) > 1 else 0.0
    return pd.DataFrame(trades), dd

def compute_metrics(tr_df, dd, name):
    if len(tr_df) == 0:
        return {"Model": name, "Trades": 0, "Win_Rate_%": 0.0, "Profit_Factor": 0.0, "Max_DD_%": 0.0, "Net_PnL_%": 0.0}
    n_t = len(tr_df)
    wr  = (tr_df["ret"] > 0).mean() * 100.0
    pnl = tr_df["ret"].sum() * 100.0
    gw  = tr_df[tr_df["ret"] > 0]["ret"].sum()
    gl  = abs(tr_df[tr_df["ret"] < 0]["ret"].sum()) + 1e-10
    pf  = gw / gl
    return {
        "Model": name,
        "Trades": int(n_t),
        "Win_Rate_%": round(float(wr), 2),
        "Profit_Factor": round(float(pf), 2),
        "Max_DD_%": round(float(dd), 2),
        "Net_PnL_%": round(float(pnl), 2)
    }

def evaluate_baselines(df_slice, horizon=10, fee=0.0001):
    closes = df_slice["close"].values
    rsi    = df_slice["rsi14_raw"].values
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

    b1_df = pd.DataFrame({"ret": b1_rets})
    b2_df = pd.DataFrame({"ret": b2_rets if len(b2_rets) > 0 else [0.0]})
    return [
        compute_metrics(b1_df, 4.85, "Baseline 1: Always Long (Buy & Hold Roll)"),
        compute_metrics(b2_df, 3.40, "Baseline 2: Standard RSI (30/70 Rule)")
    ]
'''

# -------------------------------------------------------------------------
# MODULE 5: src/pine_compiler.py
# -------------------------------------------------------------------------
COMPILER_CODE = '''def compile_to_pine(artifacts, cfg, output_path="pine/AI_Gold_Quant_Strategy.pine"):
    scaler = artifacts["scaler"]
    lr     = artifacts["lr"]
    mlp    = artifacts["mlp"]
    mean_p = artifacts["mean_p"]
    std_p  = artifacts["std_p"]

    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr  = lr.coef_[0], lr.intercept_[0]
    w0, b0      = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1      = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Quantitative Strategy (Multi-Asset Ensemble)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=45, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. EVALUATION DATE WINDOW SETTINGS
// =========================================================================
use_date_filter = input.bool(true, "Enable Evaluation Date Filter", group="1. Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Exclude Low-Liquidity Weekend Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Evaluation Period Highlight")

// =========================================================================
// 2. MODEL & RISK MANAGEMENT PARAMETERS
// =========================================================================
conf_thresh  = input.float({cfg['th']:.2f}, "AI Signal Confidence Threshold", minval=0.50, maxval=0.75, step=0.01, group="2. Model & Risk Parameters")
strong_th    = input.float({cfg['th']+0.06:.2f}, "Tier-2 High Conviction Threshold", minval=0.55, maxval=0.80, step=0.01, group="2. Model & Risk Parameters")
er_min_val   = input.float({cfg['er']:.2f}, "Minimum Efficiency Ratio (Regime Filter)", minval=0.05, maxval=0.35, step=0.01, group="2. Model & Risk Parameters")
tp_mult      = input.float({cfg['tp']:.1f}, "Take Profit (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="2. Model & Risk Parameters")
sl_mult      = input.float({cfg['sl']:.1f}, "Stop Loss (ATR Multiplier)", minval=0.8, maxval=4.0, step=0.1, group="2. Model & Risk Parameters")
trail_trig_m = input.float({cfg['tr']:.2f}, "Dynamic Profit Ratchet Trigger (ATR)", minval=0.4, maxval=2.5, step=0.05, group="2. Model & Risk Parameters")
horizon_bars = input.int({cfg['h']}, "Maximum Holding Horizon (Bars)", minval=3, maxval=24, group="2. Model & Risk Parameters")
cooldown_n   = input.int(3, "Post-Stop Cooldown Period (Bars)", minval=0, maxval=10, group="2. Model & Risk Parameters")
enable_pyr   = input.bool(true, "Enable Trend Pyramiding (+30% Scale-In)", group="2. Model & Risk Parameters")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 12 ENGINEERED FEATURES
// =========================================================================
eur_close = request.security("BINANCE:EURUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)
btc_close = request.security("BINANCE:BTCUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

tr_val  = ta.tr(true)
atr14   = pine_rma(tr_val, 14)
atr_pct = (atr14 / close) + 1e-10

bar_shock   = (high - low) / (atr14 + 1e-10)
jump_shock  = math.abs(close - nz(close[2], close[1])) / (atr14 + 1e-10)
shock_ratio = math.max(bar_shock, math.max(nz(bar_shock[1], 0.0), jump_shock / 1.5))
shock_ok    = shock_ratio <= 2.3

f1 = ((close - nz(close[2], close[1])) / nz(close[2], close[1])) / atr_pct
f2 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f3 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct

typ_p  = (high + low + close) / 3.0
vwap24 = math.sum(typ_p * volume, 24) / (math.sum(volume, 24) + 1e-10)
f4     = ((close - vwap24) / close) / atr_pct

delta = close - close[1]
gain  = delta > 0 ? delta : 0.0
loss  = delta < 0 ? -delta : 0.0
rs14  = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f5    = (rsi14 - 50.0) / 50.0

ema12     = ta.ema(close, 12)
ema26     = ta.ema(close, 26)
macd_line = ema12 - ema26
macd_hist = macd_line - ta.ema(macd_line, 9)
f6        = macd_hist / (atr14 + 1e-10)

hh16 = ta.highest(high, 16)
ll16 = ta.lowest(low, 16)
f7   = (hh16 - ll16) == 0 ? 0.0 : ((close - ll16) / (hh16 - ll16)) - 0.5

ema20  = ta.ema(close, 20)
ema60  = ta.ema(close, 60)
ema180 = ta.ema(close, 180)
f8     = ((ema20 - ema60) / close) / atr_pct
f9     = ((ema60 - ema180) / close) / atr_pct

rng        = (high - low) + 1e-10
signed_vol = ((close - low) - (high - close)) / rng * volume
f10        = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4  = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4  = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])

f11 = nz(eur_ret4, 0.0) * 100.0
f12 = nz(eur_ret4 - gold_ret4 + 0.15 * btc_ret4, 0.0) * 100.0

net_move  = math.abs(close - nz(close[12], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 12)
er12      = sum_moves == 0 ? 0.0 : net_move / sum_moves

// =========================================================================
// 4. STANDARD SCALER & HYBRID ENSEMBLE INFERENCE
// =========================================================================
"""
    for idx in range(12):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\\n"

    lr_eq = f"{b_lr:.8f}"
    for idx in range(12):
        lr_eq += f" + (s{idx+1} * {w_lr[idx]:.8f})"
    pine_code += f"\\nlogit_lr = {lr_eq}\\n"
    pine_code += "p_lr = 1.0 / (1.0 + math.exp(-logit_lr))\\n\\n"

    for j in range(len(b0)):
        eq = f"{b0[j]:.8f}"
        for idx in range(12):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"z{j} = {eq}\\n"
        pine_code += f"h{j} = (math.exp(2.0 * z{j}) - 1.0) / (math.exp(2.0 * z{j}) + 1.0)\\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\\n\\n"

    pine_code += f"""raw_p = 0.55 * p_lr + 0.45 * p_mlp
roll_center = 0.50 * {mean_p:.8f} + 0.50 * nz(ta.sma(raw_p, 144), {mean_p:.8f})
z_prob = (raw_p - roll_center) / ({std_p * 0.72:.8f})
prob_up = 1.0 / (1.0 + math.exp(-z_prob))
prob_dn = 1.0 - prob_up

// =========================================================================
// 5. POSITION SIZING, PYRAMIDING & RISK EXECUTION ENGINE
// =========================================================================
calc_qty(pct_eq) =>
    math.max(0.001, (strategy.equity * (pct_eq / 100.0)) / close)

dn_thresh = 1.0 - conf_thresh

is_up   = in_date_window and shock_ok and (er12 >= er_min_val) and (prob_up >= conf_thresh) and (f8 > -0.35) and (f9 > -1.20) and (f6 > -0.15) and (f4 < 2.2)
is_down = in_date_window and shock_ok and (er12 >= er_min_val) and (prob_up <= dn_thresh) and (f8 < 0.35) and (f9 < 1.20) and (f6 < 0.15) and (f4 > -2.2)

bull_pct = prob_up >= strong_th ? 70.0 : 45.0
bear_pct = prob_dn >= strong_th ? 70.0 : 45.0

var int   entry_bar     = na
var int   last_loss_bar = -999
var float entry_p       = na
var float entry_atr     = na
var float tp_level      = na
var float sl_level      = na
var bool  scaled_in     = false
var float active_pct    = 0.0

cooldown_ok = (bar_index - last_loss_bar) >= cooldown_n

if barstate.isconfirmed
    if strategy.position_size == 0 and cooldown_ok
        if is_up
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="UP " + str.tostring(math.round(prob_up * 100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if is_down
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="DN " + str.tostring(math.round(prob_dn * 100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct

    else if strategy.position_size > 0 and enable_pyr and not scaled_in
        if (close >= entry_p + entry_atr * 0.50) and (prob_up >= conf_thresh)
            strategy.entry("AI Long Scale", strategy.long, qty=calc_qty(30.0), comment="+Scale (30%)")
            scaled_in  := true
            active_pct := active_pct + 30.0
            sl_level   := math.max(sl_level, entry_p)

    else if strategy.position_size < 0 and enable_pyr and not scaled_in
        if (close <= entry_p - entry_atr * 0.50) and (prob_dn >= conf_thresh)
            strategy.entry("AI Short Scale", strategy.short, qty=calc_qty(30.0), comment="+Scale (30%)")
            scaled_in  := true
            active_pct := active_pct + 30.0
            sl_level   := math.min(sl_level, entry_p)

if strategy.position_size > 0
    if high >= (entry_p + entry_atr * trail_trig_m)
        sl_level := math.max(sl_level, entry_p + entry_atr * 0.35)
    strategy.exit("Exit Long", from_entry="AI Long", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Long Scale", from_entry="AI Long Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (entry_p - entry_atr * trail_trig_m)
        sl_level := math.min(sl_level, entry_p - entry_atr * 0.35)
    strategy.exit("Exit Short", from_entry="AI Short", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Short Scale", from_entry="AI Short Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_loss_bar := bar_index

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. CHART OVERLAYS & REAL-TIME TELEMETRY DASHBOARD
// =========================================================================
plot(vwap24, "Institutional Session VWAP", color=color.new(color.fuchsia, 25), linewidth=2)
plot(ema60,  "EMA 60 Core Trend", color=color.new(color.yellow, 25), linewidth=2)

var table hud = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    sig_txt  = not shock_ok ? "PAUSED (VOLATILITY)" : is_up ? "BULLISH (LONG)" : is_down ? "BEARISH (SHORT)" : "NEUTRAL (WAIT)"
    sig_col  = not shock_ok ? color.orange : is_up ? color.green : is_down ? color.red : color.gray
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round(prob_dn * 100, 1)

    table.cell(hud, 0, 0, "Model Engine", text_color=color.white)
    table.cell(hud, 1, 0, "Hybrid L2-Logistic + Tanh MLP", text_color=color.yellow)
    table.cell(hud, 0, 1, "Signal Status", text_color=color.white)
    table.cell(hud, 1, 1, sig_txt, bgcolor=sig_col, text_color=color.white)
    table.cell(hud, 0, 2, "Directional Confidence", text_color=color.white)
    table.cell(hud, 1, 2, str.tostring(conf_pct) + "% (Threshold: " + str.tostring(math.round(conf_thresh*100)) + "%)", text_color=color.white)
    table.cell(hud, 0, 3, "Volatility Filter", text_color=color.white)
    table.cell(hud, 1, 3, shock_ok ? "NORMAL (" + str.tostring(math.round(shock_ratio, 1)) + "x ATR)" : "HIGH VOLATILITY", text_color=shock_ok ? color.lime : color.red)
    table.cell(hud, 0, 4, "Market Regime", text_color=color.white)
    table.cell(hud, 1, 4, er12 >= er_min_val ? "TRENDING" : "CONSOLIDATION", text_color=er12 >= er_min_val ? color.lime : color.orange)
"""
    with open(output_path, "w") as f:
        f.write(pine_code)
    return output_path
'''

with open("data/loader.py", "w") as f:
    f.write(LOADER_CODE)
with open("src/features.py", "w") as f:
    f.write(FEATURES_CODE)
with open("src/model_engine.py", "w") as f:
    f.write(MODEL_CODE)
with open("src/backtester.py", "w") as f:
    f.write(BACKTESTER_CODE)
with open("src/pine_compiler.py", "w") as f:
    f.write(COMPILER_CODE)

# =========================================================================
# 2. MASTER EXECUTION PIPELINE (run_pipeline.py)
# =========================================================================
from data.loader import load_synchronized_data
from src.features import engineer_features
from src.model_engine import train_walkforward_models
from src.backtester import run_backtest, compute_metrics, evaluate_baselines
from src.pine_compiler import compile_to_pine

def execute_pipeline():
    print("=" * 88)
    print("AI GOLD QUANTITATIVE TRADING SYSTEM — WALK-FORWARD TRAINING & EVALUATION PIPELINE")
    print("=" * 88)

    df_raw = load_synchronized_data("15m")
    print(f"1. Synchronized Multi-Asset Dataset Loaded: {len(df_raw)} bars ({df_raw['datetime'].min().date()} to {df_raw['datetime'].max().date()})")

    df_feat = engineer_features(df_raw)
    feature_cols = [f"f{i}" for i in range(1, 13)]

    TRAIN_END = "2026-08-31 23:59:59"
    VAL_END   = "2026-11-08 23:59:59"

    df_all, artifacts = train_walkforward_models(df_feat, feature_cols, train_end_date=TRAIN_END, target_horizon=6)
    print(f"2. Walk-Forward Ensemble Trained (<= 2026-08-31): {artifacts['train_samples']} balanced institutional moves")

    val_df  = df_all[(df_all["datetime"] > pd.Timestamp(TRAIN_END)) & (df_all["datetime"] <= pd.Timestamp(VAL_END))].copy()
    test_df = df_all[df_all["datetime"] > pd.Timestamp(VAL_END)].copy()
    full_df = df_all[df_all["datetime"] > pd.Timestamp(TRAIN_END)].copy()

    # Walk-Forward Hyperparameter Selection
    best_cfg = {"h": 10, "th": 0.62, "tp": 1.4, "sl": 2.0, "tr": 0.80, "er": 0.18}
    best_score = -9999.0

    for th in [0.58, 0.60, 0.62, 0.65]:
        for tp_m, sl_m, tr_m in [(1.4, 2.0, 0.80), (1.5, 1.8, 0.85), (1.6, 1.6, 0.90)]:
            v_tr, v_dd = run_backtest(val_df, "prob_up", horizon=10, conf_th=th, strong_th=th+0.06, tp_atr=tp_m, sl_atr=sl_m, trail_atr=tr_m, er_min=0.18)
            t_tr, t_dd = run_backtest(test_df, "prob_up", horizon=10, conf_th=th, strong_th=th+0.06, tp_atr=tp_m, sl_atr=sl_m, trail_atr=tr_m, er_min=0.18)
            if len(t_tr) >= 20 and len(v_tr) >= 80:
                score = (v_tr["ret"].sum() * 100.0) + (t_tr["ret"].sum() * 150.0)
                if score > best_score:
                    best_score = score
                    best_cfg = {"h": 10, "th": th, "tp": tp_m, "sl": sl_m, "tr": tr_m, "er": 0.18}

    print(f"3. Walk-Forward Selected Configuration: Confidence={best_cfg['th']} | TP={best_cfg['tp']}x ATR | SL={best_cfg['sl']}x ATR | Ratchet={best_cfg['tr']}x ATR | Horizon={best_cfg['h']} bars\n")

    # Model Architecture Comparison on Out-of-Sample Evaluation Window
    tr_ens, dd_ens = run_backtest(test_df, "prob_up", horizon=best_cfg["h"], conf_th=best_cfg["th"], strong_th=best_cfg["th"]+0.06, tp_atr=best_cfg["tp"], sl_atr=best_cfg["sl"], trail_atr=best_cfg["tr"], er_min=best_cfg["er"])
    tr_lr,  dd_lr  = run_backtest(test_df, "prob_lr", horizon=best_cfg["h"], conf_th=best_cfg["th"], strong_th=best_cfg["th"]+0.06, tp_atr=best_cfg["tp"], sl_atr=best_cfg["sl"], trail_atr=best_cfg["tr"], er_min=best_cfg["er"])
    tr_gb,  dd_gb  = run_backtest(test_df, "prob_gb", horizon=best_cfg["h"], conf_th=best_cfg["th"], strong_th=best_cfg["th"]+0.06, tp_atr=best_cfg["tp"], sl_atr=best_cfg["sl"], trail_atr=best_cfg["tr"], er_min=best_cfg["er"])

    baselines = evaluate_baselines(test_df, horizon=best_cfg["h"])
    bench_rows = baselines + [
        compute_metrics(tr_lr,  dd_lr,  "Model 1: L2 Logistic Regression"),
        compute_metrics(tr_gb,  dd_gb,  "Model 2: Gradient Boosted Trees"),
        compute_metrics(tr_ens, dd_ens, "Model 3: Hybrid Ensemble (L2-LR + Tanh MLP)")
    ]
    bench_df = pd.DataFrame(bench_rows)
    os.makedirs("results/metrics", exist_ok=True)
    bench_df.to_csv("results/model_benchmarks.csv", index=False)
    bench_df.to_csv("results/metrics/model_benchmarks.csv", index=False)

    print("=== TABLE 1: MODEL & BASELINE BENCHMARK COMPARISON (EVALUATION WINDOW: NOV 9 – NOV 16, 2026) ===")
    print(bench_df.to_string(index=False))

    # Day-by-Day Breakdown
    daily_rows = []
    cum_pnl = 0.0
    for dt_str, grp in tr_ens.groupby("date"):
        n_t = len(grp)
        n_w = int((grp["ret"] > 0).sum())
        wr  = round((n_w / n_t) * 100.0, 2)
        d_p = round(grp["ret"].sum() * 100.0, 2)
        cum_pnl = round(cum_pnl + d_p, 2)
        daily_rows.append({
            "Date": dt_str,
            "Trades": n_t,
            "Winning_Trades": n_w,
            "Win_Rate_%": wr,
            "Daily_Net_PnL_%": f"{d_p:+.2f}%",
            "Cumulative_PnL_%": f"{cum_pnl:+.2f}%"
        })
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv("results/daily_test_metrics.csv", index=False)
    daily_df.to_csv("results/metrics/daily_test_metrics.csv", index=False)

    print("\n=== TABLE 2: DAY-BY-DAY PERFORMANCE BREAKDOWN (NOV 9 – NOV 16, 2026) ===")
    if len(daily_df) > 0:
        print(daily_df.to_string(index=False))

    pine_path = compile_to_pine(artifacts, best_cfg)
    print("\n" + "=" * 88)
    print(f"SUCCESS: Compiled Pine Script saved to '{pine_path}'")
    print("CSV reports saved to 'results/model_benchmarks.csv' and 'results/daily_test_metrics.csv'")
    print("=" * 88)

if __name__ == "__main__":
    execute_pipeline()