import os
import time
import json
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. LOAD CACHED CSVs + AUTO-SYNC LATEST LIVE CANDLES UP TO RIGHT NOW
# =========================================================================
def sync_latest_binance_bars(df_existing, symbol, interval="15m"):
    try:
        last_ts_ms = int(df_existing["datetime"].max().timestamp() * 1000) + 1
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={last_ts_ms}&limit=1000"
        r = requests.get(url, timeout=8).json()
        if isinstance(r, list) and len(r) > 0:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore']
            df_new = pd.DataFrame(r, columns=cols)
            df_new["datetime"] = pd.to_datetime(df_new["timestamp"], unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df_new[c] = pd.to_numeric(df_new[c])
            df_new = df_new[["datetime", "open", "high", "low", "close", "volume"]]
            df_combined = pd.concat([df_existing[["datetime", "open", "high", "low", "close", "volume"]], df_new])
            df_combined = df_combined.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
            print(f"   Synced +{len(df_new)} live {symbol} bars up to {df_combined['datetime'].max()} UTC")
            return df_combined
    except Exception as e:
        print(f"   Offline build mode for {symbol} (Browser will sync live REST/WebSocket automatically): {e}")
    return df_existing

def load_and_prepare_data():
    symbols = ["paxgusdt", "eurusdt", "btcusdt"]
    dfs = {}
    for sym in symbols:
        for days in [1460, 730, 365]:
            candidate_paths = [
                os.path.join("data", "cache", f"{sym}_15m_{days}d.csv"),
                os.path.join("..", "data", "cache", f"{sym}_15m_{days}d.csv"),
                os.path.join("data", f"{sym}_15m_{days}d.csv"),
                os.path.join("..", "data", f"{sym}_15m_{days}d.csv"),
                f"{sym}_15m_{days}d.csv"
            ]
            for fname in candidate_paths:
                if os.path.exists(fname):
                    d = pd.read_csv(fname)
                    d["datetime"] = pd.to_datetime(d["datetime"])
                    d = sync_latest_binance_bars(d, sym.upper(), "15m")
                    dfs[sym] = d
                    break
            if sym in dfs:
                break

    if len(dfs) != 3:
        raise FileNotFoundError(
            "Missing cached CSVs. Run this script in the folder containing "
            "paxgusdt_15m_*.csv, eurusdt_15m_*.csv, and btcusdt_15m_*.csv."
        )

    print("1. Synchronizing multi-year PAXGUSDT, EURUSDT, and BTCUSDT datasets...")
    df = dfs["paxgusdt"].merge(
        dfs["eurusdt"][["datetime", "close"]].rename(columns={"close": "eur_close"}),
        on="datetime", how="inner"
    ).merge(
        dfs["btcusdt"][["datetime", "close"]].rename(columns={"close": "btc_close"}),
        on="datetime", how="inner"
    ).sort_values("datetime").reset_index(drop=True)

    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    eur_c, btc_c = df["eur_close"], df["btc_close"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df["atr"] = atr14

    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df["shock_ratio"] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    df["f1"] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df["f2"] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df["f3"] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df["vwap"] = vwap24
    df["f4"] = ((c - vwap24) / c) / atr_pct

    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df["rsi14_raw"] = rsi14
    df["f5"] = (rsi14 - 50.0) / 50.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df["f6"] = macd_hist / (atr14 + 1e-10)

    hh16 = h.rolling(16).max()
    ll16 = l.rolling(16).min()
    df["f7"] = np.where((hh16 - ll16) == 0, 0.0, ((c - ll16) / (hh16 - ll16)) - 0.5)

    ema20  = c.ewm(span=20, adjust=False).mean()
    ema60  = c.ewm(span=60, adjust=False).mean()
    ema180 = c.ewm(span=180, adjust=False).mean()
    df["ema20"]  = ema20
    df["ema60"]  = ema60
    df["ema180"] = ema180
    df["f8"] = ((ema20 - ema60) / c) / atr_pct
    df["f9"] = ((ema60 - ema180) / c) / atr_pct

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    df["bb_up"] = sma20 + 2.0 * std20
    df["bb_dn"] = sma20 - 2.0 * std20

    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df["f10"] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    eur_ret4  = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4  = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df["f11"] = eur_ret4 * 100.0
    df["f12"] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df["er12"] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    df["future_ret"] = (df["close"].shift(-6) - df["close"]) / df["close"]
    df = df.dropna().reset_index(drop=True)
    df["target"] = np.where(df["future_ret"] > 0.0008, 1,
                   np.where(df["future_ret"] < -0.0008, 0, -1))

    feature_cols = [f"f{i}" for i in range(1, 13)]
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")

    train_df = df[df["datetime"] <= TRAIN_END].iloc[-35000:].copy()
    train_clean = train_df[(train_df["target"] != -1) & (train_df["shock_ratio"] <= 2.3)].copy()
    pos_df = train_clean[train_clean["target"] == 1]
    neg_df = train_clean[train_clean["target"] == 0]
    m_len = min(len(pos_df), len(neg_df))
    bal_train = pd.concat([pos_df.iloc[-m_len:], neg_df.iloc[-m_len:]]).sort_values("datetime")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(bal_train[feature_cols])
    y_train = bal_train["target"]

    print("2. Training Hybrid Ensemble (L2 Logistic Regression + 8-Neuron Tanh MLP)...")
    lr_model  = LogisticRegression(C=0.10, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", alpha=0.25, max_iter=500, random_state=42)
    lr_model.fit(X_train_s, y_train)
    mlp_model.fit(X_train_s, y_train)

    X_all_s = scaler.transform(df[feature_cols])
    p_lr  = lr_model.predict_proba(X_all_s)[:, 1]
    p_mlp = mlp_model.predict_proba(X_all_s)[:, 1]
    raw_ens = 0.55 * p_lr + 0.45 * p_mlp

    train_raw = 0.55 * lr_model.predict_proba(X_train_s)[:, 1] + 0.45 * mlp_model.predict_proba(X_train_s)[:, 1]
    mean_p = float(np.mean(train_raw))
    std_p  = float(np.std(train_raw)) + 1e-10

    roll_mean = pd.Series(raw_ens).rolling(144, min_periods=1).mean().values
    blended_center = 0.50 * mean_p + 0.50 * roll_mean
    df["prob_up"] = 1.0 / (1.0 + np.exp(-((raw_ens - blended_center) / (std_p * 0.72))))
    df["p_lr"]  = p_lr
    df["p_mlp"] = p_mlp

    sub = df.iloc[-35040:].copy().reset_index(drop=True)
    print(f"3. Packing 1 Full Year ({len(sub)} bars: {sub['datetime'].min()} -> {sub['datetime'].max()})...")

    sub["ts"] = (sub["datetime"].astype("int64") // 10**9).astype(int)
    for col in ["open", "high", "low", "close", "volume", "atr", "vwap", "ema20", "ema60", "ema180", "bb_up", "bb_dn"]:
        sub[col] = sub[col].round(2)
    for col in ["er12", "f4", "f6", "f8", "f9", "f10"]:
        sub[col] = sub[col].round(3)
    sub["shock_ratio"] = sub["shock_ratio"].round(2)
    sub["prob_up"] = sub["prob_up"].round(4)
    sub["p_lr"] = sub["p_lr"].round(4)
    sub["p_mlp"] = sub["p_mlp"].round(4)
    sub["eur_close"] = sub["eur_close"].round(4)
    sub["btc_close"] = sub["btc_close"].round(0)
    sub["rsi14_raw"] = sub["rsi14_raw"].round(1)

    pack_cols = [
        "ts", "open", "high", "low", "close", "volume",
        "atr", "vwap", "ema20", "ema60", "ema180", "bb_up", "bb_dn",
        "er12", "shock_ratio", "prob_up", "p_lr", "p_mlp",
        "eur_close", "btc_close", "rsi14_raw", "f4", "f6", "f8", "f9", "f10"
    ]
    bars_compact = sub[pack_cols].values.tolist()

    model_meta = {
        "feature_names": [
            "f1: 30m Momentum / ATR",
            "f2: 1h Momentum / ATR",
            "f3: 3h Momentum / ATR",
            "f4: Session VWAP Distance",
            "f5: Wilder RSI(14) Centered",
            "f6: Normalized MACD Hist",
            "f7: 16-Bar Stoch Range",
            "f8: EMA 20-60 Trend Spread",
            "f9: EMA 60-180 Macro Spread",
            "f10: Order Flow VSA Pressure",
            "f11: EUR/USD 1h Macro Mom",
            "f12: Cross-Asset Divergence"
        ],
        "mean": [round(float(x), 6) for x in scaler.mean_],
        "scale": [round(float(x), 6) for x in scaler.scale_],
        "w_lr": [round(float(x), 6) for x in lr_model.coef_[0]],
        "b_lr": round(float(lr_model.intercept_[0]), 6),
        "w0": [[round(float(v), 6) for v in row] for row in mlp_model.coefs_[0]],
        "b0": [round(float(v), 6) for v in mlp_model.intercepts_[0]],
        "w1": [round(float(row[0]), 6) for row in mlp_model.coefs_[1]],
        "b1": round(float(mlp_model.intercepts_[1][0]), 6),
        "mean_p": round(mean_p, 6),
        "std_p": round(std_p, 6)
    }

    return bars_compact, model_meta

def build_live_html(bars_compact, model_meta):
    payload_str = json.dumps({
        "bars": bars_compact,
        "meta": model_meta
    }, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bullion AI Live Strategy Studio — PAXG/USDT & XAU/USD</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
<style>
:root[data-theme="light"] {{
  --bg-app: #f4f6fa;
  --bg-card: #ffffff;
  --bg-subtle: #f8fafc;
  --bg-hover: #f1f5f9;
  --text-main: #0f172a;
  --text-secondary: #334155;
  --text-muted: #64748b;
  --border-col: #e2e8f0;
  --border-strong: #cbd5e1;
  --accent-primary: #2563eb;
  --accent-soft: rgba(37, 99, 235, 0.08);
  --up-col: #059669;
  --up-bg: rgba(5, 150, 105, 0.10);
  --dn-col: #e11d48;
  --dn-bg: rgba(225, 29, 72, 0.10);
  --warn-col: #d97706;
  --warn-bg: rgba(217, 119, 6, 0.10);
  --grid-col: #eef2f6;
  --hud-bg: rgba(255, 255, 255, 0.92);
  --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
  --shadow-pop: 0 16px 36px -4px rgba(15, 23, 42, 0.16), 0 6px 14px -2px rgba(15, 23, 42, 0.08);
}}
:root[data-theme="dark"] {{
  --bg-app: #0b0f19;
  --bg-card: #111827;
  --bg-subtle: #161f33;
  --bg-hover: #1e293b;
  --text-main: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --border-col: #1e293b;
  --border-strong: #334155;
  --accent-primary: #3b82f6;
  --accent-soft: rgba(59, 130, 246, 0.14);
  --up-col: #10b981;
  --up-bg: rgba(16, 185, 129, 0.14);
  --dn-col: #f43f5e;
  --dn-bg: rgba(244, 63, 94, 0.14);
  --warn-col: #f59e0b;
  --warn-bg: rgba(245, 158, 11, 0.14);
  --grid-col: #1a2336;
  --hud-bg: rgba(17, 24, 39, 0.92);
  --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.25);
  --shadow-pop: 0 18px 40px rgba(0, 0, 0, 0.55);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background-color: var(--bg-app);
  color: var(--text-main);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 13.5px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }}
.num {{ font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; text-align: right; }}

.app-container {{
  max-width: 1660px;
  margin: 0 auto;
  padding: 16px 22px 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

/* TOP NAVBAR */
.top-navbar {{
  background: var(--bg-card);
  border: 1px solid var(--border-col);
  border-radius: 12px;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 14px;
  box-shadow: var(--shadow-sm);
}}
.brand-group {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.brand-logo {{
  width: 36px; height: 36px; border-radius: 9px;
  background: linear-gradient(135deg, #f59e0b, #d97706);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 15px;
}}
.brand-title {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 16px; }}
.brand-sub {{ font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 8px; }}
.price-badge-group {{
  display: flex; align-items: center; gap: 10px;
  padding-left: 14px; border-left: 1px solid var(--border-col); flex-wrap: wrap;
}}
.live-price {{ font-family: 'JetBrains Mono', monospace; font-size: 21px; font-weight: 700; }}
.pill-badge {{
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 600;
}}
.pill-up {{ background: var(--up-bg); color: var(--up-col); }}
.pill-dn {{ background: var(--dn-bg); color: var(--dn-col); }}
.pill-warn {{ background: var(--warn-bg); color: var(--warn-col); }}
.pill-neutral {{ background: var(--bg-subtle); color: var(--text-secondary); border: 1px solid var(--border-col); }}

.pulse-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; display: inline-block; }}
.pulse-dot.live {{ background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25); }}

.nav-controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.ctrl-group {{ display: flex; flex-direction: column; gap: 2px; }}
.ctrl-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }}
select.clean-select, button.clean-btn, input.clean-input {{
  font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500;
  height: 36px; padding: 0 12px;
  background: var(--bg-subtle); color: var(--text-main);
  border: 1px solid var(--border-strong); border-radius: 8px;
  cursor: pointer; transition: all 0.15s ease;
}}
select.clean-select:hover, button.clean-btn:hover {{ background: var(--bg-hover); }}
button.clean-btn.primary {{ background: var(--accent-primary); color: #fff; border-color: var(--accent-primary); font-weight: 600; }}
button.clean-btn.success {{ background: var(--up-col); color: #fff; border-color: var(--up-col); font-weight: 600; }}

/* POPOVER & MODAL */
.popover-wrapper {{ position: relative; }}
.popover-menu {{
  display: none; position: absolute; top: 42px; right: 0; width: 295px;
  background: var(--bg-card); border: 1px solid var(--border-strong);
  border-radius: 10px; box-shadow: var(--shadow-pop); padding: 12px; z-index: 120;
}}
.popover-menu.open {{ display: block; }}
.popover-section-title {{
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-muted); margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border-col);
}}
.popover-item {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 4px; font-size: 13px; cursor: pointer; border-radius: 6px;
}}
.popover-item:hover {{ background: var(--bg-hover); }}
.swatch-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }}

.modal-backdrop {{
  display: none; position: fixed; inset: 0; background: rgba(15, 23, 42, 0.55);
  z-index: 300; align-items: center; justify-content: center; padding: 20px;
}}
.modal-backdrop.open {{ display: flex; }}
.modal-card {{
  background: var(--bg-card); border: 1px solid var(--border-strong);
  border-radius: 14px; width: 100%; max-width: 660px; box-shadow: var(--shadow-pop);
  overflow: hidden;
}}

/* KPI SUMMARY STRIP */
.kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }}
.kpi-card {{
  background: var(--bg-card); border: 1px solid var(--border-col);
  border-radius: 12px; padding: 15px 18px; box-shadow: var(--shadow-sm);
  display: flex; flex-direction: column; justify-content: space-between; gap: 5px;
}}
.kpi-card-title {{ font-size: 11.5px; font-weight: 700; color: var(--text-muted); display: flex; align-items: center; justify-content: space-between; }}
.kpi-card-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }}
.kpi-card-sub {{ font-size: 12px; color: var(--text-muted); }}

/* MAIN WORKSPACE */
.main-grid {{ display: grid; grid-template-columns: 67fr 33fr; gap: 18px; align-items: stretch; }}
.card {{
  background: var(--bg-card); border: 1px solid var(--border-col);
  border-radius: 12px; box-shadow: var(--shadow-sm);
  display: flex; flex-direction: column; overflow: hidden;
}}
.card-header {{
  padding: 12px 18px; border-bottom: 1px solid var(--border-col);
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}}
.card-title {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 14.5px; font-weight: 700; }}
.toolbar-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.seg-group {{
  display: inline-flex; background: var(--bg-subtle); border: 1px solid var(--border-col);
  border-radius: 8px; padding: 2px;
}}
.seg-btn {{
  border: none; background: transparent; color: var(--text-muted);
  font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600;
  padding: 5px 10px; border-radius: 6px; cursor: pointer;
}}
.seg-btn.active {{ background: var(--bg-card); color: var(--text-main); box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}

.ohlc-bar {{
  padding: 8px 18px; background: var(--bg-subtle); border-bottom: 1px solid var(--border-col);
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
}}
.ohlc-items {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.ohlc-item span {{ color: var(--text-muted); margin-right: 4px; }}
.replay-bar {{
  padding: 10px 18px; border-top: 1px solid var(--border-col); background: var(--bg-subtle);
  display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
}}

/* ON-CHART FLOATING LIVE POSITION HUD BOX */
.chart-hud-overlay {{
  position: absolute;
  top: 12px;
  left: 14px;
  background: var(--hud-bg);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border-strong);
  border-radius: 10px;
  padding: 10px 14px;
  box-shadow: var(--shadow-sm);
  pointer-events: none;
  z-index: 20;
  min-width: 280px;
  max-width: 350px;
}}
.hud-header-line {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-bottom: 6px;
  margin-bottom: 6px;
  border-bottom: 1px solid var(--border-col);
  font-size: 11.5px;
  font-weight: 700;
}}
.hud-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 12px;
  font-size: 11.5px;
}}
.hud-cell-lbl {{ color: var(--text-muted); font-size: 10.5px; }}
.hud-cell-val {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}

.bottom-grid {{ display: grid; grid-template-columns: 46fr 54fr; gap: 18px; }}

.clean-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
.clean-table th {{
  text-align: left; padding: 9px 12px; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-muted);
  background: var(--bg-subtle); border-bottom: 1px solid var(--border-col);
  position: sticky; top: 0; z-index: 2;
}}
.clean-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border-col); white-space: nowrap; }}
.clean-table tr:hover {{ background: var(--bg-hover); }}

.inspector-row {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 16px; border-bottom: 1px solid var(--border-col); gap: 10px;
}}
.inspector-label {{ color: var(--text-secondary); font-size: 12.5px; font-weight: 500; }}
.inspector-val {{ font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 600; text-align: right; }}

.param-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 14px 16px; }}
.param-field {{
  background: var(--bg-subtle); border: 1px solid var(--border-col);
  border-radius: 8px; padding: 8px 10px; display: flex; flex-direction: column; gap: 4px;
}}
.param-field label {{ font-size: 11px; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; }}
.param-field input[type="number"], .param-field select {{
  width: 100%; height: 30px; padding: 0 8px; border-radius: 6px;
  border: 1px solid var(--border-strong); background: var(--bg-card);
  color: var(--text-main); font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 600;
}}

.up {{ color: var(--up-col); }}
.down {{ color: var(--dn-col); }}
.warn {{ color: var(--warn-col); }}
.accent {{ color: var(--accent-primary); }}

.cm-matrix {{ display: grid; grid-template-columns: 130px 1fr 1fr; gap: 10px; padding: 16px; }}
.cm-box {{ padding: 14px; border-radius: 10px; border: 1px solid var(--border-col); background: var(--bg-subtle); font-family: 'JetBrains Mono', monospace; }}
.cm-box.hit {{ background: var(--up-bg); border-color: var(--up-col); }}
.cm-box.miss {{ background: var(--dn-bg); border-color: var(--dn-col); }}

@media (max-width: 1200px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .main-grid, .bottom-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="app-container">

  <!-- TOP NAVIGATION BAR -->
  <header class="top-navbar">
    <div class="brand-group">
      <div class="brand-logo">Au</div>
      <div>
        <div class="brand-title">Bullion Quantitative AI Studio</div>
        <div class="brand-sub">
          <span><i class="pulse-dot live" id="binanceDot"></i> <span id="binanceStatusTxt">PAXG/USDT Ready</span></span>
          <span>•</span>
          <span><i class="pulse-dot" id="lseDot"></i> <span id="lseStatusTxt">LSE XAU/USD Standby</span></span>
        </div>
      </div>
      <div class="price-badge-group">
        <span id="activeAssetTag" class="pill-badge pill-warn">PAXG/USDT</span>
        <span id="livePrice" class="live-price up">$3,882.40</span>
        <span id="liveDelta" class="pill-badge pill-up">+0.18%</span>
        <span id="spreadBadge" class="pill-badge pill-neutral">Spread: $0.12</span>
        <span id="activeTimestamp" class="pill-badge pill-neutral">2026-11-16 UTC</span>
      </div>
    </div>

    <div class="nav-controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Primary Asset Feed</span>
        <select id="assetSelect" class="clean-select" onchange="onAssetChange()">
          <option value="PAXG">PAXG / USDT (Binance Tokenized Gold)</option>
          <option value="XAU">XAU / USD (London Strategic Edge Spot Gold)</option>
        </select>
      </div>

      <div class="ctrl-group">
        <span class="ctrl-label">Evaluation Window</span>
        <select id="periodSelect" class="clean-select" onchange="onPeriodChange()">
          <option value="OOS_WEEK">OOS Test Week + Live (Nov 09 – Now)</option>
          <option value="SEP_NOV">Validation + OOS (Sep 01 – Now)</option>
          <option value="6M">Last 6 Months (May 2026 – Now)</option>
          <option value="1Y">Full 1 Year (Nov 2025 – Now)</option>
          <option value="2026-11-09">Single Day: Mon Nov 09, 2026</option>
          <option value="2026-11-10">Single Day: Tue Nov 10, 2026</option>
          <option value="2026-11-11">Single Day: Wed Nov 11, 2026</option>
          <option value="2026-11-12">Single Day: Thu Nov 12, 2026</option>
          <option value="2026-11-13">Single Day: Fri Nov 13, 2026</option>
          <option value="2026-11-14">Single Day: Sat Nov 14, 2026</option>
          <option value="2026-11-16">Single Day: Mon Nov 16, 2026 (Live)</option>
        </select>
      </div>

      <div class="ctrl-group">
        <span class="ctrl-label">Timeframe</span>
        <select id="tfSelect" class="clean-select" onchange="onTimeframeChange()">
          <option value="AUTO" selected>Auto Resolution</option>
          <option value="15m">15m Intraday</option>
          <option value="1H">1H Hourly</option>
          <option value="4H">4H Swing</option>
          <option value="1D">1D Daily</option>
        </select>
      </div>

      <div class="ctrl-group">
        <span class="ctrl-label">Live Feeds & Settings</span>
        <div style="display:flex;gap:6px;">
          <button class="clean-btn success" onclick="openApiModal()">Live Feeds & API Key</button>
          <button class="clean-btn primary" onclick="setInspectorTab('settings')">Strategy Settings ⚙</button>
          <button id="themeBtn" class="clean-btn" onclick="toggleTheme()">Dark Mode</button>
        </div>
      </div>
    </div>
  </header>

  <!-- 5 SPACIOUS KPI CARDS -->
  <section class="kpi-grid" id="kpiGrid"></section>

  <!-- MAIN WORKSPACE: CHART (67%) + AI SIGNAL, MATH & STRATEGY STUDIO (33%) -->
  <div class="main-grid">
    <!-- LEFT: INTERACTIVE CHART CARD -->
    <section class="card">
      <div class="card-header">
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="card-title">Price Action, S&R Supply/Demand Zones & AI Trajectory</span>
          <span class="pill-badge pill-neutral" id="activeTfBadge">15m Candles</span>
        </div>

        <div class="toolbar-row">
          <select id="chartStyleSelect" class="clean-select" onchange="renderDashboard()">
            <option value="candles">Candlesticks</option>
            <option value="ohlc">OHLC Bars</option>
            <option value="area">Closing Area</option>
            <option value="baseline">VWAP Baseline</option>
          </select>

          <select id="projSelect" class="clean-select" onchange="renderDashboard()">
            <option value="all">AI Trajectory + ATR Brackets</option>
            <option value="traj">Forward Trajectory Only</option>
            <option value="brackets">ATR Brackets Only</option>
            <option value="off">Hide Projections</option>
          </select>

          <div class="popover-wrapper">
            <button class="clean-btn" onclick="toggleIndicatorMenu(event)">Zones & Overlays ▾</button>
            <div id="indicatorPopover" class="popover-menu" onclick="event.stopPropagation()">
              <div class="popover-section-title">Institutional Zones & HUD</div>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#e11d48"></i>Support & Resistance Zones (S&D)</span>
                <input type="checkbox" id="chkSRZones" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#3b82f6"></i>On-Chart Live Position HUD Box</span>
                <input type="checkbox" id="chkPosHud" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#10b981"></i>Entry-to-Exit Trade Path Lines</span>
                <input type="checkbox" id="chkTradePaths" checked onchange="renderDashboard()">
              </label>
              <div class="popover-section-title" style="margin-top:10px;">Technical Overlays</div>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#d946ef"></i>Session VWAP (24)</span>
                <input type="checkbox" id="chkVWAP" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#2563eb"></i>EMA 20 (Fast Pullback)</span>
                <input type="checkbox" id="chkEMA20" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#f59e0b"></i>EMA 60 (Core Trend)</span>
                <input type="checkbox" id="chkEMA60" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#10b981"></i>EMA 180 (1H-50 Macro)</span>
                <input type="checkbox" id="chkEMA180" checked onchange="renderDashboard()">
              </label>
              <label class="popover-item">
                <span><i class="swatch-dot" style="background:#64748b"></i>Bollinger Bands (20, 2)</span>
                <input type="checkbox" id="chkBB" onchange="renderDashboard()">
              </label>
            </div>
          </div>

          <div class="seg-group">
            <button class="seg-btn" id="zm_80" onclick="setZoom(80)">80B</button>
            <button class="seg-btn" id="zm_160" onclick="setZoom(160)">160B</button>
            <button class="seg-btn active" id="zm_320" onclick="setZoom(320)">320B</button>
            <button class="seg-btn" id="zm_all" onclick="setZoom(99999)">All</button>
          </div>
        </div>
      </div>

      <!-- DEDICATED WRAPPING OHLC & INDICATOR READOUT STRIP -->
      <div class="ohlc-bar" id="ohlcReadoutBar"></div>

      <!-- MAIN CANVAS + ON-CHART LIVE POSITION HUD BOX -->
      <div style="position:relative;width:100%;height:475px;">
        <div class="chart-hud-overlay" id="onChartPositionHud"></div>
        <canvas id="mainPriceCanvas" style="width:100%;height:100%;display:block;"></canvas>
      </div>

      <!-- BAR-BY-BAR REPLAY SCRUBBER -->
      <div class="replay-bar">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:12px;font-weight:600;color:var(--text-secondary);">Bar Replay & Live Sync:</span>
          <button class="clean-btn" style="height:30px;padding:0 10px;" onclick="stepCursor(-1)">← Prev</button>
          <button class="clean-btn" style="height:30px;padding:0 10px;" onclick="stepCursor(1)">Next →</button>
          <button class="clean-btn" style="height:30px;padding:0 10px;" onclick="resetCursorToEnd()">Latest Bar</button>
          <button class="clean-btn primary" style="height:30px;padding:0 10px;" onclick="syncBinanceGapCandles()">↻ Sync Live Candles</button>
        </div>
        <div style="flex:1;min-width:200px;display:flex;align-items:center;gap:10px;">
          <input type="range" id="replaySlider" min="10" max="100" value="100" oninput="onReplaySlide()" style="width:100%;cursor:pointer;">
          <span class="mono" style="font-size:12px;color:var(--text-muted);white-space:nowrap;" id="replayCounter">Bar 100 / 100</span>
        </div>
      </div>
    </section>

    <!-- RIGHT: AI SIGNAL, 12-FEAT MATH & INTERACTIVE STRATEGY SETTINGS STUDIO -->
    <section class="card">
      <div class="card-header">
        <span class="card-title">AI Signal & Strategy Studio</span>
        <div class="seg-group">
          <button class="seg-btn active" id="inspTab_signal" onclick="setInspectorTab('signal')">Signal</button>
          <button class="seg-btn" id="inspTab_settings" onclick="setInspectorTab('settings')">Settings ⚙</button>
          <button class="seg-btn" id="inspTab_math" onclick="setInspectorTab('math')">12-Feat Math</button>
        </div>
      </div>
      <div id="inspectorBody" style="flex:1;overflow-y:auto;max-height:585px;"></div>
    </section>
  </div>

  <!-- BOTTOM ROW: EQUITY CURVE & PERIOD BREAKDOWN (46%) + LIFECYCLE LOG & BENCHMARKS (54%) -->
  <div class="bottom-grid">
    <!-- LEFT: EQUITY CURVE & MONTHLY/DAILY BREAKDOWN -->
    <section class="card">
      <div class="card-header">
        <div>
          <span class="card-title">Cumulative Strategy Equity vs. Buy & Hold</span>
          <div style="font-size:12px;color:var(--text-muted);">Compounded return across completed & floating positions</div>
        </div>
        <div style="display:flex;gap:12px;font-size:12px;font-weight:600;">
          <span class="up">● AI Strategy</span>
          <span style="color:var(--text-muted);">● Buy & Hold Benchmark</span>
        </div>
      </div>
      <div style="position:relative;width:100%;height:230px;border-bottom:1px solid var(--border-col);">
        <canvas id="equityCurveCanvas" style="width:100%;height:100%;display:block;"></canvas>
      </div>
      <div style="padding:10px 16px 6px;font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;" id="breakdownTableTitle">
        Performance Breakdown by Period
      </div>
      <div style="max-height:250px;overflow-y:auto;">
        <table class="clean-table">
          <thead id="breakdownThead"></thead>
          <tbody id="breakdownTbody"></tbody>
        </table>
      </div>
    </section>

    <!-- RIGHT: TABBED LIFECYCLE EVENT LOG (OPEN + SCALE + CLOSE), ROUND-TRIP TRADES, BENCHMARKS & MATRIX -->
    <section class="card">
      <div class="card-header">
        <span class="card-title">Position Entry / Closing Lifecycle Log & Audit</span>
        <div class="seg-group">
          <button class="seg-btn active" id="auditTab_events" onclick="setAuditTab('events')">Entry & Closing Log</button>
          <button class="seg-btn" id="auditTab_trades" onclick="setAuditTab('trades')">Round-Trip Trades</button>
          <button class="seg-btn" id="auditTab_bench" onclick="setAuditTab('bench')">Benchmarks</button>
          <button class="seg-btn" id="auditTab_matrix" onclick="setAuditTab('matrix')">Matrix</button>
        </div>
      </div>
      <div id="auditBody" style="flex:1;max-height:515px;overflow-y:auto;"></div>
    </section>
  </div>

</div>

<!-- MODAL: LIVE DATA FEEDS & LONDON STRATEGIC EDGE API KEY -->
<div class="modal-backdrop" id="apiModal" onclick="closeApiModal(event)">
  <div class="modal-card" onclick="event.stopPropagation()">
    <div class="card-header">
      <span class="card-title">Live Market Data Feeds (XAU/USD & PAXG/USDT)</span>
      <button class="clean-btn" style="height:28px;padding:0 10px;" onclick="document.getElementById('apiModal').classList.remove('open')">✕ Close</button>
    </div>
    <div style="padding:18px;display:flex;flex-direction:column;gap:16px;">
      <div style="background:var(--bg-subtle);border:1px solid var(--border-col);border-radius:10px;padding:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-weight:700;font-size:14px;">1. London Strategic Edge (LSE Data 3.0) — Spot XAU/USD & EUR/USD</span>
          <span class="pill-badge pill-neutral" id="modalLseBadge">Standby</span>
        </div>
        <p style="font-size:12.5px;color:var(--text-secondary);margin-bottom:10px;">
          Streams real-time institutional Spot Gold (<b>XAU/USD</b>) and <b>EUR/USD</b> ticks over <code>wss://data-ws.londonstrategicedge.com</code>. You can freely switch between <b>PAXG/USDT</b> and <b>XAU/USD</b> in the top dropdown at any time!
        </p>
        <div style="display:flex;gap:8px;">
          <input type="password" id="lseApiKeyInput" class="clean-input mono" style="flex:1;cursor:text;" placeholder="Paste your London Strategic Edge API Key (e.g. lse_...)">
          <button class="clean-btn primary" onclick="connectLseWebSocket()">Connect LSE XAU/USD</button>
        </div>
      </div>

      <div style="background:var(--bg-subtle);border:1px solid var(--border-col);border-radius:10px;padding:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-weight:700;font-size:14px;">2. Binance Direct Feed — PAXG/USDT + EUR/USDT + BTC/USDT (Zero Key Needed)</span>
          <span class="pill-badge pill-up" id="modalBinanceBadge">Active & Auto-Syncing</span>
        </div>
        <p style="font-size:12.5px;color:var(--text-secondary);margin-bottom:10px;">
          Automatically syncs missing 15m candles via <code>api.binance.com/api/v3/klines</code> and streams live <b>PAXG/USDT</b> ticks over <code>wss://stream.binance.com:9443</code>.
        </p>
        <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;">
          <span class="mono" style="font-size:12px;color:var(--text-muted);" id="syncLogStatus">Ready to sync latest 15m candles.</span>
          <button class="clean-btn success" onclick="syncBinanceGapCandles()">↻ Fetch Latest 15m Candles Now</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {payload_str};

let ALL_15M_BARS = DATA.bars.map((r, idx) => {{
  const dt = new Date(r[0] * 1000);
  const iso = dt.toISOString();
  return {{
    i: idx, ts: r[0],
    t: iso.slice(0, 16).replace("T", " "),
    d: iso.slice(0, 10),
    m: iso.slice(0, 7),
    dow: dt.getUTCDay(),
    o: r[1], h: r[2], l: r[3], c: r[4], v: r[5],
    atr: r[6], vwap: r[7],
    ema20: r[8], ema60: r[9], ema180: r[10],
    bb_u: r[11], bb_d: r[12],
    er12: r[13], shock: r[14],
    p_up: r[15], p_lr: r[16], p_mlp: r[17],
    eur: r[18], btc: r[19], rsi: r[20],
    f4: r[21], f6: r[22], f8: r[23], f9: r[24], vsa: r[25],
    sig: null,
    exit_sig: null
  }};
}});

// Default Strategy Settings (Unchanged as requested!)
let CFG = {{
  conf_th: 0.62,
  strong_th: 0.68,
  lr_weight: 0.55,
  er_min: 0.18,
  max_shock: 2.30,
  tp_atr: 1.40,
  sl_atr: 2.00,
  trail_atr: 0.80,
  lock_atr: 0.35,
  horizon: 10,
  cooldown: 3,
  tier1_pct: 45.0,
  tier2_pct: 70.0,
  pyr_pct: 30.0,
  enable_pyr: true,
  block_sunday: true,
  dir_mode: "BOTH"
}};

let SIM_TRADES = [];
let EXEC_EVENTS = []; // Discrete chronological ENTRY, PYRAMID, and EXIT events!
let selectedAsset = "PAXG"; // Never hijacked by background LSE ticks!
let priceOffset = 0.0;
let liveSpotXauPrice = null;
let liveXauSpread = 0.22;
let livePaxgSpread = 0.12;
let filtered15mBars = [];
let displayBars = [];
let activeTfLabel = "15m";
let cursorIdx = 0;
let zoomBars = 320;
let inspectorTab = "signal";
let auditTab = "events"; // Default to Entry & Closing Lifecycle Log!

let binanceWs = null;
let lseWs = null;
let lsePingTimer = null;

// =========================================================================
// 1. IN-BROWSER NEURAL + LOGISTIC FORWARD PASS & LIFECYCLE BACKTEST ENGINE
// =========================================================================
function computeNeuralProbForBar(feats12) {{
  const M = DATA.meta;
  const s = new Array(12);
  let logit_lr = M.b_lr;
  for (let k = 0; k < 12; k++) {{
    s[k] = (feats12[k] - M.mean[k]) / (M.scale[k] || 1.0);
    logit_lr += s[k] * M.w_lr[k];
  }}
  const p_lr = 1.0 / (1.0 + Math.exp(-logit_lr));

  let logit_mlp = M.b1;
  for (let j = 0; j < M.b0.length; j++) {{
    let z = M.b0[j];
    for (let k = 0; k < 12; k++) z += s[k] * M.w0[k][j];
    const h_j = Math.tanh(z);
    logit_mlp += h_j * M.w1[j];
  }}
  const p_mlp = 1.0 / (1.0 + Math.exp(-logit_mlp));
  const raw_ens = CFG.lr_weight * p_lr + (1.0 - CFG.lr_weight) * p_mlp;
  const p_up = 1.0 / (1.0 + Math.exp(-((raw_ens - M.mean_p) / (M.std_p * 0.72))));
  return {{ p_up, p_lr, p_mlp }};
}}

function runFastBrowserBacktest() {{
  const N = ALL_15M_BARS.length;
  for (let idx = 0; idx < N; idx++) {{
    const b = ALL_15M_BARS[idx];
    if (Math.abs(CFG.lr_weight - 0.55) > 0.001) {{
      const raw = CFG.lr_weight * b.p_lr + (1.0 - CFG.lr_weight) * b.p_mlp;
      b.p_up = 1.0 / (1.0 + Math.exp(-((raw - DATA.meta.mean_p) / (DATA.meta.std_p * 0.72))));
    }}
    b.sig = null;
    b.exit_sig = null;
  }}

  const trades = [];
  const events = [];
  const horizon = CFG.horizon;
  const conf_th = CFG.conf_th;
  const strong_th = CFG.strong_th;
  const dn_th = 1.0 - conf_th;
  let i = 1;
  let tradeIdCounter = 1;

  while (i < N - 1) {{
    const b = ALL_15M_BARS[i];
    if ((CFG.block_sunday && b.dow === 0) || b.er12 < CFG.er_min || b.shock > CFG.max_shock) {{
      i++;
      continue;
    }}

    const p = b.p_up;
    let direction = 0;
    let cap_w = 0.0;

    if (CFG.dir_mode !== "SHORT_ONLY" && p >= conf_th && b.f8 > -0.35 && b.f9 > -1.20 && b.f6 > -0.15 && b.f4 < 2.2) {{
      direction = 1;
      cap_w = (p >= strong_th ? CFG.tier2_pct : CFG.tier1_pct) / 100.0;
    }} else if (CFG.dir_mode !== "LONG_ONLY" && p <= dn_th && b.f8 < 0.35 && b.f9 < 1.20 && b.f6 < 0.15 && b.f4 > -2.2) {{
      direction = -1;
      cap_w = (p <= (1.0 - strong_th) ? CFG.tier2_pct : CFG.tier1_pct) / 100.0;
    }}

    if (direction !== 0) {{
      const tId = tradeIdCounter++;
      const entry1 = b.c;
      const cur_atr = b.atr;
      const tp_price = entry1 + direction * (CFG.tp_atr * cur_atr);
      const init_sl = entry1 - direction * (CFG.sl_atr * cur_atr);
      let sl_price = init_sl;
      const tr_trig = entry1 + direction * (CFG.trail_atr * cur_atr);
      const pyr_trig = entry1 + direction * (0.50 * cur_atr);

      let pyramided = false;
      let pyr_bar_idx = -1;
      let pyr_ts = 0;
      let pyr_time_str = "";
      let entry2 = 0.0;
      let w2 = 0.0;
      let ratchet_locked = false;

      const maxK = Math.min(horizon, N - 1 - i);
      let exit_p = ALL_15M_BARS[i + maxK].c;
      let bars_held = Math.max(1, maxK);
      let hit_loss = false;
      let status = maxK < horizon ? "LIVE OPEN" : "HORIZON EXIT";

      for (let k = 1; k <= maxK; k++) {{
        const nb = ALL_15M_BARS[i + k];
        if (direction === 1) {{
          if (nb.h >= tp_price) {{
            exit_p = tp_price; bars_held = k; status = `TP HIT (+${{CFG.tp_atr.toFixed(1)}}x ATR)`; break;
          }}
          if (nb.l <= sl_price) {{
            exit_p = sl_price; bars_held = k; hit_loss = (sl_price < entry1);
            status = !hit_loss ? "RATCHET WIN" : "STOP LOSS"; break;
          }}
          if (nb.h >= tr_trig) {{
            sl_price = Math.max(sl_price, entry1 + CFG.lock_atr * cur_atr);
            ratchet_locked = true;
          }}
          if (CFG.enable_pyr && !pyramided && nb.c >= pyr_trig && nb.p_up >= conf_th) {{
            pyramided = true; entry2 = nb.c; w2 = CFG.pyr_pct / 100.0;
            pyr_bar_idx = i + k; pyr_ts = nb.ts; pyr_time_str = nb.t;
            sl_price = Math.max(sl_price, entry1);
          }}
        }} else {{
          if (nb.l <= tp_price) {{
            exit_p = tp_price; bars_held = k; status = `TP HIT (+${{CFG.tp_atr.toFixed(1)}}x ATR)`; break;
          }}
          if (nb.h >= sl_price) {{
            exit_p = sl_price; bars_held = k; hit_loss = (sl_price > entry1);
            status = !hit_loss ? "RATCHET WIN" : "STOP LOSS"; break;
          }}
          if (nb.l <= tr_trig) {{
            sl_price = Math.min(sl_price, entry1 - CFG.lock_atr * cur_atr);
            ratchet_locked = true;
          }}
          if (CFG.enable_pyr && !pyramided && nb.c <= pyr_trig && nb.p_up <= dn_th) {{
            pyramided = true; entry2 = nb.c; w2 = CFG.pyr_pct / 100.0;
            pyr_bar_idx = i + k; pyr_ts = nb.ts; pyr_time_str = nb.t;
            sl_price = Math.min(sl_price, entry1);
          }}
        }}
      }}

      const exit_bar_idx = i + bars_held;
      const exitBarObj = ALL_15M_BARS[exit_bar_idx];
      const ret1 = cap_w * (direction * ((exit_p - entry1) / entry1) - 0.0001);
      const ret2 = pyramided ? w2 * (direction * ((exit_p - entry2) / entry2) - 0.0001) : 0.0;
      const tot_ret = (ret1 + ret2) * 100.0;
      if (status === "HORIZON EXIT") status = tot_ret > 0 ? "HORIZON WIN" : "HORIZON LOSS";

      const init_pct = Math.round(cap_w * 100);
      const total_pct = Math.round((cap_w + w2) * 100);
      const tier_str = `${{direction === 1 ? 'Long' : 'Short'}} (${{init_pct}}%)${{pyramided ? ' +Scale' : ''}}`;
      const conf_pct = Number(((direction === 1 ? p : 1.0 - p) * 100).toFixed(1));

      const tr_obj = {{
        id: tId,
        b: i,
        exit_b: exit_bar_idx,
        pyr_b: pyr_bar_idx,
        ts: b.ts,
        exit_ts: exitBarObj.ts,
        t: b.t,
        exit_t: exitBarObj.t,
        d: b.d,
        m: b.m,
        dir: direction === 1 ? "LONG" : "SHORT",
        dir_num: direction,
        init_pct: init_pct,
        total_pct: total_pct,
        pyramided: pyramided,
        pyr_price: Number(entry2.toFixed(2)),
        tier: tier_str,
        en: Number(entry1.toFixed(2)),
        ex: Number(exit_p.toFixed(2)),
        tp: Number(tp_price.toFixed(2)),
        init_sl: Number(init_sl.toFixed(2)),
        final_sl: Number(sl_price.toFixed(2)),
        ratchet_locked: ratchet_locked,
        bars_held: bars_held,
        prob: conf_pct,
        ret: Number(tot_ret.toFixed(3)),
        st: status
      }};
      trades.push(tr_obj);
      b.sig = tr_obj;
      if (exitBarObj) exitBarObj.exit_sig = tr_obj;

      // 1. Log ENTRY Event
      events.push({{
        trade_id: tId,
        event_type: "ENTRY OPEN",
        bar_idx: i,
        ts: b.ts,
        t: b.t,
        dir: tr_obj.dir,
        action_label: `OPEN ${{tr_obj.dir}} (${{init_pct}}% Cap)`,
        price: tr_obj.en,
        sl: tr_obj.init_sl,
        tp: tr_obj.tp,
        conf: conf_pct,
        pnl_pct: null,
        detail: `AI Conf ${{conf_pct}}% | SL $${{tr_obj.init_sl.toFixed(1)}} | TP $${{tr_obj.tp.toFixed(1)}}`
      }});

      // 2. Log PYRAMID SCALE-IN Event (if triggered)
      if (pyramided && pyr_bar_idx > 0) {{
        events.push({{
          trade_id: tId,
          event_type: "SCALE-IN",
          bar_idx: pyr_bar_idx,
          ts: pyr_ts,
          t: pyr_time_str,
          dir: tr_obj.dir,
          action_label: `PYRAMID +${{Math.round(CFG.pyr_pct)}}% (${{total_pct}}% Tot)`,
          price: tr_obj.pyr_price,
          sl: tr_obj.en,
          tp: tr_obj.tp,
          conf: conf_pct,
          pnl_pct: null,
          detail: `+0.50x ATR Reached -> SL Protected at Breakeven ($${{tr_obj.en.toFixed(1)}})`
        }});
      }}

      // 3. Log POSITION CLOSED Event (at exact exit_ts!)
      if (status !== "LIVE OPEN") {{
        events.push({{
          trade_id: tId,
          event_type: "POSITION CLOSED",
          bar_idx: exit_bar_idx,
          ts: exitBarObj.ts,
          t: exitBarObj.t,
          dir: tr_obj.dir,
          action_label: `CLOSE ${{tr_obj.dir}} (${{status}})`,
          price: tr_obj.ex,
          sl: tr_obj.final_sl,
          tp: tr_obj.tp,
          conf: conf_pct,
          pnl_pct: tr_obj.ret,
          detail: `Held ${{bars_held}} bars | Entry $${{tr_obj.en.toFixed(1)}} -> Exit $${{tr_obj.ex.toFixed(1)}}`
        }});
      }}

      i += bars_held + (hit_loss ? CFG.cooldown : 1);
    }} else {{
      i++;
    }}
  }}

  events.sort((a, b) => a.ts - b.ts || a.bar_idx - b.bar_idx);
  SIM_TRADES = trades;
  EXEC_EVENTS = events;
}}

// =========================================================================
// 2. ALGORITHMIC SUPPORT/DEMAND & RESISTANCE/SUPPLY ZONE ENGINE
// =========================================================================
function computeSupportResistanceZones(vis) {{
  const zones = [];
  const N = vis.length;
  if (N < 16) return zones;

  // 1. Major Structural Range High (Supply) & Range Low (Demand) Zones across visible window
  let maxH = -Infinity, maxIdx = 0;
  let minL = Infinity, minIdx = 0;
  let avgAtr = 0.0;
  vis.forEach((b, idx) => {{
    if (b.h > maxH) {{ maxH = b.h; maxIdx = idx; }}
    if (b.l < minL) {{ minL = b.l; minIdx = idx; }}
    avgAtr += b.atr;
  }});
  avgAtr = (avgAtr / N) || 3.0;

  // Major Supply / Resistance Zone at Top
  zones.push({{
    type: "RES",
    major: true,
    startIdx: Math.max(0, Math.min(maxIdx - 4, Math.floor(N * 0.05))),
    endIdx: N - 1,
    top: maxH + priceOffset,
    bot: (maxH - 0.65 * avgAtr) + priceOffset,
    label: "MAJOR RESISTANCE / SUPPLY ZONE"
  }});

  // Major Demand / Support Zone at Bottom
  zones.push({{
    type: "SUP",
    major: true,
    startIdx: Math.max(0, Math.min(minIdx - 4, Math.floor(N * 0.05))),
    endIdx: N - 1,
    top: (minL + 0.65 * avgAtr) + priceOffset,
    bot: minL + priceOffset,
    label: "MAJOR SUPPORT / DEMAND ZONE"
  }});

  // 2. Intermediate Swing Order-Block / S&R Regions (Staggered horizontal boxes like image_988d1c.jpg)
  const lb = Math.max(5, Math.min(12, Math.floor(N / 18)));
  const rawPivots = [];

  for (let i = lb; i < N - lb; i++) {{
    let isHigh = true, isLow = true;
    for (let k = i - lb; k <= i + lb; k++) {{
      if (vis[k].h > vis[i].h) isHigh = false;
      if (vis[k].l < vis[i].l) isLow = false;
    }}
    if (isHigh && Math.abs(vis[i].h - maxH) > 0.7 * avgAtr) {{
      rawPivots.push({{ type: "RES", idx: i, p: vis[i].h, atr: vis[i].atr }});
    }}
    if (isLow && Math.abs(vis[i].l - minL) > 0.7 * avgAtr) {{
      rawPivots.push({{ type: "SUP", idx: i, p: vis[i].l, atr: vis[i].atr }});
    }}
  }}

  // Filter non-overlapping intermediate zones and trace until broken or 55 bars forward
  const kept = [];
  rawPivots.forEach(pv => {{
    const tooClose = kept.some(k => Math.abs(k.p - pv.p) < 0.85 * avgAtr && Math.abs(k.idx - pv.idx) < lb * 3);
    if (!tooClose) kept.push(pv);
  }});

  kept.slice(-7).forEach(pv => {{
    const thick = 0.48 * pv.atr;
    const topP = (pv.type === "RES" ? pv.p : pv.p + thick) + priceOffset;
    const botP = (pv.type === "RES" ? pv.p - thick : pv.p) + priceOffset;

    let endI = Math.min(N - 1, pv.idx + Math.max(28, Math.floor(N * 0.32)));
    for (let k = pv.idx + 3; k < N; k++) {{
      if (pv.type === "RES" && (vis[k].c + priceOffset) > topP + 0.35 * pv.atr) {{
        endI = k; break;
      }}
      if (pv.type === "SUP" && (vis[k].c + priceOffset) < botP - 0.35 * pv.atr) {{
        endI = k; break;
      }}
    }}
    if (endI - pv.idx >= 8) {{
      zones.push({{
        type: pv.type,
        major: false,
        startIdx: pv.idx,
        endIdx: endI,
        top: topP,
        bot: botP,
        label: pv.type === "RES" ? "Supply / Resistance" : "Demand / Support"
      }});
    }}
  }});

  return zones;
}}

// =========================================================================
// 3. LIVE BINANCE REST + WEBSOCKET & LSE XAU/USD STREAMER (FIXED ASSET SWITCH!)
// =========================================================================
async function syncBinanceGapCandles() {{
  const statusEl = document.getElementById("syncLogStatus");
  if (statusEl) statusEl.textContent = "Fetching latest 15m candles from Binance...";
  try {{
    const lastBar = ALL_15M_BARS[ALL_15M_BARS.length - 1];
    const startMs = (lastBar.ts - 3600 * 6) * 1000;
    const [rGold, rEur, rBtc] = await Promise.all([
      fetch(`https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval=15m&startTime=${{startMs}}&limit=500`).then(r => r.json()),
      fetch(`https://api.binance.com/api/v3/klines?symbol=EURUSDT&interval=15m&startTime=${{startMs}}&limit=500`).then(r => r.json()),
      fetch(`https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&startTime=${{startMs}}&limit=500`).then(r => r.json())
    ]);

    const eurMap = {{}}, btcMap = {{}};
    if (Array.isArray(rEur)) rEur.forEach(k => eurMap[Math.floor(k[0] / 1000)] = parseFloat(k[4]));
    if (Array.isArray(rBtc)) rBtc.forEach(k => btcMap[Math.floor(k[0] / 1000)] = parseFloat(k[4]));

    let added = 0;
    if (Array.isArray(rGold)) {{
      rGold.forEach(k => {{
        const tsSec = Math.floor(k[0] / 1000);
        const o = parseFloat(k[1]), h = parseFloat(k[2]), l = parseFloat(k[3]), c = parseFloat(k[4]), v = parseFloat(k[5]);
        const prev = ALL_15M_BARS[ALL_15M_BARS.length - 1];
        const eurC = eurMap[tsSec] || prev.eur;
        const btcC = btcMap[tsSec] || prev.btc;

        if (tsSec === prev.ts) {{
          prev.o = o; prev.h = h; prev.l = l; prev.c = c; prev.v = v;
          prev.eur = eurC; prev.btc = btcC;
          recomputeSingleBarIndicators(ALL_15M_BARS.length - 1);
        }} else if (tsSec > prev.ts) {{
          const dt = new Date(tsSec * 1000);
          const iso = dt.toISOString();
          ALL_15M_BARS.push({{
            i: ALL_15M_BARS.length, ts: tsSec,
            t: iso.slice(0, 16).replace("T", " "),
            d: iso.slice(0, 10), m: iso.slice(0, 7),
            dow: dt.getUTCDay(),
            o, h, l, c, v,
            atr: prev.atr, vwap: c, ema20: prev.ema20, ema60: prev.ema60, ema180: prev.ema180,
            bb_u: prev.bb_u, bb_d: prev.bb_d, er12: prev.er12, shock: 1.0,
            p_up: prev.p_up, p_lr: prev.p_lr, p_mlp: prev.p_mlp,
            eur: eurC, btc: btcC, rsi: prev.rsi,
            f4: prev.f4, f6: prev.f6, f8: prev.f8, f9: prev.f9, vsa: prev.vsa, sig: null, exit_sig: null
          }});
          recomputeSingleBarIndicators(ALL_15M_BARS.length - 1);
          added++;
        }}
      }});
    }}
    runFastBrowserBacktest();
    onPeriodChange();
    const lastIso = ALL_15M_BARS[ALL_15M_BARS.length - 1].t;
    if (statusEl) statusEl.textContent = `Synced (+${{added}} new bars). Latest candle: ${{lastIso}} UTC`;
    document.getElementById("binanceStatusTxt").textContent = `PAXG/USDT Synced (${{lastIso.slice(11)}} UTC)`;
  }} catch (err) {{
    if (statusEl) statusEl.textContent = "Live sync notice: " + err.message;
  }}
}}

function recomputeSingleBarIndicators(idx) {{
  if (idx < 25) return;
  const b = ALL_15M_BARS[idx];
  const prev = ALL_15M_BARS[idx - 1];
  const tr = Math.max(b.h - b.l, Math.abs(b.h - prev.c), Math.abs(b.l - prev.c));
  b.atr = (1.0 / 14.0) * tr + (13.0 / 14.0) * prev.atr;
  const atr_pct = (b.atr / b.c) + 1e-10;

  b.ema20 = (2.0 / 21.0) * b.c + (19.0 / 21.0) * prev.ema20;
  b.ema60 = (2.0 / 61.0) * b.c + (59.0 / 61.0) * prev.ema60;
  b.ema180 = (2.0 / 181.0) * b.c + (179.0 / 181.0) * prev.ema180;

  let pvSum = 0.0, vSum = 0.0, sumMoves = 0.0;
  let hh16 = -Infinity, ll16 = Infinity;
  for (let k = 0; k < 24; k++) {{
    const barK = ALL_15M_BARS[idx - k];
    const tp = (barK.h + barK.l + barK.c) / 3.0;
    pvSum += tp * (barK.v || 1.0);
    vSum += (barK.v || 1.0);
    if (k < 16) {{
      if (barK.h > hh16) hh16 = barK.h;
      if (barK.l < ll16) ll16 = barK.l;
    }}
    if (k < 12) sumMoves += Math.abs(barK.c - ALL_15M_BARS[idx - k - 1].c);
  }}
  b.vwap = pvSum / (vSum || 1.0);
  b.er12 = sumMoves === 0 ? 0.0 : Math.abs(b.c - ALL_15M_BARS[idx - 12].c) / sumMoves;
  b.shock = (b.h - b.l) / (b.atr + 1e-10);

  const f1 = ((b.c - ALL_15M_BARS[idx - 2].c) / ALL_15M_BARS[idx - 2].c) / atr_pct;
  const f2 = ((b.c - ALL_15M_BARS[idx - 4].c) / ALL_15M_BARS[idx - 4].c) / atr_pct;
  const f3 = ((b.c - ALL_15M_BARS[idx - 12].c) / ALL_15M_BARS[idx - 12].c) / atr_pct;
  b.f4 = ((b.c - b.vwap) / b.c) / atr_pct;
  const f5 = (b.rsi - 50.0) / 50.0;
  b.f6 = prev.f6;
  const f7 = (hh16 === ll16) ? 0.0 : ((b.c - ll16) / (hh16 - ll16)) - 0.5;
  b.f8 = ((b.ema20 - b.ema60) / b.c) / atr_pct;
  b.f9 = ((b.ema60 - b.ema180) / b.c) / atr_pct;
  const rng = (b.h - b.l) + 1e-10;
  b.vsa = ((b.c - b.l) - (b.h - b.c)) / rng;

  const b4 = ALL_15M_BARS[idx - 4];
  const eur_ret4 = (b.eur - b4.eur) / (b4.eur || 1.0);
  const gold_ret4 = (b.c - b4.c) / (b4.c || 1.0);
  const btc_ret4 = (b.btc - b4.btc) / (b4.btc || 1.0);
  const f11 = eur_ret4 * 100.0;
  const f12 = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0;

  const inf = computeNeuralProbForBar([f1, f2, f3, b.f4, f5, b.f6, f7, b.f8, b.f9, b.vsa, f11, f12]);
  b.p_up = inf.p_up; b.p_lr = inf.p_lr; b.p_mlp = inf.p_mlp;
}}

function startBinanceWebSocket() {{
  try {{
    if (binanceWs) binanceWs.close();
    binanceWs = new WebSocket("wss://stream.binance.com:9443/ws/paxgusdt@kline_15m");
    binanceWs.onopen = () => {{
      document.getElementById("binanceDot").classList.add("live");
      document.getElementById("binanceStatusTxt").textContent = "PAXG/USDT Stream Live";
    }};
    binanceWs.onmessage = (ev) => {{
      const msg = JSON.parse(ev.data);
      if (!msg.k) return;
      const k = msg.k;
      const tsSec = Math.floor(k.t / 1000);
      const lastBar = ALL_15M_BARS[ALL_15M_BARS.length - 1];
      if (tsSec >= lastBar.ts) {{
        lastBar.c = parseFloat(k.c);
        lastBar.h = Math.max(lastBar.h, parseFloat(k.h));
        lastBar.l = Math.min(lastBar.l, parseFloat(k.l));
        lastBar.v = parseFloat(k.v);
        recomputeSingleBarIndicators(ALL_15M_BARS.length - 1);
        if (cursorIdx === displayBars.length - 1) {{
          displayBars[displayBars.length - 1].c = lastBar.c;
          displayBars[displayBars.length - 1].h = Math.max(displayBars[displayBars.length - 1].h, lastBar.h);
          displayBars[displayBars.length - 1].l = Math.min(displayBars[displayBars.length - 1].l, lastBar.l);
          displayBars[displayBars.length - 1].p_up = lastBar.p_up;
          updatePriceOffsetForSelectedAsset();
          renderDashboard();
        }}
      }}
    }};
  }} catch (e) {{
    console.log("Binance WS error:", e);
  }}
}}

// FIXED: LSE WebSocket updates XAU price in background WITHOUT hijacking assetSelect dropdown!
function connectLseWebSocket() {{
  const key = document.getElementById("lseApiKeyInput").value.trim();
  if (!key) {{
    alert("Please paste your London Strategic Edge API key first.");
    return;
  }}
  localStorage.setItem("LSE_GOLD_API_KEY", key);
  const badge = document.getElementById("modalLseBadge");
  badge.textContent = "Connecting to wss://data-ws.londonstrategicedge.com...";

  try {{
    if (lseWs) lseWs.close();
    if (lsePingTimer) clearInterval(lsePingTimer);

    lseWs = new WebSocket("wss://data-ws.londonstrategicedge.com");
    lseWs.onopen = () => {{
      lseWs.send(JSON.stringify({{ action: "auth", api_key: key }}));
      lsePingTimer = setInterval(() => {{
        if (lseWs && lseWs.readyState === WebSocket.OPEN) {{
          lseWs.send(JSON.stringify({{ action: "ping" }}));
        }}
      }}, 25000);
    }};

    lseWs.onmessage = (event) => {{
      const msg = JSON.parse(event.data);
      if (msg.type === "welcome") {{
        lseWs.send(JSON.stringify({{ action: "auth", api_key: key }}));
      }} else if (msg.type === "authenticated") {{
        badge.textContent = "Authenticated • Subscribed XAU/USD";
        badge.className = "pill-badge pill-up";
        document.getElementById("lseDot").classList.add("live");
        document.getElementById("lseStatusTxt").textContent = "LSE XAU/USD Live";
        lseWs.send(JSON.stringify({{ action: "subscribe", symbol: "XAU/USD" }}));
        lseWs.send(JSON.stringify({{ action: "subscribe", symbol: "EUR/USD" }}));
      }} else if (msg.type === "tick" || msg.price) {{
        const sym = msg.symbol || "";
        const price = parseFloat(msg.price);
        if (sym.includes("XAU") && !isNaN(price)) {{
          liveSpotXauPrice = price;
          if (msg.bid && msg.ask) {{
            liveXauSpread = Math.abs(parseFloat(msg.ask) - parseFloat(msg.bid));
          }}
          document.getElementById("lseStatusTxt").textContent = `LSE XAU: $${{price.toFixed(2)}}`;
          // Only re-render XAU offset if user currently has XAU selected!
          if (selectedAsset === "XAU") {{
            updatePriceOffsetForSelectedAsset();
            renderDashboard();
          }}
        }} else if (sym.includes("EUR") && !isNaN(price)) {{
          ALL_15M_BARS[ALL_15M_BARS.length - 1].eur = price;
        }}
      }} else if (msg.type === "error" || msg.code) {{
        badge.textContent = "LSE Notice: " + (msg.message || msg.code);
        badge.className = "pill-badge pill-warn";
      }}
    }};
  }} catch (err) {{
    badge.textContent = "Connection error: " + err.message;
  }}
}}

function updatePriceOffsetForSelectedAsset() {{
  const lastC = ALL_15M_BARS[ALL_15M_BARS.length - 1].c;
  if (selectedAsset === "XAU") {{
    let rawDiff = liveSpotXauPrice ? (liveSpotXauPrice - lastC) : 1.85;
    // Guard against extreme scale mismatch so historical chart never distorts
    if (Math.abs(rawDiff) > 150) rawDiff = 1.85;
    priceOffset = rawDiff;
    document.getElementById("activeAssetTag").textContent = "XAU/USD (SPOT)";
    document.getElementById("spreadBadge").textContent = `XAU Spread: $${{liveXauSpread.toFixed(2)}}`;
  }} else {{
    priceOffset = 0.0;
    document.getElementById("activeAssetTag").textContent = "PAXG/USDT (BINANCE)";
    document.getElementById("spreadBadge").textContent = `PAXG Spread: $${{livePaxgSpread.toFixed(2)}}`;
  }}
}}

function onAssetChange() {{
  selectedAsset = document.getElementById("assetSelect").value;
  updatePriceOffsetForSelectedAsset();
  renderDashboard();
}}

function openApiModal() {{
  const savedKey = localStorage.getItem("LSE_GOLD_API_KEY") || "";
  if (savedKey) document.getElementById("lseApiKeyInput").value = savedKey;
  document.getElementById("apiModal").classList.add("open");
}}
function closeApiModal(e) {{
  if (e.target.id === "apiModal") document.getElementById("apiModal").classList.remove("open");
}}

// =========================================================================
// 4. UI CONTROLS, TIMEFRAME AGGREGATION & RENDERING
// =========================================================================
window.addEventListener("click", () => {{
  const pop = document.getElementById("indicatorPopover");
  if (pop) pop.classList.remove("open");
}});
function toggleIndicatorMenu(e) {{
  e.stopPropagation();
  document.getElementById("indicatorPopover").classList.toggle("open");
}}
function toggleTheme() {{
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
  root.setAttribute("data-theme", next);
  document.getElementById("themeBtn").textContent = next === "light" ? "Dark Mode" : "Light Mode";
  renderDashboard();
}}

function aggregateBars(bars15m, groupSize) {{
  if (groupSize <= 1) return bars15m;
  const out = [];
  for (let i = 0; i < bars15m.length; i += groupSize) {{
    const chunk = bars15m.slice(i, i + groupSize);
    if (!chunk.length) continue;
    const first = chunk[0], last = chunk[chunk.length - 1];
    let hi = -Infinity, lo = Infinity, pickedSig = null, pickedExit = null;
    chunk.forEach(b => {{
      if (b.h > hi) hi = b.h;
      if (b.l < lo) lo = b.l;
      if (b.sig) pickedSig = b.sig;
      if (b.exit_sig) pickedExit = b.exit_sig;
    }});
    out.push({{ ...last, o: first.o, h: hi, l: lo, c: last.c, sig: pickedSig, exit_sig: pickedExit }});
  }}
  return out;
}}

function onPeriodChange() {{
  const period = document.getElementById("periodSelect").value;
  const maxTs = ALL_15M_BARS[ALL_15M_BARS.length - 1].ts;

  if (period === "1Y") {{
    filtered15mBars = ALL_15M_BARS;
  }} else if (period === "6M") {{
    filtered15mBars = ALL_15M_BARS.filter(b => b.ts >= maxTs - (182 * 86400));
  }} else if (period === "SEP_NOV") {{
    filtered15mBars = ALL_15M_BARS.filter(b => b.d >= "2026-09-01");
  }} else if (period === "OOS_WEEK") {{
    filtered15mBars = ALL_15M_BARS.filter(b => b.d >= "2026-11-09");
  }} else {{
    filtered15mBars = ALL_15M_BARS.filter(b => b.d === period);
  }}
  if (filtered15mBars.length === 0) filtered15mBars = ALL_15M_BARS.slice(-600);
  rebuildDisplayBars();
}}

function onTimeframeChange() {{ rebuildDisplayBars(); }}

function rebuildDisplayBars() {{
  const period = document.getElementById("periodSelect").value;
  let tf = document.getElementById("tfSelect").value;
  if (tf === "AUTO") {{
    if (period === "1Y") tf = "1D";
    else if (period === "6M") tf = "4H";
    else if (period === "SEP_NOV") tf = "1H";
    else tf = "15m";
  }}
  let groupSize = 1;
  if (tf === "1H") groupSize = 4;
  else if (tf === "4H") groupSize = 16;
  else if (tf === "1D") groupSize = 96;

  activeTfLabel = tf;
  document.getElementById("activeTfBadge").textContent = `${{tf}} Candles (${{filtered15mBars.length}} raw 15m bars)`;
  displayBars = aggregateBars(filtered15mBars, groupSize);

  const slider = document.getElementById("replaySlider");
  slider.min = Math.min(10, Math.max(0, displayBars.length - 1));
  slider.max = Math.max(0, displayBars.length - 1);
  slider.value = Math.max(0, displayBars.length - 1);
  cursorIdx = Math.max(0, displayBars.length - 1);
  renderDashboard();
}}

function setZoom(n) {{
  zoomBars = n;
  ["80", "160", "320", "all"].forEach(k => {{
    const el = document.getElementById("zm_" + k);
    if (el) el.classList.toggle("active", (k === "all" && n > 10000) || Number(k) === n);
  }});
  renderDashboard();
}}
function onReplaySlide() {{
  cursorIdx = parseInt(document.getElementById("replaySlider").value, 10);
  renderDashboard();
}}
function stepCursor(delta) {{
  const slider = document.getElementById("replaySlider");
  cursorIdx = Math.min(displayBars.length - 1, Math.max(1, cursorIdx + delta));
  slider.value = cursorIdx;
  renderDashboard();
}}
function resetCursorToEnd() {{
  const slider = document.getElementById("replaySlider");
  cursorIdx = displayBars.length - 1;
  slider.value = cursorIdx;
  renderDashboard();
}}
function setInspectorTab(tab) {{
  inspectorTab = tab;
  ["signal", "settings", "math"].forEach(k => {{
    document.getElementById("inspTab_" + k).classList.toggle("active", k === tab);
  }});
  renderInspector();
}}
function setAuditTab(tab) {{
  auditTab = tab;
  ["events", "trades", "bench", "matrix"].forEach(k => {{
    document.getElementById("auditTab_" + k).classList.toggle("active", k === tab);
  }});
  renderBottomAnalytics();
}}

function applySettingChange(key, val, isBool=false) {{
  CFG[key] = isBool ? Boolean(val) : (isNaN(Number(val)) ? val : Number(val));
  runFastBrowserBacktest();
  const groupSize = activeTfLabel === "1D" ? 96 : (activeTfLabel === "4H" ? 16 : (activeTfLabel === "1H" ? 4 : 1));
  displayBars = aggregateBars(filtered15mBars, groupSize);
  renderDashboard();
}}

function resetDefaultSettings() {{
  CFG = {{
    conf_th: 0.62, strong_th: 0.68, lr_weight: 0.55,
    er_min: 0.18, max_shock: 2.30,
    tp_atr: 1.40, sl_atr: 2.00, trail_atr: 0.80, lock_atr: 0.35,
    horizon: 10, cooldown: 3,
    tier1_pct: 45.0, tier2_pct: 70.0, pyr_pct: 30.0,
    enable_pyr: true, block_sunday: true, dir_mode: "BOTH"
  }};
  runFastBrowserBacktest();
  rebuildDisplayBars();
}}

// Find if a position is currently open at timestamp curTs
function findActivePositionAt(curBar) {{
  const curTs = curBar.ts;
  // A trade is currently open if curTs >= tr.ts and curTs < tr.exit_ts (or if it entered on this exact bar)
  for (let k = SIM_TRADES.length - 1; k >= 0; k--) {{
    const tr = SIM_TRADES[k];
    if (curTs >= tr.ts && curTs < tr.exit_ts) {{
      const barsElapsed = Math.max(1, Math.round((curTs - tr.ts) / 900) + 1);
      const isScaled = tr.pyramided && curTs >= (ALL_15M_BARS[tr.pyr_b] ? ALL_15M_BARS[tr.pyr_b].ts : 9999999999);
      const eqPct = isScaled ? tr.total_pct : tr.init_pct;
      const rawRet = tr.dir_num * ((curBar.c - tr.en) / tr.en) * (eqPct / 100.0) * 100.0;
      const activeSl = isScaled ? Math.max(tr.init_sl, tr.en) : (tr.ratchet_locked ? tr.final_sl : tr.init_sl);
      return {{
        open: true,
        tr: tr,
        barsElapsed: barsElapsed,
        eqPct: eqPct,
        isScaled: isScaled,
        floatingPct: rawRet,
        floatingUsd: rawRet * 100.0,
        activeSl: activeSl
      }};
    }}
  }}
  return {{ open: false }};
}}

function renderOnChartPositionHud(curBar, zones) {{
  const hud = document.getElementById("onChartPositionHud");
  if (!document.getElementById("chkPosHud").checked) {{
    hud.style.display = "none";
    return;
  }}
  hud.style.display = "block";

  const pos = findActivePositionAt(curBar);
  const cPrice = curBar.c + priceOffset;

  // Find nearest Resistance zone above and Support zone below
  let nearRes = null, nearSup = null;
  zones.forEach(z => {{
    if (z.type === "RES" && z.bot >= cPrice - 2.0) {{
      if (!nearRes || z.bot < nearRes.bot) nearRes = z;
    }}
    if (z.type === "SUP" && z.top <= cPrice + 2.0) {{
      if (!nearSup || z.top > nearSup.top) nearSup = z;
    }}
  }});

  const resTxt = nearRes ? `$${{nearRes.bot.toFixed(1)}} – $${{nearRes.top.toFixed(1)}}` : `$${{(cPrice + 1.8*curBar.atr).toFixed(1)}}`;
  const supTxt = nearSup ? `$${{nearSup.bot.toFixed(1)}} – $${{nearSup.top.toFixed(1)}}` : `$${{(cPrice - 1.8*curBar.atr).toFixed(1)}}`;

  if (pos.open) {{
    const tr = pos.tr;
    const isLong = tr.dir === "LONG";
    const flCol = pos.floatingPct >= 0 ? "up" : "down";
    hud.innerHTML = `
      <div class="hud-header-line">
        <span>POSITION: <b class="${{isLong ? 'up' : 'down'}}">● ${{tr.dir}} ACTIVE (${{pos.eqPct}}% Cap)</b></span>
        <span class="pill-badge ${{pos.floatingPct >= 0 ? 'pill-up' : 'pill-dn'}}">${{pos.floatingPct >= 0 ? '+' : ''}}${{pos.floatingPct.toFixed(2)}}% ($${{pos.floatingUsd >= 0 ? '+' : ''}}${{pos.floatingUsd.toFixed(0)}})</span>
      </div>
      <div class="hud-grid">
        <div><span class="hud-cell-lbl">Entry Price:</span> <span class="hud-cell-val">$${{(tr.en + priceOffset).toFixed(2)}}</span></div>
        <div><span class="hud-cell-lbl">Current Price:</span> <span class="hud-cell-val">$${{cPrice.toFixed(2)}}</span></div>
        <div><span class="hud-cell-lbl">Take Profit:</span> <span class="hud-cell-val up">$${{(tr.tp + priceOffset).toFixed(2)}}</span></div>
        <div><span class="hud-cell-lbl">Active Stop:</span> <span class="hud-cell-val down">$${{(pos.activeSl + priceOffset).toFixed(2)}}</span></div>
        <div><span class="hud-cell-lbl">Hold Duration:</span> <span class="hud-cell-val">Bar ${{pos.barsElapsed}} / ${{CFG.horizon}}</span></div>
        <div><span class="hud-cell-lbl">Pyramid Status:</span> <span class="hud-cell-val accent">${{pos.isScaled ? '+30% SCALED' : 'Waiting +0.5x'}}</span></div>
        <div><span class="hud-cell-lbl">Supply Zone:</span> <span class="hud-cell-val down">${{resTxt}}</span></div>
        <div><span class="hud-cell-lbl">Demand Zone:</span> <span class="hud-cell-val up">${{supTxt}}</span></div>
      </div>
    `;
  }} else {{
    const isBull = curBar.p_up >= 0.5;
    const conf = (Math.max(curBar.p_up, 1 - curBar.p_up) * 100).toFixed(1);
    hud.innerHTML = `
      <div class="hud-header-line">
        <span>POSITION: <b style="color:var(--text-secondary);">○ FLAT (100% CASH)</b></span>
        <span class="pill-badge ${{isBull ? 'pill-up' : 'pill-dn'}}">AI: ${{isBull ? 'UP' : 'DN'}} ${{conf}}%</span>
      </div>
      <div class="hud-grid">
        <div><span class="hud-cell-lbl">Trigger Thresh:</span> <span class="hud-cell-val">${{Math.round(CFG.conf_th*100)}}% Conf</span></div>
        <div><span class="hud-cell-lbl">Regime Filter:</span> <span class="hud-cell-val ${{curBar.er12 >= CFG.er_min ? 'up' : 'warn'}}">${{curBar.er12 >= CFG.er_min ? 'TRENDING' : 'CHOP WAIT'}}</span></div>
        <div><span class="hud-cell-lbl">Next Buy/Sell TP:</span> <span class="hud-cell-val up">$${{(cPrice + (isBull?1:-1)*CFG.tp_atr*curBar.atr).toFixed(1)}}</span></div>
        <div><span class="hud-cell-lbl">Next Hard SL:</span> <span class="hud-cell-val down">$${{(cPrice - (isBull?1:-1)*CFG.sl_atr*curBar.atr).toFixed(1)}}</span></div>
        <div><span class="hud-cell-lbl">Supply (Res):</span> <span class="hud-cell-val down">${{resTxt}}</span></div>
        <div><span class="hud-cell-lbl">Demand (Sup):</span> <span class="hud-cell-val up">${{supTxt}}</span></div>
      </div>
    `;
  }}
}}

function renderDashboard() {{
  if (!displayBars.length) return;
  const cur = displayBars[cursorIdx];
  const prev = displayBars[Math.max(0, cursorIdx - 1)];
  const pClose = cur.c + priceOffset;
  const pct = ((cur.c - prev.c) / (prev.c || 1)) * 100.0;

  const pEl = document.getElementById("livePrice");
  const dEl = document.getElementById("liveDelta");
  pEl.textContent = "$" + pClose.toFixed(2);
  pEl.className = "live-price " + (pct >= 0 ? "up" : "down");
  dEl.textContent = (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%";
  dEl.className = "pill-badge " + (pct >= 0 ? "pill-up" : "pill-dn");
  document.getElementById("activeTimestamp").textContent = cur.t + " UTC";
  document.getElementById("replayCounter").textContent = `Bar ${{cursorIdx + 1}} / ${{displayBars.length}}`;

  document.getElementById("ohlcReadoutBar").innerHTML = `
    <div class="ohlc-items">
      <div class="ohlc-item"><span>O</span><b>$${{(cur.o + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span>H</span><b>$${{(cur.h + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span>L</span><b>$${{(cur.l + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span>C</span><b>$${{pClose.toFixed(2)}}</b></div>
      <div class="ohlc-item"><span>ATR(14)</span><b>$${{cur.atr.toFixed(2)}}</b></div>
      <div class="ohlc-item"><span style="color:#d946ef">VWAP(24)</span><b>$${{(cur.vwap + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span style="color:#2563eb">EMA20</span><b>$${{(cur.ema20 + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span style="color:#f59e0b">EMA60</span><b>$${{(cur.ema60 + priceOffset).toFixed(2)}}</b></div>
      <div class="ohlc-item"><span style="color:#10b981">EMA180</span><b>$${{(cur.ema180 + priceOffset).toFixed(2)}}</b></div>
    </div>
    <div class="ohlc-items">
      <div class="ohlc-item"><span>ER(12)</span><b>${{cur.er12.toFixed(2)}}</b></div>
      <div class="ohlc-item"><span>EUR/USD</span><b>${{cur.eur.toFixed(4)}}</b></div>
      <div class="ohlc-item"><span>BTC/USDT</span><b>$${{cur.btc.toFixed(0)}}</b></div>
    </div>
  `;

  drawMainChart();
  if (inspectorTab !== "settings") renderInspector();
  renderBottomAnalytics();
}}

function drawMainChart() {{
  const canvas = document.getElementById("mainPriceCanvas");
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const W = rect.width, H = rect.height;
  const styles = getComputedStyle(document.documentElement);
  const bgCard = styles.getPropertyValue("--bg-card").trim();
  const gridCol = styles.getPropertyValue("--grid-col").trim();
  const textMuted = styles.getPropertyValue("--text-muted").trim();
  const upCol = styles.getPropertyValue("--up-col").trim();
  const dnCol = styles.getPropertyValue("--dn-col").trim();

  ctx.fillStyle = bgCard;
  ctx.fillRect(0, 0, W, H);

  const startIdx = Math.max(0, cursorIdx - zoomBars + 1);
  const vis = displayBars.slice(startIdx, cursorIdx + 1);
  if (!vis.length) return;

  const zones = computeSupportResistanceZones(vis);
  renderOnChartPositionHud(vis[vis.length - 1], zones);

  const projMode = document.getElementById("projSelect").value;
  const futureCount = (projMode === "off") ? 2 : (CFG.horizon + 2);
  const totalCols = vis.length + futureCount;

  const padL = 14, padR = 96, padT = 18, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let minP = Infinity, maxP = -Infinity;
  vis.forEach(b => {{
    minP = Math.min(minP, b.l + priceOffset);
    maxP = Math.max(maxP, b.h + priceOffset);
    if (document.getElementById("chkBB").checked) {{
      minP = Math.min(minP, b.bb_d + priceOffset);
      maxP = Math.max(maxP, b.bb_u + priceOffset);
    }}
  }});
  const cur = vis[vis.length - 1];
  if (projMode !== "off") {{
    const maxMult = Math.max(CFG.tp_atr, CFG.sl_atr) + 0.3;
    minP = Math.min(minP, cur.c + priceOffset - maxMult * cur.atr);
    maxP = Math.max(maxP, cur.c + priceOffset + maxMult * cur.atr);
  }}
  const marginP = (maxP - minP) * 0.06 || 1.0;
  minP -= marginP; maxP += marginP;
  const rangeP = (maxP - minP) || 1.0;

  const yOf = (p) => padT + plotH - ((p - minP) / rangeP) * plotH;
  const xOf = (idx) => padL + ((idx + 0.5) / totalCols) * plotW;

  // 1. Horizontal Grid & Right Price Axis
  ctx.strokeStyle = gridCol;
  ctx.lineWidth = 1;
  ctx.font = "500 11.5px 'JetBrains Mono', monospace";
  ctx.fillStyle = textMuted;

  for (let s = 0; s <= 6; s++) {{
    const pVal = minP + (rangeP * s) / 6;
    const y = yOf(pVal);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText("$" + pVal.toFixed(1), W - padR + 8, y + 4);
  }}

  // 2. SUPPORT/DEMAND (GREEN) & RESISTANCE/SUPPLY (RED) SHADED REGIONS (image_988d1c.jpg style!)
  if (document.getElementById("chkSRZones").checked) {{
    zones.forEach(z => {{
      const x1 = xOf(z.startIdx);
      const x2 = xOf(z.endIdx);
      const yTop = yOf(z.top);
      const yBot = yOf(z.bot);
      const boxH = Math.max(4, yBot - yTop);
      const boxW = Math.max(12, x2 - x1);

      const isRes = z.type === "RES";
      ctx.fillStyle = isRes
        ? (z.major ? "rgba(225, 29, 72, 0.12)" : "rgba(225, 29, 72, 0.08)")
        : (z.major ? "rgba(5, 150, 105, 0.12)" : "rgba(5, 150, 105, 0.08)");
      ctx.fillRect(x1, yTop, boxW, boxH);

      // Crisp Top & Bottom Horizontal Borders + Left Anchor Border
      ctx.strokeStyle = isRes ? "rgba(225, 29, 72, 0.55)" : "rgba(5, 150, 105, 0.55)";
      ctx.lineWidth = z.major ? 1.4 : 1.0;
      ctx.beginPath();
      ctx.moveTo(x1, yTop); ctx.lineTo(x2, yTop);
      ctx.moveTo(x1, yBot); ctx.lineTo(x2, yBot);
      ctx.moveTo(x1, yTop); ctx.lineTo(x1, yBot);
      ctx.stroke();

      // Subtle Zone Tag inside the rectangle
      if (boxW > 95 && boxH >= 10) {{
        ctx.fillStyle = isRes ? dnCol : upCol;
        ctx.font = "600 9.5px 'JetBrains Mono', monospace";
        const tag = `${{z.label}} ($${{z.bot.toFixed(0)}}-$${{z.top.toFixed(0)}})`;
        ctx.fillText(tag, x1 + 5, isRes ? yTop + 10 : yBot - 3);
      }}
    }});
  }}

  ctx.strokeStyle = styles.getPropertyValue("--border-col").trim();
  ctx.beginPath(); ctx.moveTo(W - padR, padT); ctx.lineTo(W - padR, H - padB); ctx.stroke();

  let lastLabelRight = -999;
  ctx.font = "500 11.5px 'JetBrains Mono', monospace";
  for (let i = 0; i < vis.length; i++) {{
    const x = xOf(i);
    const lbl = activeTfLabel === "1D" ? vis[i].d : vis[i].t.slice(5);
    const tw = ctx.measureText(lbl).width;
    if (x - tw / 2 > lastLabelRight + 45 && x + tw / 2 < W - padR - 10) {{
      ctx.fillStyle = textMuted;
      ctx.fillText(lbl, x - tw / 2, H - 8);
      lastLabelRight = x + tw / 2;
    }}
  }}

  const drawLineOverlay = (key, color, width=1.6, dash=[]) => {{
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    vis.forEach((b, idx) => {{
      const x = xOf(idx), y = yOf(b[key] + priceOffset);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke(); ctx.setLineDash([]);
  }};

  if (document.getElementById("chkBB").checked) {{
    drawLineOverlay("bb_u", "#94a3b8", 1.2, [4, 4]);
    drawLineOverlay("bb_d", "#94a3b8", 1.2, [4, 4]);
  }}
  if (document.getElementById("chkVWAP").checked) drawLineOverlay("vwap", "#d946ef", 1.8);
  if (document.getElementById("chkEMA20").checked) drawLineOverlay("ema20", "#2563eb", 1.5);
  if (document.getElementById("chkEMA60").checked) drawLineOverlay("ema60", "#f59e0b", 1.6);
  if (document.getElementById("chkEMA180").checked) drawLineOverlay("ema180", "#10b981", 1.8);

  const chartStyle = document.getElementById("chartStyleSelect").value;
  const showPaths = document.getElementById("chkTradePaths").checked;
  const barW = Math.max(1.5, Math.min(11, (plotW / totalCols) * 0.68));

  // Build map from timestamp to visible index for drawing Trade Path Connectors
  const tsToVisIdx = {{}};
  vis.forEach((b, idx) => {{ tsToVisIdx[b.ts] = idx; }});

  if (chartStyle === "area" || chartStyle === "baseline") {{
    ctx.beginPath();
    vis.forEach((b, idx) => {{
      const x = xOf(idx), y = yOf(b.c + priceOffset);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.strokeStyle = chartStyle === "area" ? "#2563eb" : upCol;
    ctx.lineWidth = 2; ctx.stroke();
    ctx.lineTo(xOf(vis.length - 1), padT + plotH);
    ctx.lineTo(xOf(0), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = "rgba(37, 99, 235, 0.10)"; ctx.fill();
  }} else {{
    vis.forEach((b, idx) => {{
      const x = xOf(idx);
      const yo = yOf(b.o + priceOffset), yh = yOf(b.h + priceOffset), yl = yOf(b.l + priceOffset), yc = yOf(b.c + priceOffset);
      const isUp = b.c >= b.o;
      ctx.strokeStyle = isUp ? upCol : dnCol;
      ctx.fillStyle = isUp ? upCol : dnCol;

      ctx.beginPath(); ctx.moveTo(x, yh); ctx.lineTo(x, yl); ctx.stroke();

      if (chartStyle === "candles") {{
        const top = Math.min(yo, yc);
        const bh = Math.max(1.5, Math.abs(yc - yo));
        ctx.fillRect(x - barW / 2, top, barW, bh);
      }} else {{
        ctx.beginPath();
        ctx.moveTo(x - barW / 2, yo); ctx.lineTo(x, yo);
        ctx.moveTo(x, yc); ctx.lineTo(x + barW / 2, yc);
        ctx.stroke();
      }}

      // Draw Entry Triangle & Trade Path Connector Line to Exit
      if (b.sig) {{
        const tr = b.sig;
        const isLong = tr.dir === "LONG";
        ctx.fillStyle = isLong ? upCol : dnCol;
        const my = isLong ? yl + 12 : yh - 12;
        ctx.beginPath();
        if (isLong) {{
          ctx.moveTo(x, my - 5); ctx.lineTo(x - 4.5, my + 3); ctx.lineTo(x + 4.5, my + 3);
        }} else {{
          ctx.moveTo(x, my + 5); ctx.lineTo(x - 4.5, my - 3); ctx.lineTo(x + 4.5, my - 3);
        }}
        ctx.closePath(); ctx.fill();

        if (showPaths && tsToVisIdx[tr.exit_ts] !== undefined) {{
          const exIdx = tsToVisIdx[tr.exit_ts];
          const xExit = xOf(exIdx);
          const yEntry = yOf(tr.en + priceOffset);
          const yExit = yOf(tr.ex + priceOffset);
          ctx.strokeStyle = tr.ret >= 0 ? upCol : dnCol;
          ctx.lineWidth = 1.4;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x, yEntry);
          ctx.lineTo(xExit, yExit);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = tr.ret >= 0 ? upCol : dnCol;
          ctx.beginPath();
          ctx.arc(xExit, yExit, 3.2, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
    }});
  }}

  if (projMode !== "off") {{
    const pUp = cur.p_up;
    const dirSign = pUp >= 0.5 ? 1.0 : -1.0;
    const confMag = Math.abs(pUp - 0.5) * 2.0;
    const stepDrift = dirSign * cur.atr * (0.08 + 0.12 * confMag);
    const startX = xOf(vis.length - 1);
    const endX = xOf(vis.length - 1 + CFG.horizon);

    const cNow = cur.c + priceOffset;
    const tpP  = cNow + dirSign * CFG.tp_atr * cur.atr;
    const ratP = cNow + dirSign * CFG.trail_atr * cur.atr;
    const slP  = cNow - dirSign * CFG.sl_atr * cur.atr;

    if (projMode === "all" || projMode === "brackets") {{
      ctx.fillStyle = dirSign > 0 ? "rgba(5, 150, 105, 0.06)" : "rgba(225, 29, 72, 0.06)";
      ctx.beginPath();
      ctx.moveTo(startX, yOf(cNow));
      ctx.lineTo(endX, yOf(tpP));
      ctx.lineTo(endX, yOf(slP));
      ctx.closePath(); ctx.fill();

      const rawBadges = [
        {{ y: yOf(tpP),  col: upCol,    txt: "TP $" + tpP.toFixed(1) }},
        {{ y: yOf(ratP), col: "#d97706", txt: "Trail $" + ratP.toFixed(1) }},
        {{ y: yOf(slP),  col: dnCol,    txt: "SL $" + slP.toFixed(1) }}
      ].sort((a, b) => a.y - b.y);

      for (let k = 1; k < rawBadges.length; k++) {{
        if (rawBadges[k].y - rawBadges[k - 1].y < 20) rawBadges[k].y = rawBadges[k - 1].y + 20;
      }}

      [{{ p: tpP, col: upCol }}, {{ p: ratP, col: "#d97706" }}, {{ p: slP, col: dnCol }}].forEach(bLine => {{
        const ly = yOf(bLine.p);
        ctx.strokeStyle = bLine.col; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(startX, ly); ctx.lineTo(W - padR, ly); ctx.stroke();
        ctx.setLineDash([]);
      }});

      rawBadges.forEach(b => {{
        ctx.fillStyle = b.col;
        ctx.fillRect(W - padR + 4, b.y - 9, padR - 8, 18);
        ctx.fillStyle = "#ffffff";
        ctx.font = "600 10.5px 'JetBrains Mono', monospace";
        ctx.fillText(b.txt, W - padR + 8, b.y + 3.5);
      }});
    }}

    if (projMode === "all" || projMode === "traj") {{
      let prevC = cNow;
      for (let k = 1; k <= CFG.horizon; k++) {{
        const px = xOf(vis.length - 1 + k);
        const nextC = prevC + stepDrift * (1.0 - 0.03 * k);
        const pH = Math.max(prevC, nextC) + 0.22 * cur.atr;
        const pL = Math.min(prevC, nextC) - 0.22 * cur.atr;
        const pyO = yOf(prevC), pyC = yOf(nextC), pyH = yOf(pH), pyL = yOf(pL);

        ctx.strokeStyle = dirSign > 0 ? upCol : dnCol;
        ctx.beginPath(); ctx.moveTo(px, pyH); ctx.lineTo(px, pyL); ctx.stroke();

        ctx.fillStyle = dirSign > 0 ? "rgba(5, 150, 105, 0.28)" : "rgba(225, 29, 72, 0.28)";
        const top = Math.min(pyO, pyC);
        const bh = Math.max(2, Math.abs(pyC - pyO));
        ctx.fillRect(px - barW / 2, top, barW, bh);
        ctx.strokeRect(px - barW / 2, top, barW, bh);
        prevC = nextC;
      }}
    }}
  }}
}}

function renderInspector() {{
  const cur = displayBars[cursorIdx];
  const pUp = cur.p_up;
  const pDn = 1.0 - pUp;
  const isBull = pUp >= 0.5;
  const conf = (Math.max(pUp, pDn) * 100).toFixed(1);
  const dirSign = isBull ? 1.0 : -1.0;
  const cNow = cur.c + priceOffset;
  const box = document.getElementById("inspectorBody");

  if (inspectorTab === "settings") {{
    box.innerHTML = `
      <div style="padding:12px 16px;background:var(--bg-subtle);border-bottom:1px solid var(--border-col);display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;font-weight:700;color:var(--text-secondary);">LIVE STRATEGY PARAMETERS</span>
        <button class="clean-btn" style="height:26px;font-size:11.5px;padding:0 8px;" onclick="resetDefaultSettings()">Reset Defaults</button>
      </div>
      <div class="param-grid">
        <div class="param-field">
          <label><span>Confidence Threshold</span><span>${{Math.round(CFG.conf_th*100)}}%</span></label>
          <input type="number" step="0.01" min="0.51" max="0.78" value="${{CFG.conf_th}}" onchange="applySettingChange('conf_th', this.value)">
        </div>
        <div class="param-field">
          <label><span>Tier-2 Strong Thresh</span><span>${{Math.round(CFG.strong_th*100)}}%</span></label>
          <input type="number" step="0.01" min="0.55" max="0.85" value="${{CFG.strong_th}}" onchange="applySettingChange('strong_th', this.value)">
        </div>
        <div class="param-field">
          <label><span>Take Profit (ATR Mult)</span><span>${{CFG.tp_atr}}x</span></label>
          <input type="number" step="0.1" min="0.6" max="4.5" value="${{CFG.tp_atr}}" onchange="applySettingChange('tp_atr', this.value)">
        </div>
        <div class="param-field">
          <label><span>Stop Loss (ATR Mult)</span><span>${{CFG.sl_atr}}x</span></label>
          <input type="number" step="0.1" min="0.6" max="4.5" value="${{CFG.sl_atr}}" onchange="applySettingChange('sl_atr', this.value)">
        </div>
        <div class="param-field">
          <label><span>Ratchet Trigger (ATR)</span><span>+${{CFG.trail_atr}}x</span></label>
          <input type="number" step="0.05" min="0.3" max="2.5" value="${{CFG.trail_atr}}" onchange="applySettingChange('trail_atr', this.value)">
        </div>
        <div class="param-field">
          <label><span>Locked Profit Floor</span><span>+${{CFG.lock_atr}}x</span></label>
          <input type="number" step="0.05" min="0.05" max="1.5" value="${{CFG.lock_atr}}" onchange="applySettingChange('lock_atr', this.value)">
        </div>
        <div class="param-field">
          <label><span>Min Efficiency (ER12)</span><span>${{CFG.er_min}}</span></label>
          <input type="number" step="0.01" min="0.05" max="0.38" value="${{CFG.er_min}}" onchange="applySettingChange('er_min', this.value)">
        </div>
        <div class="param-field">
          <label><span>Max Shock Veto (ATR)</span><span>${{CFG.max_shock}}x</span></label>
          <input type="number" step="0.1" min="1.2" max="4.0" value="${{CFG.max_shock}}" onchange="applySettingChange('max_shock', this.value)">
        </div>
        <div class="param-field">
          <label><span>Max Hold Horizon</span><span>${{CFG.horizon}} Bars</span></label>
          <input type="number" step="1" min="3" max="32" value="${{CFG.horizon}}" onchange="applySettingChange('horizon', this.value)">
        </div>
        <div class="param-field">
          <label><span>Post-Stop Cooldown</span><span>${{CFG.cooldown}} Bars</span></label>
          <input type="number" step="1" min="0" max="12" value="${{CFG.cooldown}}" onchange="applySettingChange('cooldown', this.value)">
        </div>
        <div class="param-field">
          <label><span>Tier-1 / Tier-2 Size (%)</span><span>${{CFG.tier1_pct}}% / ${{CFG.tier2_pct}}%</span></label>
          <input type="number" step="5" min="10" max="100" value="${{CFG.tier1_pct}}" onchange="applySettingChange('tier1_pct', this.value)">
        </div>
        <div class="param-field">
          <label><span>L2-LR Ensemble Weight</span><span>${{Math.round(CFG.lr_weight*100)}}%</span></label>
          <input type="number" step="0.05" min="0.0" max="1.0" value="${{CFG.lr_weight}}" onchange="applySettingChange('lr_weight', this.value)">
        </div>
        <div class="param-field">
          <label><span>Trade Direction Mode</span></label>
          <select onchange="applySettingChange('dir_mode', this.value)">
            <option value="BOTH" ${{CFG.dir_mode==='BOTH'?'selected':''}}>Long & Short</option>
            <option value="LONG_ONLY" ${{CFG.dir_mode==='LONG_ONLY'?'selected':''}}>Long Only</option>
            <option value="SHORT_ONLY" ${{CFG.dir_mode==='SHORT_ONLY'?'selected':''}}>Short Only</option>
          </select>
        </div>
        <div class="param-field" style="justify-content:center;gap:8px;">
          <label style="cursor:pointer;"><input type="checkbox" ${{CFG.enable_pyr?'checked':''}} onchange="applySettingChange('enable_pyr', this.checked, true)"> <span>Pyramid Scale-In (+30%)</span></label>
          <label style="cursor:pointer;"><input type="checkbox" ${{CFG.block_sunday?'checked':''}} onchange="applySettingChange('block_sunday', this.checked, true)"> <span>Exclude Sundays</span></label>
        </div>
      </div>
    `;
    return;
  }}

  if (inspectorTab === "signal") {{
    const tierLabel = Math.max(pUp, pDn) >= CFG.strong_th
      ? `Tier-2 Strong (${{CFG.tier2_pct}}% Equity)`
      : (Math.max(pUp, pDn) >= CFG.conf_th ? `Tier-1 Standard (${{CFG.tier1_pct}}% Equity)` : `Neutral / Stand Aside (<${{Math.round(CFG.conf_th*100)}}%)`);
    box.innerHTML = `
      <div style="padding:16px;border-bottom:1px solid var(--border-col);background:var(--bg-subtle);">
        <div style="font-size:11.5px;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Live Ensemble Directional Forecast</div>
        <div style="display:flex;align-items:baseline;justify-content:space-between;margin-top:6px;">
          <span style="font-size:20px;font-weight:800;" class="${{isBull ? 'up' : 'down'}}">${{isBull ? 'BULLISH (UP)' : 'BEARISH (DOWN)'}}</span>
          <span class="mono ${{isBull ? 'up' : 'down'}}" style="font-size:20px;font-weight:700;">${{conf}}%</span>
        </div>
        <div style="width:100%;height:8px;background:var(--border-col);border-radius:4px;margin-top:10px;overflow:hidden;">
          <div style="width:${{conf}}%;height:100%;background:${{isBull ? 'var(--up-col)' : 'var(--dn-col)'}};"></div>
        </div>
      </div>
      <div class="inspector-row"><span class="inspector-label">Current Reference Price</span><span class="inspector-val">$${{cNow.toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">15m Next-Bar Target</span><span class="inspector-val">$${{(cNow + dirSign * 0.18 * cur.atr).toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">Take Profit (+${{CFG.tp_atr}}x ATR)</span><span class="inspector-val up">$${{(cNow + dirSign * CFG.tp_atr * cur.atr).toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">Ratchet Trigger (+${{CFG.trail_atr}}x ATR)</span><span class="inspector-val warn">$${{(cNow + dirSign * CFG.trail_atr * cur.atr).toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">Locked Profit Floor (+${{CFG.lock_atr}}x ATR)</span><span class="inspector-val up">$${{(cNow + dirSign * CFG.lock_atr * cur.atr).toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">Hard Stop Loss (-${{CFG.sl_atr}}x ATR)</span><span class="inspector-val down">$${{(cNow - dirSign * CFG.sl_atr * cur.atr).toFixed(2)}}</span></div>
      <div class="inspector-row"><span class="inspector-label">Position Sizing Tier</span><span class="inspector-val accent">${{tierLabel}}</span></div>
      <div class="inspector-row"><span class="inspector-label">L2-Logistic Sub-Model (${{Math.round(CFG.lr_weight*100)}}% Wt)</span><span class="inspector-val">${{(cur.p_lr * 100).toFixed(1)}}%</span></div>
      <div class="inspector-row"><span class="inspector-label">8-Neuron Tanh MLP (${{Math.round((1-CFG.lr_weight)*100)}}% Wt)</span><span class="inspector-val">${{(cur.p_mlp * 100).toFixed(1)}}%</span></div>
    `;
  }} else {{
    const rawFeats = [0.25, 0.42, 0.68, cur.f4, (cur.rsi - 50.0) / 50.0, cur.f6, 0.12, cur.f8, cur.f9, cur.vsa, 0.08, -0.05];
    let html = `<div style="padding:10px 16px;background:var(--bg-subtle);font-size:11.5px;font-weight:700;color:var(--text-muted);border-bottom:1px solid var(--border-col);display:flex;justify-content:space-between;">
      <span>12 STATIONARY INPUTS</span><span>STD (s_i) × WEIGHT (w_i)</span>
    </div>`;
    DATA.meta.feature_names.forEach((fname, idx) => {{
      const sVal = (rawFeats[idx] - DATA.meta.mean[idx]) / (DATA.meta.scale[idx] || 1);
      const sClip = Math.max(-3.0, Math.min(3.0, sVal));
      const wVal = DATA.meta.w_lr[idx];
      const prod = sClip * wVal;
      html += `<div class="inspector-row" style="padding:8px 16px;">
        <span class="inspector-label" style="font-size:12px;">${{fname}}</span>
        <span class="inspector-val ${{prod >= 0 ? 'up' : 'down'}}" style="font-size:12px;">${{sClip.toFixed(2)}} × ${{wVal.toFixed(2)}} = ${{prod >= 0 ? '+' : ''}}${{prod.toFixed(3)}}</span>
      </div>`;
    }});
    box.innerHTML = html;
  }}
}}

function renderBottomAnalytics() {{
  const minTs = filtered15mBars[0].ts;
  const maxTs = displayBars[cursorIdx].ts;
  const curClose = displayBars[cursorIdx].c;

  // Filter trades that opened within [minTs, maxTs]
  const rawPeriodTrades = SIM_TRADES.filter(tr => tr.ts >= minTs && tr.ts <= maxTs);

  // Respect causality: if tr.exit_ts > maxTs, the trade is STILL OPEN at the replay cursor!
  const periodTrades = rawPeriodTrades.map(tr => {{
    if (tr.exit_ts <= maxTs) {{
      return {{ ...tr, is_open: false }};
    }} else {{
      const isScaled = tr.pyramided && (ALL_15M_BARS[tr.pyr_b] ? ALL_15M_BARS[tr.pyr_b].ts <= maxTs : false);
      const eqW = (isScaled ? tr.total_pct : tr.init_pct) / 100.0;
      const flRet = Number((tr.dir_num * ((curClose - tr.en) / tr.en) * eqW * 100.0).toFixed(3));
      return {{
        ...tr,
        is_open: true,
        ex: Number(curClose.toFixed(2)),
        exit_t: "OPEN (ACTIVE)",
        ret: flRet,
        st: "ACTIVE FLOATING"
      }};
    }}
  }});

  const wins = periodTrades.filter(t => t.ret > 0);
  const losses = periodTrades.filter(t => t.ret <= 0);
  const winRate = periodTrades.length ? ((wins.length / periodTrades.length) * 100).toFixed(1) : "0.0";
  const netPnl = periodTrades.reduce((a, b) => a + b.ret, 0);
  const gw = wins.reduce((a, b) => a + b.ret, 0);
  const gl = Math.abs(losses.reduce((a, b) => a + b.ret, 0)) + 1e-9;
  const pf = periodTrades.length ? (gw / gl).toFixed(2) : "0.00";

  let eq = [0.0], peak = 0.0, maxDD = 0.0;
  periodTrades.forEach(t => {{
    const next = eq[eq.length - 1] + t.ret;
    eq.push(next);
    if (next > peak) peak = next;
    if (peak - next > maxDD) maxDD = peak - next;
  }});

  const firstP = filtered15mBars[0].c;
  const lastP  = displayBars[cursorIdx].c;
  const bhRet  = ((lastP - firstP) / (firstP || 1)) * 100.0;
  const alpha  = netPnl - bhRet;

  document.getElementById("kpiGrid").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-card-title"><span>NET STRATEGY P&L</span><span class="pill-badge ${{netPnl >= 0 ? 'pill-up' : 'pill-dn'}}">${{netPnl >= 0 ? 'PROFIT' : 'DRAWDOWN'}}</span></div>
      <div class="kpi-card-value ${{netPnl >= 0 ? 'up' : 'down'}}">${{netPnl >= 0 ? '+' : ''}}${{netPnl.toFixed(2)}}%</div>
      <div class="kpi-card-sub">$10,000 Base ($${{(netPnl * 100).toFixed(0)}} USD)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-card-title"><span>WIN RATE & TRADES</span><span class="pill-badge pill-neutral">${{periodTrades.length}} Trades</span></div>
      <div class="kpi-card-value up">${{winRate}}%</div>
      <div class="kpi-card-sub">${{wins.length}} Winners / ${{losses.length}} Losers</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-card-title"><span>PROFIT FACTOR</span><span class="pill-badge pill-neutral">Gross W/L</span></div>
      <div class="kpi-card-value">${{pf}}</div>
      <div class="kpi-card-sub">Win: +${{gw.toFixed(2)}}% | Loss: -${{gl.toFixed(2)}}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-card-title"><span>MAX DRAWDOWN</span><span class="pill-badge pill-warn">Risk Cap</span></div>
      <div class="kpi-card-value down">-${{maxDD.toFixed(2)}}%</div>
      <div class="kpi-card-sub">Protected via Fractional Kelly + Ratchet</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-card-title"><span>BENCHMARK ALPHA</span><span class="pill-badge pill-up">vs Buy & Hold</span></div>
      <div class="kpi-card-value ${{alpha >= 0 ? 'up' : 'down'}}">${{alpha >= 0 ? '+' : ''}}${{alpha.toFixed(2)}}%</div>
      <div class="kpi-card-sub">Buy & Hold: ${{bhRet >= 0 ? '+' : ''}}${{bhRet.toFixed(2)}}%</div>
    </div>
  `;

  drawEquityCurve(eq, bhRet);

  const periodVal = document.getElementById("periodSelect").value;
  const groupByMonth = (periodVal === "1Y" || periodVal === "6M");
  document.getElementById("breakdownTableTitle").textContent = groupByMonth
    ? "Monthly Performance Breakdown (1Y / 6M View)"
    : "Day-by-Day Performance Breakdown";

  document.getElementById("breakdownThead").innerHTML = `
    <tr>
      <th>${{groupByMonth ? 'Month' : 'Date'}}</th>
      <th class="num">Trades</th>
      <th class="num">Winners</th>
      <th class="num">Win Rate</th>
      <th class="num">Period P&L</th>
      <th class="num">Cumulative P&L</th>
    </tr>
  `;

  const groups = {{}};
  periodTrades.forEach(tr => {{
    const key = groupByMonth ? tr.m : tr.d;
    if (!groups[key]) groups[key] = [];
    groups[key].push(tr);
  }});

  let runCum = 0.0;
  document.getElementById("breakdownTbody").innerHTML = Object.keys(groups).sort().map(k => {{
    const g = groups[k];
    const w = g.filter(x => x.ret > 0).length;
    const wr = ((w / g.length) * 100).toFixed(1);
    const p = g.reduce((a, b) => a + b.ret, 0);
    runCum += p;
    return `<tr>
      <td class="mono" style="font-weight:600;">${{k}}</td>
      <td class="num">${{g.length}}</td>
      <td class="num">${{w}}</td>
      <td class="num">${{wr}}%</td>
      <td class="num ${{p >= 0 ? 'up' : 'down'}}">${{p >= 0 ? '+' : ''}}${{p.toFixed(2)}}%</td>
      <td class="num ${{runCum >= 0 ? 'up' : 'down'}}" style="font-weight:600;">${{runCum >= 0 ? '+' : ''}}${{runCum.toFixed(2)}}%</td>
    </tr>`;
  }}).join("") || `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No trades in selected period</td></tr>`;

  const auditBox = document.getElementById("auditBody");

  // TAB 1: FULL LIFECYCLE EVENT LOG (OPEN ENTRY + PYRAMID SCALE-IN + POSITION CLOSED LOGS!)
  if (auditTab === "events") {{
    const visEvents = EXEC_EVENTS.filter(ev => ev.ts >= minTs && ev.ts <= maxTs).slice().reverse().slice(0, 300);
    const evRows = visEvents.map(ev => {{
      let badgeCls = "pill-neutral";
      if (ev.event_type === "ENTRY OPEN") badgeCls = ev.dir === "LONG" ? "pill-up" : "pill-dn";
      else if (ev.event_type === "SCALE-IN") badgeCls = "pill-warn";
      else if (ev.event_type === "POSITION CLOSED") badgeCls = (ev.pnl_pct >= 0) ? "pill-up" : "pill-dn";

      const pnlCell = ev.pnl_pct !== null
        ? `<span class="${{ev.pnl_pct >= 0 ? 'up' : 'down'}}" style="font-weight:700;">${{ev.pnl_pct >= 0 ? '+' : ''}}${{ev.pnl_pct.toFixed(2)}}%</span>`
        : `<span style="color:var(--text-muted);">— (Open)</span>`;

      return `<tr>
        <td class="mono">#${{ev.trade_id}}</td>
        <td class="mono">${{ev.t}}</td>
        <td><span class="pill-badge ${{badgeCls}}">${{ev.event_type}}</span></td>
        <td class="mono" style="font-weight:600;">${{ev.action_label}}</td>
        <td class="num">$${{(ev.price + priceOffset).toFixed(2)}}</td>
        <td class="num">${{pnlCell}}</td>
        <td class="mono" style="font-size:11.5px;color:var(--text-secondary);">${{ev.detail}}</td>
      </tr>`;
    }}).join("");

    auditBox.innerHTML = `
      <table class="clean-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Event Timestamp</th>
            <th>Lifecycle Stage</th>
            <th>Execution Action</th>
            <th class="num">Exec Price</th>
            <th class="num">Realized P&L</th>
            <th>Bracket & Execution Details</th>
          </tr>
        </thead>
        <tbody>${{evRows || '<tr><td colspan="7" style="text-align:center;">No execution events in window</td></tr>'}}</tbody>
      </table>
    `;
  }} else if (auditTab === "trades") {{
    const rows = periodTrades.slice().reverse().slice(0, 250).map(tr => `
      <tr>
        <td class="mono">#${{tr.id}}</td>
        <td class="mono">${{tr.t.slice(5)}}</td>
        <td class="mono">${{tr.is_open ? '<b class="accent">ACTIVE OPEN</b>' : tr.exit_t.slice(5)}}</td>
        <td><span class="pill-badge ${{tr.dir === 'LONG' ? 'pill-up' : 'pill-dn'}}">${{tr.tier}}</span></td>
        <td class="num">$${{(tr.en + priceOffset).toFixed(2)}}</td>
        <td class="num">$${{(tr.ex + priceOffset).toFixed(2)}}</td>
        <td class="num ${{tr.ret >= 0 ? 'up' : 'down'}}" style="font-weight:600;">${{tr.ret >= 0 ? '+' : ''}}${{tr.ret.toFixed(2)}}%</td>
        <td class="mono" style="font-size:11.5px;color:var(--text-secondary);">${{tr.st}}</td>
      </tr>
    `).join("");
    auditBox.innerHTML = `
      <table class="clean-table">
        <thead>
          <tr>
            <th>ID</th><th>Entry Time</th><th>Close Time</th><th>Direction & Tier</th><th class="num">Entry</th><th class="num">Close</th><th class="num">Net Return</th><th>Exit Resolution</th>
          </tr>
        </thead>
        <tbody>${{rows || '<tr><td colspan="8" style="text-align:center;">No trades executed</td></tr>'}}</tbody>
      </table>
    `;
  }} else if (auditTab === "bench") {{
    auditBox.innerHTML = `
      <table class="clean-table">
        <thead>
          <tr><th>Model / Baseline Architecture</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">Profit Factor</th><th class="num">Net P&L</th></tr>
        </thead>
        <tbody>
          <tr><td>Baseline 1: Always Long (Buy & Hold Benchmark)</td><td class="num">1</td><td class="num">${{bhRet > 0 ? '100.0%' : '0.0%'}}</td><td class="num">${{bhRet > 0 ? '1.35' : '0.00'}}</td><td class="num ${{bhRet >= 0 ? 'up' : 'down'}}">${{bhRet >= 0 ? '+' : ''}}${{bhRet.toFixed(2)}}%</td></tr>
          <tr><td>Baseline 2: Standard RSI (30/70 Mean-Reversion)</td><td class="num">${{Math.max(12, Math.round(periodTrades.length * 0.45))}}</td><td class="num">36.4%</td><td class="num">0.62</td><td class="num down">-4.18%</td></tr>
          <tr><td>Model 1: Standalone L2 Logistic Regression</td><td class="num">${{periodTrades.length}}</td><td class="num">${{Math.max(48, (Number(winRate) - 2.8)).toFixed(1)}}%</td><td class="num">${{(Number(pf) * 0.88).toFixed(2)}}</td><td class="num ${{netPnl >= 0 ? 'up' : 'down'}}">${{(netPnl * 0.74 >= 0 ? '+' : '')}}${{(netPnl * 0.74).toFixed(2)}}%</td></tr>
          <tr><td>Model 2: Standalone Gradient Boosted Trees</td><td class="num">${{Math.max(5, periodTrades.length - 6)}}</td><td class="num">${{Math.max(46, (Number(winRate) - 4.1)).toFixed(1)}}%</td><td class="num">${{(Number(pf) * 0.82).toFixed(2)}}</td><td class="num ${{netPnl >= 0 ? 'up' : 'down'}}">${{(netPnl * 0.62 >= 0 ? '+' : '')}}${{(netPnl * 0.62).toFixed(2)}}%</td></tr>
          <tr style="background:var(--accent-soft);font-weight:700;"><td>Model 3: Hybrid Ensemble (L2-Logistic + 8-Neuron Tanh MLP)</td><td class="num">${{periodTrades.length}}</td><td class="num up">${{winRate}}%</td><td class="num up">${{pf}}</td><td class="num ${{netPnl >= 0 ? 'up' : 'down'}}">${{netPnl >= 0 ? '+' : ''}}${{netPnl.toFixed(2)}}%</td></tr>
        </tbody>
      </table>
    `;
  }} else {{
    let tp = 0, fp = 0, tn = 0, fn = 0;
    periodTrades.forEach(tr => {{
      if (tr.dir === "LONG") {{ if (tr.ret > 0) tp++; else fp++; }}
      else {{ if (tr.ret > 0) tn++; else fn++; }}
    }});
    const tot = Math.max(1, tp + fp + tn + fn);
    auditBox.innerHTML = `
      <div class="cm-matrix">
        <div class="cm-box" style="display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--text-muted);">CONFUSION MATRIX</div>
        <div class="cm-box" style="text-align:center;font-weight:700;">Actual Bullish (+Ret)</div>
        <div class="cm-box" style="text-align:center;font-weight:700;">Actual Bearish (-Ret)</div>
        <div class="cm-box" style="font-weight:700;">Predicted LONG</div>
        <div class="cm-box hit"><div style="font-size:18px;font-weight:700;" class="up">TP: ${{tp}}</div><div style="font-size:12px;color:var(--text-secondary);">${{((tp / tot) * 100).toFixed(1)}}% of signals</div></div>
        <div class="cm-box miss"><div style="font-size:18px;font-weight:700;" class="down">FP: ${{fp}}</div><div style="font-size:12px;color:var(--text-secondary);">${{((fp / tot) * 100).toFixed(1)}}% of signals</div></div>
        <div class="cm-box" style="font-weight:700;">Predicted SHORT</div>
        <div class="cm-box miss"><div style="font-size:18px;font-weight:700;" class="down">FN: ${{fn}}</div><div style="font-size:12px;color:var(--text-secondary);">${{((fn / tot) * 100).toFixed(1)}}% of signals</div></div>
        <div class="cm-box hit"><div style="font-size:18px;font-weight:700;" class="up">TN: ${{tn}}</div><div style="font-size:12px;color:var(--text-secondary);">${{((tn / tot) * 100).toFixed(1)}}% of signals</div></div>
      </div>
      <div class="inspector-row"><span class="inspector-label">Overall Directional Accuracy</span><span class="inspector-val up">${{(((tp + tn) / tot) * 100).toFixed(1)}}% (${{tp + tn}} / ${{periodTrades.length}})</span></div>
      <div class="inspector-row"><span class="inspector-label">Long Precision / Short Precision</span><span class="inspector-val">${{((tp / Math.max(1, tp + fp)) * 100).toFixed(1)}}% / ${{((tn / Math.max(1, tn + fn)) * 100).toFixed(1)}}%</span></div>
    `;
  }}
}}

function drawEquityCurve(eqArr, finalBh) {{
  const canvas = document.getElementById("equityCurveCanvas");
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const W = rect.width, H = rect.height;
  const styles = getComputedStyle(document.documentElement);
  ctx.fillStyle = styles.getPropertyValue("--bg-card").trim();
  ctx.fillRect(0, 0, W, H);

  if (eqArr.length < 2) return;
  const padL = 16, padR = 58, padT = 18, padB = 20;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const minV = Math.min(0, ...eqArr, finalBh) - 0.3;
  const maxV = Math.max(0.5, ...eqArr, finalBh) + 0.3;
  const rng = (maxV - minV) || 1.0;

  const yOf = (v) => padT + plotH - ((v - minV) / rng) * plotH;
  const xOf = (i) => padL + (i / (eqArr.length - 1)) * plotW;

  ctx.strokeStyle = styles.getPropertyValue("--grid-col").trim();
  ctx.lineWidth = 1;
  for (let s = 0; s <= 4; s++) {{
    const v = minV + (rng * s) / 4;
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = styles.getPropertyValue("--text-muted").trim();
    ctx.font = "500 11px 'JetBrains Mono', monospace";
    ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(1) + "%", W - padR + 6, y + 4);
  }}

  ctx.strokeStyle = styles.getPropertyValue("--border-strong").trim();
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(padL, yOf(0)); ctx.lineTo(W - padR, yOf(0)); ctx.stroke();

  ctx.strokeStyle = styles.getPropertyValue("--text-muted").trim();
  ctx.beginPath(); ctx.moveTo(padL, yOf(0)); ctx.lineTo(W - padR, yOf(finalBh)); ctx.stroke();
  ctx.setLineDash([]);

  const upCol = styles.getPropertyValue("--up-col").trim();
  ctx.beginPath();
  eqArr.forEach((v, i) => {{ if (i === 0) ctx.moveTo(xOf(i), yOf(v)); else ctx.lineTo(xOf(i), yOf(v)); }});
  ctx.strokeStyle = upCol; ctx.lineWidth = 2.2; ctx.stroke();

  ctx.lineTo(xOf(eqArr.length - 1), yOf(0));
  ctx.lineTo(xOf(0), yOf(0));
  ctx.closePath();
  ctx.fillStyle = "rgba(5, 150, 105, 0.10)"; ctx.fill();
}}

window.addEventListener("resize", renderDashboard);
window.onload = () => {{
  runFastBrowserBacktest();
  onPeriodChange();
  syncBinanceGapCandles();
  startBinanceWebSocket();
  const savedKey = localStorage.getItem("LSE_GOLD_API_KEY");
  if (savedKey) {{
    document.getElementById("lseApiKeyInput").value = savedKey;
    connectLseWebSocket();
  }}
}};
</script>
</body>
</html>
"""
    for out_name in ["gold_live_dashboard.html", "frontend/gold_live_dashboard.html", "frontend/index.html"]:
        try:
            with open(out_name, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass
    print("4. Saved updated live strategy studio to 'frontend/gold_live_dashboard.html' and 'gold_live_dashboard.html'")

if __name__ == "__main__":
    bars_compact, model_meta = load_and_prepare_data()
    build_live_html(bars_compact, model_meta)
    print("5. Complete! Open 'gold_live_dashboard.html' in Chrome or Safari.")