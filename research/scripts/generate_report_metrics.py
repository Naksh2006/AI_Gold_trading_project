import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def main():
    symbols = ["paxgusdt", "eurusdt", "btcusdt"]
    dfs = {}
    for sym in symbols:
        for days in [1460, 730, 365]:
            fname = f"{sym}_15m_{days}d.csv"
            if os.path.exists(fname):
                d = pd.read_csv(fname)
                d["datetime"] = pd.to_datetime(d["datetime"])
                dfs[sym] = d
                break

    df = dfs["paxgusdt"].merge(
        dfs["eurusdt"][["datetime", "close"]].rename(columns={"close": "eur_close"}), on="datetime", how="inner"
    ).merge(
        dfs["btcusdt"][["datetime", "close"]].rename(columns={"close": "btc_close"}), on="datetime", how="inner"
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
    df["f4"] = ((c - vwap24) / c) / atr_pct

    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
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
    df["f8"] = ((ema20 - ema60) / c) / atr_pct
    df["f9"] = ((ema60 - ema180) / c) / atr_pct

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
    feature_names = [
        "f1 (30m Return / ATR)", "f2 (1h Return / ATR)", "f3 (3h Return / ATR)",
        "f4 (24-Bar VWAP Deviation)", "f5 (Wilder RSI-14 Centered)", "f6 (Normalized MACD Hist)",
        "f7 (16-Bar Stochastic Pos)", "f8 (EMA 20-60 Spread / ATR)", "f9 (EMA 60-180 Macro / ATR)",
        "f10 (Order Flow VSA 4-Bar)", "f11 (EUR/USD 1h Momentum)", "f12 (Cross-Asset Divergence)"
    ]

    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df["datetime"] <= TRAIN_END].iloc[-35000:].copy()
    train_clean = train_df[(train_df["target"] != -1) & (train_df["shock_ratio"] <= 2.3)].copy()
    pos_df = train_clean[train_clean["target"] == 1]
    neg_df = train_clean[train_clean["target"] == 0]
    m_len = min(len(pos_df), len(neg_df))
    bal_train = pd.concat([pos_df.iloc[-m_len:], neg_df.iloc[-m_len:]]).sort_values("datetime")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(bal_train[feature_cols])
    y_train = bal_train["target"].values

    lr  = LogisticRegression(C=0.10, random_state=42).fit(X_train_s, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", alpha=0.25, max_iter=500, random_state=42).fit(X_train_s, y_train)
    gb  = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.10, random_state=42).fit(X_train_s, y_train)

    oos_clean = df[(df["datetime"] > TRAIN_END) & (df["target"] != -1)].copy()
    X_oos_s = scaler.transform(oos_clean[feature_cols])
    y_oos = oos_clean["target"].values

    print("\n================ COPY THIS VERIFIED ML METRICS BLOCK INTO CLAUDE ================")
    print("\n### Verified Out-of-Sample ML Classification Metrics (Clean Directional Moves > Aug 31, 2026)")
    print("| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC | Brier Score | Log-Loss |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, probs in [
        ("L2 Logistic Regression (C=0.10)", lr.predict_proba(X_oos_s)[:, 1]),
        ("Gradient Boosted Decision Trees", gb.predict_proba(X_oos_s)[:, 1]),
        ("8-Neuron Tanh MLP (alpha=0.25)", mlp.predict_proba(X_oos_s)[:, 1]),
        ("Hybrid Ensemble (0.55 LR + 0.45 MLP)", 0.55 * lr.predict_proba(X_oos_s)[:, 1] + 0.45 * mlp.predict_proba(X_oos_s)[:, 1])
    ]:
        preds = (probs >= np.median(probs)).astype(int)
        acc = accuracy_score(y_oos, preds) * 100
        prec = precision_score(y_oos, preds, zero_division=0) * 100
        rec = recall_score(y_oos, preds, zero_division=0) * 100
        f1 = f1_score(y_oos, preds, zero_division=0)
        auc = roc_auc_score(y_oos, probs)
        brier = brier_score_loss(y_oos, probs)
        ll = log_loss(y_oos, probs)
        print(f"| {name} | {acc:.2f}% | {prec:.2f}% | {rec:.2f}% | {f1:.4f} | {auc:.4f} | {brier:.4f} | {ll:.4f} |")

    print("\n### Verified Standardized Feature Parameters & Learned Weights")
    print("| Feature ID & Description | Scaler Mean (mu) | Scaler Std (sigma) | L2-LR Weight (w_i) | MLP L1 Mean Abs Weight | Importance Rank |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    w_lr = lr.coef_[0]
    w_mlp_abs = np.mean(np.abs(mlp.coefs_[0]), axis=1)
    combined_imp = 0.55 * np.abs(w_lr) + 0.45 * w_mlp_abs
    ranks = np.argsort(np.argsort(-combined_imp)) + 1

    for idx in range(12):
        print(f"| {feature_names[idx]} | {scaler.mean_[idx]:+.6f} | {scaler.scale_[idx]:.6f} | {w_lr[idx]:+.6f} | {w_mlp_abs[idx]:.6f} | #{ranks[idx]} |")
    print("=================================================================================\n")

if __name__ == "__main__":
    main()