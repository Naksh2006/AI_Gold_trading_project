import numpy as np
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
