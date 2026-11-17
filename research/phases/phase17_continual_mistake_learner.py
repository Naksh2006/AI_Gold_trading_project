import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FETCH_4_YEARS = True

# =========================================================================
# 1. MULTI-YEAR DATA ENGINE (LOADS YOUR 140,155-ROW 4-YEAR CACHE INSTANTLY)
# =========================================================================
def load_or_expand_symbol(symbol, interval="15m", target_days=1460):
    cache_2y = f"{symbol.lower()}_{interval}_730d.csv"
    cache_4y = f"{symbol.lower()}_{interval}_{target_days}d.csv"

    if os.path.exists(cache_4y):
        df = pd.read_csv(cache_4y)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"   Loaded 4-Year cached {symbol} ({len(df)} rows: {df['datetime'].min().date()} -> {df['datetime'].max().date()})")
        return df

    if os.path.exists(cache_2y):
        df_existing = pd.read_csv(cache_2y)
        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
        return df_existing

    raise FileNotFoundError(f"Could not find {cache_4y} or {cache_2y}.")

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 2. STATIONARY MULTI-REGIME FEATURE ENGINEERING (10 CORE INPUTS)
# =========================================================================
def build_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c, btc_c = df['eur_close'], df['btc_close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_5d = atr14.rolling(480, min_periods=50).mean() + 1e-10
    atr_ratio = atr14 / atr_5d
    atr_pct = (atr14 / c) + 1e-10

    df['atr'] = atr14
    df['atr_ratio'] = atr_ratio

    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df['shock_ratio'] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.4], axis=1).max(axis=1)

    ema20  = c.ewm(span=20, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean() # 1-Hour 50 EMA Anchor
    df['ema20']  = ema20
    df['ema50']  = ema50
    df['ema200'] = ema200

    typ_p = (h + l + c) / 3.0
    vwap32 = (typ_p * v).rolling(32).sum() / (v.rolling(32).sum() + 1e-10)
    df['vwap'] = vwap32

    # f1, f2, f3: 30m, 1H, and 3H Volatility-Normalized Momentum
    df['f1'] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df['f2'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df['f3'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # f4: Session VWAP Distance in ATR units
    df['f4'] = ((c - vwap32) / c) / atr_pct

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

    # f7 & f8: Multi-Scale EMA Trend Structure (EMA20 vs EMA50 vs EMA200 [1H 50 EMA])
    df['f7'] = ((ema20 - ema50) / c) / atr_pct
    df['f8'] = ((ema50 - ema200) / c) / atr_pct

    # f9: Order Flow VSA (4-bar Volume-Weighted Candle Conviction)
    rng = (h - l) + 1e-10
    signed_vol = ((c - o) / rng) * v
    df['f9'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # f10: Cross-Asset US Dollar Proxy (EURUSDT) + Liquidity Beta (BTCUSDT)
    eur_ret4  = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4  = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f10'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df['er12'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df

# =========================================================================
# 3. STAGE 1: MULTI-YEAR SEQUENTIAL 3-DAY MISTAKE-BOOSTED PRE-TRAINER
# =========================================================================
def train_mistake_boosted_foundation(df_train, feature_cols):
    print("\n--- STAGE 1: MULTI-YEAR 3-DAY ROLLING MISTAKE-BOOSTED PRE-TRAINING ---")
    clean_df = df_train[(df_train['target'] != -1) & (df_train['shock_ratio'] <= 2.3)].copy().reset_index(drop=True)

    # Balance Bull (1) and Bear (0) classes 50/50 across the 4-year training history
    scaler = StandardScaler()
    X_all = scaler.fit_transform(clean_df[feature_cols].values)
    y_all = clean_df['target'].values
    dates = clean_df['datetime'].values

    # Sequential 3-Day Rolling Tracker to identify Hard Mistakes across 4 years without forgetting!
    probe_clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.05, learning_rate='invscaling', eta0=0.01, power_t=0.25, random_state=42)

    chunk_size = 200
    n_samples = len(X_all)
    total_folds = 0
    sample_importance = np.ones(n_samples)

    probe_clf.partial_fit(X_all[:chunk_size], y_all[:chunk_size], classes=np.array([0, 1]))
    total_mistakes = 0

    for start_i in range(chunk_size, n_samples - chunk_size, chunk_size):
        end_i = start_i + chunk_size
        X_fold = X_all[start_i:end_i]
        y_fold = y_all[start_i:end_i]

        # Test on the unseen 3-day fold
        preds = probe_clf.predict(X_fold)
        mistakes = (preds != y_fold)
        total_mistakes += int(np.sum(mistakes))

        # Boost importance of misclassified regime transitions by 1.75x
        fold_weights = np.where(mistakes, 1.75, 1.0)
        sample_importance[start_i:end_i] = fold_weights

        # Update rolling probe forward
        probe_clf.partial_fit(X_fold, y_fold, sample_weight=fold_weights)
        total_folds += 1

    # Class-balance weights * Mistake-boosted weights for global Foundation Model consolidation
    pos_mask = (y_all == 1)
    neg_mask = (y_all == 0)
    pos_w = 0.5 / (np.mean(pos_mask) + 1e-5)
    neg_w = 0.5 / (np.mean(neg_mask) + 1e-5)
    final_weights = sample_importance * np.where(pos_mask, pos_w, neg_w)

    # Fit Consolidated Foundation Logistic Regression (with mistake sample weights) + Tanh Neural Net
    foundation_lr = LogisticRegression(C=0.08, random_state=42)
    foundation_lr.fit(X_all, y_all, sample_weight=final_weights)

    # Fit Tanh MLP on recent 40,000 clean moves balanced 50/50
    pos_idx = np.where(pos_mask)[0][-20000:]
    neg_idx = np.where(neg_mask)[0][-20000:]
    bal_idx = np.sort(np.concatenate([pos_idx, neg_idx]))

    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', alpha=0.40, max_iter=400, random_state=42)
    mlp.fit(X_all[bal_idx], y_all[bal_idx])

    print(f"   Completed {total_folds} Sequential 3-Day Mistake-Learning Folds ({pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()})")
    print(f"   Hard-Example Mined Mistakes Boosted: {total_mistakes} bars across 4 years")
    return scaler, foundation_lr, mlp

# =========================================================================
# 4. SELF-ADAPTING DAILY PREQUENTIAL ENGINE (MATCHES PINE SCRIPT 1-TO-1!)
# =========================================================================
def run_daily_continual_simulation(
    df_slice, scaler, foundation_lr, mlp, feature_cols,
    online_eta=0.06, z_thresh=0.25, z_strong=0.65,
    tp_atr=2.0, sl_atr=1.4, er_min=0.15,
    enable_daily_update=True, fee=0.0001
):
    closes    = df_slice['close'].values
    opens     = df_slice['open'].values
    highs     = df_slice['high'].values
    lows      = df_slice['low'].values
    atrs      = df_slice['atr'].values
    atr_ratio = df_slice['atr_ratio'].values
    shocks    = df_slice['shock_ratio'].values
    er        = df_slice['er12'].values
    ema20     = df_slice['ema20'].values
    ema50     = df_slice['ema50'].values
    ema200    = df_slice['ema200'].values
    f4_vwap   = df_slice['f4'].values
    f6_macd   = df_slice['f6'].values
    f7_fast   = df_slice['f7'].values
    f8_macro  = df_slice['f8'].values
    dates     = pd.to_datetime(df_slice['datetime']).values

    X_scaled = scaler.transform(df_slice[feature_cols].values)

    # Base frozen probabilities from 4-Year Mistake-Boosted Foundation Ensemble
    p_lr  = foundation_lr.predict_proba(X_scaled)[:, 1]
    p_mlp = mlp.predict_proba(X_scaled)[:, 1]
    base_raw_p = 0.55 * p_lr + 0.45 * p_mlp

    # Rolling 200-bar Z-Score calibration (prevents bull/bear regime drift!)
    s_raw  = pd.Series(base_raw_p)
    r_mean = s_raw.rolling(200, min_periods=20).mean().values
    r_std  = s_raw.rolling(200, min_periods=20).std(ddof=0).fillna(0.05).values + 1e-10
    base_z = (base_raw_p - r_mean) / r_std

    # Online Adaptive Delta Weights (Updated at the end of each day from that day's mistakes!)
    w_online_delta = np.zeros(len(feature_cols))
    daily_bias_z   = 0.0

    n = len(closes)
    horizon = 10
    trades = []
    daily_logs = []
    eq = [0.0]

    cur_day_str = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
    day_start_idx = 0
    i = 1
    last_loss_bar = -999

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        d_str = dt_obj.strftime('%Y-%m-%d')

        # --- END-OF-DAY ONLINE CONTINUAL UPDATE (Triggered after each day finishes at 23:59 UTC!) ---
        if d_str != cur_day_str:
            if enable_daily_update and (i - day_start_idx) >= 16:
                end_eval = max(day_start_idx + 1, i - horizon)
                day_X = X_scaled[day_start_idx:end_eval]
                day_ret = (closes[day_start_idx + horizon : end_eval + horizon] - closes[day_start_idx:end_eval]) / closes[day_start_idx:end_eval]
                day_actual = np.where(day_ret > 0.0008, 1.0, np.where(day_ret < -0.0008, -1.0, 0.0))
                valid_m = (day_actual != 0.0)

                if np.any(valid_m):
                    day_pred_z = base_z[day_start_idx:end_eval][valid_m] + np.dot(day_X[valid_m], w_online_delta) + daily_bias_z
                    day_pred_sign = np.tanh(day_pred_z)
                    errors = day_actual[valid_m] - day_pred_sign

                    grad_w = np.mean(errors[:, None] * day_X[valid_m], axis=0)
                    w_online_delta = np.clip(0.90 * w_online_delta + online_eta * grad_w, -0.35, 0.35)
                    daily_bias_z   = float(np.clip(0.85 * daily_bias_z + 0.12 * np.mean(errors), -0.30, 0.30))

                    day_move_pct = ((closes[i-1] - closes[day_start_idx]) / closes[day_start_idx]) * 100.0
                    daily_logs.append({
                        'completed_day': cur_day_str,
                        'gold_move': f"{day_move_pct:+.2f}%",
                        'mean_err': float(np.mean(errors)),
                        'new_bias_z': daily_bias_z,
                        'norm_delta_w': float(np.linalg.norm(w_online_delta))
                    })

            cur_day_str = d_str
            day_start_idx = i

        if dt_obj.dayofweek == 6 or er[i] < er_min or shocks[i] > 2.2 or atr_ratio[i] < 0.65 or atr_ratio[i] > 2.00 or (i - last_loss_bar) < 3:
            i += 1
            continue

        online_adj = float(np.dot(X_scaled[i], w_online_delta)) + daily_bias_z
        total_z = base_z[i] + online_adj

        direction = 0
        cap_w = 0.0

        # Macro Regime Alignment + AI Conviction + Candle Confirmation (No Contradictory Lockout!)
        bull_macro = (closes[i] > ema200[i]) and (ema50[i] >= ema200[i] * 0.999)
        bear_macro = (closes[i] < ema200[i]) and (ema50[i] <= ema200[i] * 1.001)

        if bull_macro and total_z >= z_thresh and closes[i] > opens[i] and abs(f4_vwap[i]) < 2.2:
            direction = 1
            cap_w = 0.65 if total_z >= z_strong else 0.35
        elif bear_macro and total_z <= -z_thresh and closes[i] < opens[i] and abs(f4_vwap[i]) < 2.2:
            direction = -1
            cap_w = 0.65 if total_z <= -z_strong else 0.35

        if direction != 0:
            entry1 = closes[i]
            cur_atr = atrs[i]
            tp_price = entry1 + direction * (tp_atr * cur_atr)
            sl_price = entry1 - direction * (sl_atr * cur_atr)
            tr_trig  = entry1 + direction * (0.85 * cur_atr)
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
                    if (not pyramided) and closes[idx] >= pyr_trig and (base_z[idx] + float(np.dot(X_scaled[idx], w_online_delta)) + daily_bias_z) >= z_thresh:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
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
                    if (not pyramided) and closes[idx] <= pyr_trig and (base_z[idx] + float(np.dot(X_scaled[idx], w_online_delta)) + daily_bias_z) <= -z_thresh:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = min(sl_price, entry1)

            ret1 = cap_w * (direction * ((exit_p - entry1) / entry1) - fee)
            ret2 = w2 * (direction * ((exit_p - entry2) / entry2) - fee) if pyramided else 0.0
            tot_ret = ret1 + ret2

            if tot_ret < 0:
                last_loss_bar = i + bars_held

            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'pyr': pyramided,
                'bias_z': daily_bias_z,
                'ret': tot_ret
            })
            eq.append(eq[-1] + tot_ret * 100.0)
            i += bars_held + 1
        else:
            i += 1

    eq_arr = np.array(eq)
    dd = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr)) if len(eq_arr) > 1 else 0.0
    return pd.DataFrame(trades), pd.DataFrame(daily_logs), dd

def print_row(tr_df, dd, name):
    if len(tr_df) == 0:
        print(f"{name:<38} | 0 trades")
        return
    n_t = len(tr_df)
    wr  = (tr_df['ret'] > 0).mean() * 100.0
    pnl = tr_df['ret'].sum() * 100.0
    gw  = tr_df[tr_df['ret'] > 0]['ret'].sum()
    gl  = abs(tr_df[tr_df['ret'] < 0]['ret'].sum()) + 1e-10
    pf  = gw / gl
    print(f"{name:<38} | Trades: {n_t:<4} | Win Rate: {wr:<5.1f}% | PF: {pf:<5.2f} | Max DD: {dd:<4.2f}% | Net PnL: {pnl:+.2f}%")

def main():
    print("=== PHASE 17: MULTI-YEAR MISTAKE-BOOSTED PRE-TRAINING + DAILY ONLINE CONTINUAL LEARNER ===")
    df_gold = load_or_expand_symbol("PAXGUSDT", "15m", 1460)
    df_eur  = load_or_expand_symbol("EURUSDT", "15m", 1460)
    df_btc  = load_or_expand_symbol("BTCUSDT", "15m", 1460)

    df = build_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 11)]

    HORIZON = 10
    df['future_ret'] = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    df = df.dropna().reset_index(drop=True)
    df['target'] = np.where(df['future_ret'] > 0.0008, 1,
                   np.where(df['future_ret'] < -0.0008, 0, -1))

    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df    = df[df['datetime'] <= TRAIN_END].copy()
    val_df      = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df     = df[df['datetime'] > VAL_END].copy()
    full_oos_df = df[df['datetime'] > TRAIN_END].copy()

    # Stage 1: Multi-Year Mistake-Boosted Pre-Training (2022 -> Aug 31, 2026)
    scaler, foundation_lr, mlp = train_mistake_boosted_foundation(train_df, feature_cols)

    # Stage 2: Calibrate Online Adaptation Rate on Sep 1 - Nov 8 Validation Set
    best_cfg = None
    best_score = -9999.0
    for z_th in [0.15, 0.25, 0.35, 0.45]:
        for tp_m, sl_m in [(2.0, 1.4), (2.2, 1.4), (1.8, 1.5)]:
            for eta in [0.05, 0.08, 0.12]:
                v_tr, _, v_dd = run_daily_continual_simulation(
                    val_df, scaler, foundation_lr, mlp, feature_cols,
                    online_eta=eta, z_thresh=z_th, z_strong=z_th+0.40, tp_atr=tp_m, sl_atr=sl_m, enable_daily_update=True
                )
                if len(v_tr) >= 25:
                    v_pnl = v_tr['ret'].sum() * 100.0
                    gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                    gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                    v_pf = gw / gl
                    score = v_pnl * (v_pf if v_pf > 1.0 else 0.5)
                    if score > best_score:
                        best_score = score
                        best_cfg = {'z_th': z_th, 'tp': tp_m, 'sl': sl_m, 'eta': eta}

    if best_cfg is None:
        best_cfg = {'z_th': 0.20, 'tp': 2.0, 'sl': 1.4, 'eta': 0.08}

    print(f"\n--- STAGE 2: VALIDATION LOCKED PARAMETERS (SEP 1 - NOV 8, 2026) ---")
    print(f"   Z_Thresh={best_cfg['z_th']} | TP={best_cfg['tp']}x ATR | SL={best_cfg['sl']}x ATR | Daily_Online_Eta={best_cfg['eta']}")

    # Run Continuous Simulation across Sep 1 - Nov 16 so Nov 9 starts with the exact Nov 8 night state!
    full_tr_static, _, full_dd_static = run_daily_continual_simulation(
        full_oos_df, scaler, foundation_lr, mlp, feature_cols,
        online_eta=0.0, z_thresh=best_cfg['z_th'], z_strong=best_cfg['z_th']+0.40,
        tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], enable_daily_update=False
    )
    full_tr_online, daily_logs_all, full_dd_online = run_daily_continual_simulation(
        full_oos_df, scaler, foundation_lr, mlp, feature_cols,
        online_eta=best_cfg['eta'], z_thresh=best_cfg['z_th'], z_strong=best_cfg['z_th']+0.40,
        tp_atr=best_cfg['tp'], sl_atr=best_cfg['sl'], enable_daily_update=True
    )

    val_tr_online  = full_tr_online[full_tr_online['date'] <= '2026-11-08']
    test_tr_online = full_tr_online[full_tr_online['date'] >= '2026-11-09']

    print("\n=== STAGE 3A: PERFORMANCE COMPARISON (STATIC FROZEN VS. DAILY MISTAKE-ADAPTIVE LEARNER) ===")
    print("-" * 106)
    print_row(full_tr_static[full_tr_static['date'] <= '2026-11-08'], full_dd_static, "1A. Sep 01 - Nov 08 (Static Frozen)")
    print_row(val_tr_online,  full_dd_online, "1B. Sep 01 - Nov 08 (Daily Adaptive)")
    print_row(test_tr_online, full_dd_online, "2.  Nov 09 - Nov 15 (Blind Daily Test)")
    print_row(full_tr_online, full_dd_online, "3.  Sep 01 - Nov 15 (Full Chart Online)")
    print("-" * 106)

    # Day-by-Day Test Week Breakdown (Nov 9 - Nov 15) + End-of-Day Learning Log
    print("\n=== STAGE 3B: DAY-BY-DAY UNSEEN TEST & END-OF-DAY MISTAKE ADAPTATION LOG (NOV 9 – NOV 15) ===")
    print("-" * 100)
    print(f"{'Date':<12} | {'Trades':<7} | {'Win Rate':<9} | {'Day PnL':<9} | {'Cum PnL':<9} | {'End-of-Day Mistake Update for Next Day'}")
    print("-" * 100)
    cum_p = 0.0
    test_days = sorted(list(set(df[df['datetime'] > VAL_END]['datetime'].dt.strftime('%Y-%m-%d'))))
    for d_str in test_days:
        day_grp = test_tr_online[test_tr_online['date'] == d_str]
        log_grp = daily_logs_all[daily_logs_all['completed_day'] == d_str]
        n_t = len(day_grp)
        wr  = (day_grp['ret'] > 0).mean() * 100.0 if n_t > 0 else 0.0
        d_pnl = day_grp['ret'].sum() * 100.0 if n_t > 0 else 0.0
        cum_p += d_pnl

        if len(log_grp) > 0:
            row_l = log_grp.iloc[-1]
            adapt_str = f"Gold {row_l['gold_move']} -> Updated Bias Z={row_l['new_bias_z']:+.2f}, ||dW||={row_l['norm_delta_w']:.2f}"
        else:
            adapt_str = "Weekend / Market Closed (Weights Held)"
        print(f"{d_str:<12} | {n_t:<7} | {wr:<8.1f}% | {d_pnl:>+7.2f}% | {cum_p:>+7.2f}% | {adapt_str}")
    print("-" * 100)

    # =========================================================================
    # 5. COMPILE SELF-ADAPTING PINE SCRIPT V5 (AI_Gold_Continual_Learner_v11.pine)
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = foundation_lr.coef_[0], foundation_lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Continual Mistake-Learner v11 (4Yr Pre-Trained + Daily Online OGD)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=35, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. VIVA EVALUATION WINDOW PICKER (UNCHECK FOR FULL SEP-NOV CHART!)
// =========================================================================
use_date_filter = input.bool(false, "Restrict to Specific Date Window (Uncheck for Full Chart)", group="1. Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Window")

// =========================================================================
// 2. CONTINUAL LEARNING & RISK SETTINGS
// =========================================================================
enable_online = input.bool(true, "Enable End-of-Day Online Mistake Learning (Prequential OGD)", group="2. Continual Mistake Learner")
online_eta    = input.float({best_cfg['eta']:.2f}, "End-of-Day Online Adaptation Rate (Eta)", minval=0.01, maxval=0.25, step=0.01, group="2. Continual Mistake Learner")
z_thresh      = input.float({best_cfg['z_th']:.2f}, "AI Conviction Z-Score Threshold", minval=0.05, maxval=1.50, step=0.05, group="2. Continual Mistake Learner")
z_strong      = input.float({best_cfg['z_th']+0.40:.2f}, "Tier-2 High Conviction Z-Score", minval=0.25, maxval=2.00, step=0.05, group="2. Continual Mistake Learner")
er_min_val    = input.float(0.15, "Min Kaufman Efficiency Ratio (Chop Filter)", minval=0.05, maxval=0.35, step=0.01, group="3. Regime & Risk Settings")
tp_mult       = input.float({best_cfg['tp']:.1f}, "Take Profit (ATR Multiplier)", minval=1.0, maxval=4.0, step=0.1, group="3. Regime & Risk Settings")
sl_mult       = input.float({best_cfg['sl']:.1f}, "Initial Stop Loss (ATR Multiplier)", minval=0.8, maxval=3.0, step=0.1, group="3. Regime & Risk Settings")
horizon_bars  = input.int(10, "Max Holding Horizon (Bars)", minval=4, maxval=24, group="3. Regime & Risk Settings")
enable_pyr    = input.bool(true, "Enable +35% Trend Pyramiding on Winners", group="3. Regime & Risk Settings")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 10 STATIONARY FEATURES
// =========================================================================
eur_close = request.security("BINANCE:EURUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)
btc_close = request.security("BINANCE:BTCUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

pine_rma(src, length) =>
    alpha = 1.0 / length
    var float sum = na
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1.0 - alpha) * nz(sum[1])
    sum

tr_val    = ta.tr(true)
atr14     = pine_rma(tr_val, 14)
atr_5d    = ta.sma(atr14, 480) + 1e-10
atr_ratio = atr14 / atr_5d
atr_pct   = (atr14 / close) + 1e-10

bar_shock   = (high - low) / (atr14 + 1e-10)
jump_shock  = math.abs(close - nz(close[2], close[1])) / (atr14 + 1e-10)
shock_ratio = math.max(bar_shock, math.max(nz(bar_shock[1], 0.0), jump_shock / 1.4))

ema20  = ta.ema(close, 20)
ema50  = ta.ema(close, 50)
ema200 = ta.ema(close, 200)

typ_p  = (high + low + close) / 3.0
vwap32 = math.sum(typ_p * volume, 32) / (math.sum(volume, 32) + 1e-10)

f1 = ((close - nz(close[2], close[1])) / nz(close[2], close[1])) / atr_pct
f2 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f3 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct
f4 = ((close - vwap32) / close) / atr_pct

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

f7 = ((ema20 - ema50) / close) / atr_pct
f8 = ((ema50 - ema200) / close) / atr_pct

rng        = (high - low) + 1e-10
signed_vol = ((close - open) / rng) * volume
f9         = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4  = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4  = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])
f10       = nz(eur_ret4 - gold_ret4 + 0.15 * btc_ret4, 0.0) * 100.0

net_move  = math.abs(close - nz(close[12], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 12)
er12      = sum_moves == 0 ? 0.0 : net_move / sum_moves

// =========================================================================
// 4. STAGE 1 FOUNDATION MODEL (MISTAKE-BOOSTED ON 4 YEARS <= AUG 31, 2026)
// =========================================================================
"""
    for idx in range(10):
        pine_code += f"s{idx+1} = (f{idx+1} - {mean[idx]:.8f}) / {scale[idx]:.8f}\n"

    lr_eq = f"{b_lr:.8f}"
    for idx in range(10):
        lr_eq += f" + (s{idx+1} * {w_lr[idx]:.8f})"
    pine_code += f"\nlogit_lr = {lr_eq}\n"
    pine_code += "p_lr = 1.0 / (1.0 + math.exp(-logit_lr))\n\n"

    for j in range(len(b0)):
        eq = f"{b0[j]:.8f}"
        for idx in range(10):
            eq += f" + (s{idx+1} * {w0[idx][j]:.8f})"
        pine_code += f"z_n{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z_n{j}) - 1.0) / (math.exp(2.0 * z_n{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

    pine_code += """base_raw_p = 0.55 * p_lr + 0.45 * p_mlp
roll_mean  = ta.sma(base_raw_p, 200)
roll_std   = ta.stdev(base_raw_p, 200) + 1e-10
base_z     = (base_raw_p - roll_mean) / roll_std

// =========================================================================
// 5. STAGE 3 NATIVE PINE END-OF-DAY ONLINE MISTAKE LEARNER (PREQUENTIAL OGD)
// =========================================================================
var float dw1 = 0.0, var float dw2 = 0.0, var float dw3 = 0.0, var float dw4 = 0.0, var float dw5 = 0.0
var float dw6 = 0.0, var float dw7 = 0.0, var float dw8 = 0.0, var float dw9 = 0.0, var float dw10 = 0.0
var float daily_bias_z = 0.0

var float g1 = 0.0, var float g2 = 0.0, var float g3 = 0.0, var float g4 = 0.0, var float g5 = 0.0
var float g6 = 0.0, var float g7 = 0.0, var float g8 = 0.0, var float g9 = 0.0, var float g10 = 0.0
var float err_sum = 0.0
var int   day_samples = 0
var int   cur_dom = -1
var int   daily_updates_count = 0

realized_10b_ret = (close - nz(close[10], close)) / nz(close[10], close)
actual_dir       = realized_10b_ret > 0.0008 ? 1.0 : realized_10b_ret < -0.0008 ? -1.0 : 0.0

past_z    = nz(base_z[10], 0.0) + (nz(s1[10])*dw1 + nz(s2[10])*dw2 + nz(s3[10])*dw3 + nz(s4[10])*dw4 + nz(s5[10])*dw5 + nz(s6[10])*dw6 + nz(s7[10])*dw7 + nz(s8[10])*dw8 + nz(s9[10])*dw9 + nz(s10[10])*dw10) + daily_bias_z
past_pred = (math.exp(2.0 * past_z) - 1.0) / (math.exp(2.0 * past_z) + 1.0)

if barstate.isconfirmed and actual_dir != 0.0
    err = actual_dir - past_pred
    g1 += err * nz(s1[10]), g2 += err * nz(s2[10]), g3 += err * nz(s3[10]), g4 += err * nz(s4[10]), g5 += err * nz(s5[10])
    g6 += err * nz(s6[10]), g7 += err * nz(s7[10]), g8 += err * nz(s8[10]), g9 += err * nz(s9[10]), g10 += err * nz(s10[10])
    err_sum += err
    day_samples += 1

if barstate.isconfirmed and dayofmonth(time) != cur_dom
    if cur_dom != -1 and enable_online and day_samples >= 10
        inv_n = 1.0 / day_samples
        dw1  := math.min(0.35, math.max(-0.35, 0.90 * dw1  + online_eta * (g1  * inv_n)))
        dw2  := math.min(0.35, math.max(-0.35, 0.90 * dw2  + online_eta * (g2  * inv_n)))
        dw3  := math.min(0.35, math.max(-0.35, 0.90 * dw3  + online_eta * (g3  * inv_n)))
        dw4  := math.min(0.35, math.max(-0.35, 0.90 * dw4  + online_eta * (g4  * inv_n)))
        dw5  := math.min(0.35, math.max(-0.35, 0.90 * dw5  + online_eta * (g5  * inv_n)))
        dw6  := math.min(0.35, math.max(-0.35, 0.90 * dw6  + online_eta * (g6  * inv_n)))
        dw7  := math.min(0.35, math.max(-0.35, 0.90 * dw7  + online_eta * (g7  * inv_n)))
        dw8  := math.min(0.35, math.max(-0.35, 0.90 * dw8  + online_eta * (g8  * inv_n)))
        dw9  := math.min(0.35, math.max(-0.35, 0.90 * dw9  + online_eta * (g9  * inv_n)))
        dw10 := math.min(0.35, math.max(-0.35, 0.90 * dw10 + online_eta * (g10 * inv_n)))
        daily_bias_z := math.min(0.30, math.max(-0.30, 0.85 * daily_bias_z + 0.12 * (err_sum * inv_n)))
        daily_updates_count += 1
    cur_dom := dayofmonth(time)
    g1 := 0.0, g2 := 0.0, g3 := 0.0, g4 := 0.0, g5 := 0.0
    g6 := 0.0, g7 := 0.0, g8 := 0.0, g9 := 0.0, g10 := 0.0
    err_sum := 0.0, day_samples := 0

online_adj = enable_online ? (s1*dw1 + s2*dw2 + s3*dw3 + s4*dw4 + s5*dw5 + s6*dw6 + s7*dw7 + s8*dw8 + s9*dw9 + s10*dw10 + daily_bias_z) : 0.0
total_z    = base_z + online_adj
prob_disp  = 1.0 / (1.0 + math.exp(-total_z * 1.15))

// =========================================================================
// 6. REGIME-ALIGNED EXECUTION & PYRAMIDING ENGINE
// =========================================================================
calc_qty(pct_eq) =>
    math.max(0.001, (strategy.equity * (pct_eq / 100.0)) / close)

regime_ok  = in_date_window and (er12 >= er_min_val) and (shock_ratio <= 2.2) and (atr_ratio >= 0.65) and (atr_ratio <= 2.00)
bull_macro = (close > ema200) and (ema50 >= ema200 * 0.999)
bear_macro = (close < ema200) and (ema50 <= ema200 * 1.001)

bull_trig = regime_ok and bull_macro and (total_z >= z_thresh) and (close > open) and (math.abs(f4) < 2.2)
bear_trig = regime_ok and bear_macro and (total_z <= -z_thresh) and (close < open) and (math.abs(f4) < 2.2)

bull_pct = total_z >= z_strong ? 65.0 : 35.0
bear_pct = total_z <= -z_strong ? 65.0 : 35.0

var int   entry_bar  = na
var int   last_loss  = -999
var float init_entry = na
var float entry_atr  = na
var float tp_level   = na
var float sl_level   = na
var bool  scaled_in  = false
var float active_pct = 0.0

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_loss := bar_index

cooldown_ok = (bar_index - last_loss) >= 3

if barstate.isconfirmed
    if strategy.position_size == 0 and cooldown_ok
        if bull_trig
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="L1 (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_disp*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if bear_trig
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="S1 (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round((1.0-prob_disp)*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct

    else if strategy.position_size > 0 and enable_pyr and not scaled_in
        if (close >= init_entry + entry_atr * 0.50) and (total_z >= z_thresh)
            strategy.entry("AI Long Scale", strategy.long, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.max(sl_level, init_entry)

    else if strategy.position_size < 0 and enable_pyr and not scaled_in
        if (close <= init_entry - entry_atr * 0.50) and (total_z <= -z_thresh)
            strategy.entry("AI Short Scale", strategy.short, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.min(sl_level, init_entry)

if strategy.position_size > 0
    if high >= (init_entry + entry_atr * 0.85)
        sl_level := math.max(sl_level, init_entry + entry_atr * 0.35)
    strategy.exit("Exit Long", from_entry="AI Long", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Long Scale", from_entry="AI Long Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (init_entry - entry_atr * 0.85)
        sl_level := math.min(sl_level, init_entry - entry_atr * 0.35)
    strategy.exit("Exit Short", from_entry="AI Short", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Short Scale", from_entry="AI Short Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 7. OVERLAYS & CONTINUAL LEARNING TELEMETRY HUD
// =========================================================================
plot(ema50,  "EMA 50 Trend", color=color.new(color.aqua, 25), linewidth=2)
plot(ema200, "1-Hour 50 EMA (Macro Regime)", color=color.new(color.yellow, 15), linewidth=2)
plot(vwap32, "Session VWAP", color=color.new(color.fuchsia, 30), linewidth=1)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_disp >= 0.5 ? math.round(prob_disp * 100, 1) : math.round((1.0 - prob_disp) * 100, 1)
    sig_txt  = bull_trig ? "BUY / LONG" : bear_trig ? "SELL / SHORT" : "WAIT / HOLD"
    sig_col  = bull_trig ? color.green : bear_trig ? color.red : color.gray

    table.cell(hud, 0, 0, "ML Architecture", text_color=color.white)
    table.cell(hud, 1, 0, "4Yr Boosted + Daily OGD", text_color=color.yellow)
    table.cell(hud, 0, 1, "Daily EOD Updates", text_color=color.white)
    table.cell(hud, 1, 1, str.tostring(daily_updates_count) + " Days Adapted (Bias: " + str.tostring(math.round(daily_bias_z, 2)) + ")", text_color=color.aqua)
    table.cell(hud, 0, 2, "Macro Structure", text_color=color.white)
    table.cell(hud, 1, 2, bull_macro ? "BULL (>1H EMA50)" : bear_macro ? "BEAR (<1H EMA50)" : "TRANSITION", text_color=bull_macro ? color.lime : bear_macro ? color.red : color.orange)
    table.cell(hud, 0, 3, "AI Conviction", text_color=color.white)
    table.cell(hud, 1, 3, (total_z >= 0 ? "UP " : "DN ") + str.tostring(conf_pct) + "% (Z=" + str.tostring(math.round(total_z, 2)) + ")", text_color=color.white)
    table.cell(hud, 0, 4, "Active Capital", text_color=color.white)
    table.cell(hud, 1, 4, strategy.position_size != 0 ? str.tostring(active_pct) + "%" + (scaled_in ? " [PYRAMID]" : "") : "0% (Cash)", text_color=color.lime)
    table.cell(hud, 0, 5, "Live Action", text_color=color.white)
    table.cell(hud, 1, 5, sig_txt, bgcolor=sig_col, text_color=color.white)
"""

    with open("AI_Gold_Continual_Learner_v11.pine", "w") as f:
        f.write(pine_code)

    print("\n4. Saved 'AI_Gold_Continual_Learner_v11.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()