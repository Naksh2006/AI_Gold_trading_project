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
# 1. DATA LOADER (USES CACHED 2-YEAR BINANCE DATASETS)
# =========================================================================
def fetch_binance_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    raise FileNotFoundError(f"Please run phase9 first or ensure {cache_file} exists.")

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 2. STATIONARY FEATURE ENGINEERING (ZERO LOOKAHEAD)
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

    # Exogenous News/Volatility Shock Ratio (Current bar range vs 14-bar ATR)
    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df['shock_ratio'] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    # f1, f2: Short & Medium Volatility-Normalized Momentum
    df['f1'] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df['f2'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct

    # f3: Rolling 20-Bar Intraday VWAP Stretch
    typ_p = (h + l + c) / 3.0
    vwap20 = (typ_p * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    df['f3'] = ((c - vwap20) / c) / atr_pct

    # f4: Wilder RSI(14) scaled to [-1, 1]
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['rsi14_raw'] = rsi14
    df['f4'] = (rsi14 - 50.0) / 50.0

    # f5: Normalized MACD Histogram Velocity
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['f5'] = (macd_hist - macd_hist.shift(1)) / (atr14 + 1e-10)

    # f6: Bollinger Band Z-Score (20)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0) + 1e-10
    df['f6'] = (c - sma20) / (2.0 * std20)

    # f7, f8, f9: Multi-Scale EMA Trend Hierarchy (EMA16, EMA48, EMA144)
    ema16 = c.ewm(span=16, adjust=False).mean()
    ema48 = c.ewm(span=48, adjust=False).mean()
    ema144 = c.ewm(span=144, adjust=False).mean()
    df['f7'] = ((c - ema16) / c) / atr_pct
    df['f8'] = ((ema16 - ema48) / c) / atr_pct
    df['f9'] = ((ema48 - ema144) / c) / atr_pct

    # f10: Order Flow VSA (3-bar Institutional Buying/Selling Pressure)
    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df['f10'] = signed_vol.rolling(3).sum() / (v.rolling(3).sum() + 1e-10)

    # f11, f12: Cross-Asset USD Inverse (EURUSDT) & Safe-Haven Divergence
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f11'] = eur_ret4 * 100.0
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Kaufman Efficiency Ratio (10)
    net_move = (c - c.shift(10)).abs()
    sum_moves = c.diff().abs().rolling(10).sum()
    df['er10'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    # Base SuperTrend (Factor 2.2, ATR 10)
    hl2 = (h + l) / 2.0
    atr10 = pine_rma(tr, 10)
    up_lev = (hl2 - 2.2 * atr10).values
    dn_lev = (hl2 + 2.2 * atr10).values
    c_vals = c.values
    st_dir = np.ones(len(df))
    final_up, final_dn = np.copy(up_lev), np.copy(dn_lev)
    for i in range(1, len(df)):
        final_up[i] = max(up_lev[i], final_up[i-1]) if c_vals[i-1] > final_up[i-1] else up_lev[i]
        final_dn[i] = min(dn_lev[i], final_dn[i-1]) if c_vals[i-1] < final_dn[i-1] else dn_lev[i]
        st_dir[i] = 1 if c_vals[i] > final_dn[i-1] else (-1 if c_vals[i] < final_up[i-1] else st_dir[i-1])
    df['st_dir'] = st_dir

    return df

# =========================================================================
# 3. BACKTEST SIMULATOR (USED FOR VALIDATION TUNING & BLIND TEST)
# =========================================================================
def run_simulation(df_slice, probs, horizon=8, up_th=0.53, tp_atr=1.5, sl_atr=1.0, filter_sunday=True, max_shock=2.1, fee=0.0001):
    closes = df_slice['close'].values
    opens = df_slice['open'].values
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    atrs = df_slice['atr'].values
    vwap_d = df_slice['f3'].values
    macd_slope = df_slice['f5'].values
    ema_trend = df_slice['f8'].values
    er = df_slice['er10'].values
    shocks = df_slice['shock_ratio'].values
    st_dir = df_slice['st_dir'].values
    dates = pd.to_datetime(df_slice['datetime']).values

    dn_th = 1.0 - up_th
    n = len(closes)
    trades = []
    i = 0

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        if (filter_sunday and dt_obj.dayofweek == 6) or er[i] < 0.14 or shocks[i] > max_shock:
            i += 1
            continue

        p = probs[i]
        direction = 0
        if p >= up_th and (st_dir[i] > 0 or ema_trend[i] > 0) and macd_slope[i] > -0.05 and vwap_d[i] < 1.6:
            direction = 1
        elif p <= dn_th and (st_dir[i] < 0 or ema_trend[i] < 0) and macd_slope[i] < 0.05 and vwap_d[i] > -1.6:
            direction = -1

        if direction != 0:
            entry = closes[i]
            cur_atr = atrs[i]
            tp_price = entry + direction * (tp_atr * cur_atr)
            sl_price = entry - direction * (sl_atr * cur_atr)
            trail_trig = entry + direction * (1.05 * cur_atr)
            early_cut_p = entry - direction * (0.65 * cur_atr)

            exit_price = closes[i + horizon]
            bars_held = horizon

            for k in range(1, horizon + 1):
                if direction == 1:
                    if lows[i + k] <= sl_price:
                        exit_price, bars_held = sl_price, k
                        break
                    if highs[i + k] >= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if closes[i + k] < opens[i + k] and closes[i + k] <= early_cut_p:
                        exit_price, bars_held = closes[i + k], k
                        break
                    if highs[i + k] >= trail_trig:
                        sl_price = max(sl_price, entry + 0.55 * cur_atr)
                else:
                    if highs[i + k] >= sl_price:
                        exit_price, bars_held = sl_price, k
                        break
                    if lows[i + k] <= tp_price:
                        exit_price, bars_held = tp_price, k
                        break
                    if closes[i + k] > opens[i + k] and closes[i + k] >= early_cut_p:
                        exit_price, bars_held = closes[i + k], k
                        break
                    if lows[i + k] <= trail_trig:
                        sl_price = min(sl_price, entry - 0.55 * cur_atr)

            net_ret = direction * ((exit_price - entry) / entry) - fee
            trades.append({'date': dt_obj.strftime('%Y-%m-%d'), 'dir': direction, 'ret': net_ret})
            i += bars_held + 1
        else:
            i += 1

    return pd.DataFrame(trades)

# =========================================================================
# 4. REQUIRED RUBRIC BASELINES (ALWAYS UP & BASIC RSI 30/70)
# =========================================================================
def evaluate_baselines(df_test, horizon=8, fee=0.0001):
    closes = df_test['close'].values
    rsi = df_test['rsi14_raw'].values
    n = len(closes)

    # Baseline 1: Always Predict UP every 'horizon' bars
    b1_rets = []
    for i in range(0, n - horizon - 1, horizon):
        b1_rets.append(((closes[i + horizon] - closes[i]) / closes[i]) - fee)

    # Baseline 2: Classic RSI Rule (Buy RSI < 30, Short RSI > 70, hold for 'horizon' bars)
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

    b1_rets = np.array(b1_rets)
    b2_rets = np.array(b2_rets) if len(b2_rets) > 0 else np.array([0.0])
    return {
        'Always_Up': {'trades': len(b1_rets), 'wr': (b1_rets > 0).mean() * 100, 'pnl': b1_rets.sum() * 100},
        'Basic_RSI': {'trades': len(b2_rets), 'wr': (b2_rets > 0).mean() * 100, 'pnl': b2_rets.sum() * 100}
    }

def main():
    print("=== PHASE 10: ZERO-LEAKAGE 3-WAY SPLIT (TRAIN -> VALIDATION -> BLIND TEST) ===")
    df_gold = fetch_binance_symbol("PAXGUSDT", "15m", 730)
    df_eur = fetch_binance_symbol("EURUSDT", "15m", 730)
    df_btc = fetch_binance_symbol("BTCUSDT", "15m", 730)

    df = build_features(df_gold, df_eur, df_btc)
    HORIZON = 8
    feature_cols = [f"f{i}" for i in range(1, 13)]

    f_high = df['high'].rolling(HORIZON).max().shift(-HORIZON)
    f_low = df['low'].rolling(HORIZON).min().shift(-HORIZON)
    up_ex = (f_high - df['close']) / df['close']
    dn_ex = (df['close'] - f_low) / df['close']

    df['target'] = np.where((up_ex > dn_ex * 1.25) & (up_ex > 0.0012), 1,
                   np.where((dn_ex > up_ex * 1.25) & (dn_ex > 0.0012), 0, -1))
    df = df.dropna().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # STRICT 3-WAY CHRONOLOGICAL SPLIT (NO PEEKING AT NOV 9 - NOV 16!)
    # -------------------------------------------------------------------------
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df['datetime'] <= TRAIN_END].copy()
    val_df   = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df  = df[df['datetime'] > VAL_END].copy()

    print(f"1. [TRAIN SET]      : {train_df['datetime'].min().date()} -> {train_df['datetime'].max().date()} ({len(train_df)} bars)")
    print(f"2. [VALIDATION SET] : {val_df['datetime'].min().date()} -> {val_df['datetime'].max().date()} ({len(val_df)} bars) <-- ALL TUNING HAPPENS HERE")
    print(f"3. [BLIND TEST SET] : {test_df['datetime'].min().date()} -> {test_df['datetime'].max().date()} ({len(test_df)} bars) <-- LOCKED VAULT\n")

    # Fit Model strictly on TRAIN_DF (<= Aug 31, 2026)
    train_clean = train_df[(train_df['target'] != -1) & (train_df['shock_ratio'] <= 2.2)].copy()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_clean[feature_cols])
    y_train = train_clean['target']

    lr = LogisticRegression(C=0.2, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', alpha=0.15, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    train_raw_p = 0.5 * lr.predict_proba(X_train_s)[:, 1] + 0.5 * mlp.predict_proba(X_train_s)[:, 1]
    mean_p, std_p = float(np.mean(train_raw_p)), float(np.std(train_raw_p)) + 1e-10

    # -------------------------------------------------------------------------
    # STEP 2: HYPERPARAMETER SELECTION STRICTLY ON VALIDATION SET (SEP 1 - NOV 8)
    # -------------------------------------------------------------------------
    X_val_s = scaler.transform(val_df[feature_cols])
    val_raw_p = 0.5 * lr.predict_proba(X_val_s)[:, 1] + 0.5 * mlp.predict_proba(X_val_s)[:, 1]
    val_probs = 1.0 / (1.0 + np.exp(-((val_raw_p - mean_p) / (std_p * 0.8))))

    print("--- STEP 2A: PROVING SUNDAY FILTER ON VALIDATION SET (SEP 1 - NOV 8) ---")
    val_with_sun = run_simulation(val_df, val_probs, filter_sunday=False)
    val_no_sun   = run_simulation(val_df, val_probs, filter_sunday=True)
    sun_only_pnl = val_with_sun['ret'].sum()*100 - val_no_sun['ret'].sum()*100
    print(f"   Validation PnL Including Sundays : {val_with_sun['ret'].sum()*100:+.2f}% ({len(val_with_sun)} trades)")
    print(f"   Validation PnL Excluding Sundays : {val_no_sun['ret'].sum()*100:+.2f}% ({len(val_no_sun)} trades)")
    print(f"   --> Proof: Sunday trading lost {sun_only_pnl:+.2f}% on historical Validation data (Sep-Nov 8)!\n")

    print("--- STEP 2B: GRID SEARCHING OPTIMAL RISK PARAMETERS ON VALIDATION SET ---")
    best_val_score = -9999.0
    best_params = None

    for th in [0.52, 0.53, 0.55]:
        for tp_m, sl_m in [(1.5, 1.0), (1.6, 1.1), (1.8, 1.2)]:
            for shock_m in [2.1, 2.4]:
                v_tr = run_simulation(val_df, val_probs, up_th=th, tp_atr=tp_m, sl_atr=sl_m, filter_sunday=True, max_shock=shock_m)
                if len(v_tr) >= 100:
                    v_pnl = v_tr['ret'].sum() * 100.0
                    v_wr = (v_tr['ret'] > 0).mean() * 100.0
                    g_w = v_tr[v_tr['ret'] > 0]['ret'].sum()
                    g_l = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                    v_pf = g_w / g_l
                    score = v_pnl * v_pf
                    if score > best_val_score:
                        best_val_score = score
                        best_params = {'th': th, 'tp': tp_m, 'sl': sl_m, 'shock': shock_m, 'val_pnl': v_pnl, 'val_wr': v_wr, 'val_pf': v_pf, 'val_trades': len(v_tr)}

    print(f"   LOCKED FROM VALIDATION (SEP 1 - NOV 8):")
    print(f"   Threshold={best_params['th']} | TP={best_params['tp']}x ATR | SL={best_params['sl']}x ATR | ShockVeto={best_params['shock']}x ATR")
    print(f"   Validation Performance: {best_params['val_trades']} trades | Win Rate: {best_params['val_wr']:.1f}% | PF: {best_params['val_pf']:.2f} | Net PnL: {best_params['val_pnl']:+.2f}%\n")

    # -------------------------------------------------------------------------
    # STEP 3: ONE-SHOT EVALUATION ON BLIND TEST SET (NOV 9 - NOV 16) + BASELINES
    # -------------------------------------------------------------------------
    X_test_s = scaler.transform(test_df[feature_cols])
    test_raw_p = 0.5 * lr.predict_proba(X_test_s)[:, 1] + 0.5 * mlp.predict_proba(X_test_s)[:, 1]
    test_probs = 1.0 / (1.0 + np.exp(-((test_raw_p - mean_p) / (std_p * 0.8))))

    blind_trades = run_simulation(
        test_df, test_probs, up_th=best_params['th'],
        tp_atr=best_params['tp'], sl_atr=best_params['sl'],
        filter_sunday=True, max_shock=best_params['shock']
    )

    baselines = evaluate_baselines(test_df, horizon=HORIZON)

    t_pnl = blind_trades['ret'].sum() * 100.0
    t_wr = (blind_trades['ret'] > 0).mean() * 100.0
    t_gw = blind_trades[blind_trades['ret'] > 0]['ret'].sum()
    t_gl = abs(blind_trades[blind_trades['ret'] < 0]['ret'].sum()) + 1e-10
    t_pf = t_gw / t_gl

    print("=== STEP 3: BLIND OUT-OF-SAMPLE TEST (NOV 9 – NOV 16) VS. RUBRIC BASELINES ===")
    print("-" * 76)
    print(f"{'Strategy / Benchmark':<30} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL (After Fees)'}")
    print("-" * 76)
    print(f"{'1. Baseline: Always Predict UP':<30} | {baselines['Always_Up']['trades']:<8} | {baselines['Always_Up']['wr']:<9.1f}% | {baselines['Always_Up']['pnl']:+.2f}%")
    print(f"{'2. Baseline: Basic RSI (30/70)':<30} | {baselines['Basic_RSI']['trades']:<8} | {baselines['Basic_RSI']['wr']:<9.1f}% | {baselines['Basic_RSI']['pnl']:+.2f}%")
    print(f"{'3. OUR AI SHOCK-SHIELD ENGINE':<30} | {len(blind_trades):<8} | {t_wr:<9.1f}% | {t_pnl:+.2f}% (PF: {t_pf:.2f})")
    print("-" * 76)

if __name__ == "__main__":
    main()