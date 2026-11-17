import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_cached_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file}. Please run phase9 first to cache data.")
    df = pd.read_csv(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. PAPER + POC RETEST + MACRO AI FEATURE ENGINEERING (12 INPUTS)
# =========================================================================
def build_paper_poc_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c, btc_c = df['eur_close'], df['btc_close']

    # ATR(14) - Wilder's Smoothing (Table 1, Bhatti 2026)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df['atr'] = atr14

    # Core EMAs from Paper Table 1: EMA 200 (Regime), EMA 50 (Pullback & Close-Trail), EMA 20 (Final-Leg Trail)
    ema20  = c.ewm(span=20, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df['ema20']  = ema20
    df['ema50']  = ema50
    df['ema200'] = ema200

    # Session / Rolling Institutional VWAP (Eq. 1, Bhatti 2026)
    typ_p = (h + l + c) / 3.0
    vwap_sess = (typ_p * v).rolling(32).sum() / (v.rolling(32).sum() + 1e-10)
    df['vwap'] = vwap_sess

    # Rolling Volume Profile Point of Control (POC) Approximation (Gold-POC-Retest Repo)
    # Weights prices by (Volume^2 / Range) to locate the high-volume institutional node over 48 bars (12 hours)
    vol_density = (v ** 1.5) / ((h - l) + 0.1 * atr14)
    poc48 = (typ_p * vol_density).rolling(48).sum() / (vol_density.rolling(48).sum() + 1e-10)
    df['poc'] = poc48

    # Volume MA(20) Confirmation (Condition C5, Bhatti 2026)
    vol_ma20 = v.rolling(20).mean() + 1e-10
    df['vol_ratio'] = v / vol_ma20

    # ADX(14) Directional Trend Strength (Section 7.3 Future Extension, Bhatti 2026)
    up_move = h - h.shift(1)
    dn_move = l.shift(1) - l
    plus_dm  = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    plus_di  = 100.0 * pine_rma(plus_dm, 14) / (atr14 + 1e-10)
    minus_di = 100.0 * pine_rma(minus_dm, 14) / (atr14 + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx14 = pine_rma(dx, 14)
    df['adx'] = adx14

    # --- 12 AI INPUT FEATURES (100% PINE-SCRIPT COMPATIBLE) ---
    # f1: Macro Regime Distance from EMA 200 in ATR units (Eq. 3)
    df['f1'] = ((c - ema200) / c) / atr_pct
    # f2: Intermediate Trend Spread (EMA 50 - EMA 200) in ATR units
    df['f2'] = ((ema50 - ema200) / c) / atr_pct
    # f3: Institutional VWAP Distance in ATR units (Eq. 1 & C2)
    df['f3'] = ((c - vwap_sess) / c) / atr_pct
    # f4: Volume Profile POC Retest Distance in ATR units (POC Repo)
    df['f4'] = ((c - poc48) / c) / atr_pct
    # f5: Pullback Proximity to EMA 50 (C3)
    df['f5'] = ((c - ema50) / c) / atr_pct
    # f6: Rejection Wick & Candle Body Conviction (C4 Pin-Bar / Engulfing Microstructure)
    rng = (h - l) + 1e-10
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    body_signed = c - o
    df['f6'] = (body_signed + 0.5 * (lower_wick - upper_wick)) / rng
    # f7: Volume Confirmation Ratio (C5: V / MA20(V))
    df['f7'] = np.clip(v / vol_ma20, 0.2, 4.0) - 1.0
    # f8: Candle Expansion vs ATR (C6: (H - L) / ATR14)
    df['f8'] = np.clip((h - l) / (atr14 + 1e-10), 0.1, 3.5) - 1.0
    # f9: ADX(14) Trend Strength Normalized + Directional DI Spread
    df['f9'] = ((plus_di - minus_di) / 50.0) * (adx14 / 25.0)
    # f10: Wilder RSI(14) Pullback Oscillator
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['rsi14_raw'] = rsi14
    df['f10'] = (rsi14 - 50.0) / 50.0
    # f11 & f12: Cross-Asset Correlation Filter (Section 7.2 Limitation Fix: USD Proxy & Liquidity Beta)
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4 = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f11'] = eur_ret4 * 100.0
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Boundary Ambiguity Zone Check (Section 3.3: Close must be > 0.08% away from EMA 200)
    df['ema200_dist_pct'] = ((c - ema200).abs() / ema200) * 100.0

    return df

# =========================================================================
# 2. BHATTI (2026) CLOSE-CONDITIONED EMA50/EMA20 TRAIL + POC SIMULATOR
# =========================================================================
def simulate_paper_ai_strategy(
    df_slice, ai_z, max_hold=24,
    z_min=0.15, z_strong=0.65, adx_min=16.0,
    target_r=3.0, final_leg_r=2.0, sl_atr_pad=0.6,
    cooldown=2, fee=0.0001
):
    closes  = df_slice['close'].values
    opens   = df_slice['open'].values
    highs   = df_slice['high'].values
    lows    = df_slice['low'].values
    atrs    = df_slice['atr'].values
    ema20   = df_slice['ema20'].values
    ema50   = df_slice['ema50'].values
    ema200  = df_slice['ema200'].values
    vwap    = df_slice['vwap'].values
    poc     = df_slice['poc'].values
    adx     = df_slice['adx'].values
    f6_rej  = df_slice['f6'].values
    vol_r   = df_slice['vol_ratio'].values
    b_dist  = df_slice['ema200_dist_pct'].values
    dates   = pd.to_datetime(df_slice['datetime']).values

    n = len(closes)
    trades = []
    eq = [0.0]
    i = 2

    # Daily loss limit tracker (Section 5.3: cease trading for the day if 3 losses occur in 1 day)
    cur_day = None
    daily_losses = 0

    while i < n - max_hold - 1:
        dt_obj = pd.Timestamp(dates[i])
        day_str = dt_obj.strftime('%Y-%m-%d')
        if day_str != cur_day:
            cur_day = day_str
            daily_losses = 0

        # Block Sundays, Daily 3-Loss Limit (Section 5.3), Boundary Zone (Section 3.3), and Extreme News Wicks
        bar_range_atr = (highs[i] - lows[i]) / (atrs[i] + 1e-10)
        if dt_obj.dayofweek == 6 or daily_losses >= 3 or b_dist[i] < 0.06 or adx[i] < adx_min or bar_range_atr > 2.4:
            i += 1
            continue

        z = ai_z[i]
        direction = 0
        cap_w = 0.0

        # --- LONG REGIME & PULLBACK/POC REJECTION CONDITIONS (Section 3.4 + POC Repo) ---
        # C1: Close > EMA200 & EMA50 > EMA200
        # C2: Close > VWAP or Close > POC (Institutional Buy Side)
        # C3 & C4: Pullback/Bounce near EMA50/EMA20/POC with bullish rejection candle (close > open & f6_rej > 0.10)
        near_bull_support = (min(lows[i], lows[i-1]) <= max(ema50[i], ema20[i], poc[i]) + 0.35 * atrs[i]) and (closes[i] > ema50[i])
        if (closes[i] > ema200[i]) and (ema50[i] >= ema200[i] * 0.9995) and (closes[i] > vwap[i] or closes[i] > poc[i]):
            if near_bull_support and (closes[i] > opens[i]) and (f6_rej[i] > 0.10) and (z >= z_min):
                direction = 1
                cap_w = 0.70 if z >= z_strong else 0.45

        # --- SHORT REGIME & PULLBACK/POC REJECTION CONDITIONS (Symmetric Inverse) ---
        near_bear_resist = (max(highs[i], highs[i-1]) >= min(ema50[i], ema20[i], poc[i]) - 0.35 * atrs[i]) and (closes[i] < ema50[i])
        if direction == 0 and (closes[i] < ema200[i]) and (ema50[i] <= ema200[i] * 1.0005) and (closes[i] < vwap[i] or closes[i] < poc[i]):
            if near_bear_resist and (closes[i] < opens[i]) and (f6_rej[i] < -0.10) and (z <= -z_min):
                direction = -1
                cap_w = 0.70 if z <= -z_strong else 0.45

        if direction != 0:
            entry_p = closes[i]
            cur_atr = atrs[i]

            # Initial Structural Stop-Loss (Eq. 4, Section 3.6): Low/High of signal bar +/- sl_atr_pad * ATR14
            if direction == 1:
                hard_sl = min(lows[i], lows[i-1]) - sl_atr_pad * cur_atr
                risk_1r = max(0.6 * cur_atr, min(2.0 * cur_atr, entry_p - hard_sl))
                hard_sl = entry_p - risk_1r
                tp_p    = entry_p + target_r * risk_1r
            else:
                hard_sl = max(highs[i], highs[i-1]) + sl_atr_pad * cur_atr
                risk_1r = max(0.6 * cur_atr, min(2.0 * cur_atr, hard_sl - entry_p))
                hard_sl = entry_p + risk_1r
                tp_p    = entry_p - target_r * risk_1r

            exit_p = closes[i + max_hold]
            bars_held = max_hold
            exit_reason = "Time"
            pyramided = False
            entry2 = 0.0
            w2 = 0.0

            # Bar-by-Bar Exit Management (Section 4.1 - 4.4: Close-Only EMA50/EMA20 Trail + 1R Milestone)
            for k in range(1, max_hold + 1):
                idx = i + k
                if direction == 1:
                    # 1. Hard disaster SL or 3R Target
                    if lows[idx] <= hard_sl:
                        exit_p, bars_held, exit_reason = hard_sl, k, "Hard SL"
                        break
                    if highs[idx] >= tp_p:
                        exit_p, bars_held, exit_reason = tp_p, k, "3R Target"
                        break

                    # Current floating R-multiple
                    float_r = (highs[idx] - entry_p) / risk_1r
                    close_r = (closes[idx] - entry_p) / risk_1r

                    # VWAP / 1R Milestone Protection: Once trade hits +1.0R, raise disaster stop to Breakeven!
                    if float_r >= 1.0:
                        hard_sl = max(hard_sl, entry_p + 0.10 * risk_1r)

                    # Pyramid Scale-In on Confirmed Impulse (> +0.6R and AI Z still supportive)
                    if (not pyramided) and close_r >= 0.55 and ai_z[idx] >= z_min:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.30
                        hard_sl = max(hard_sl, entry_p)

                    # Section 4.4 Final-Leg Tightening: Switch close-conditioned trail from EMA50 to EMA20 at >= 2.0R
                    active_ema = ema20[idx] if float_r >= final_leg_r else ema50[idx]

                    # Section 4.2 CORE RULE: Exit ONLY if 15m candle CLOSES below active EMA (ignores intrabar wicks!)
                    if k >= 2 and closes[idx] < active_ema:
                        exit_p, bars_held, exit_reason = closes[idx], k, "EMA Close Trail"
                        break
                else:
                    if highs[idx] >= hard_sl:
                        exit_p, bars_held, exit_reason = hard_sl, k, "Hard SL"
                        break
                    if lows[idx] <= tp_p:
                        exit_p, bars_held, exit_reason = tp_p, k, "3R Target"
                        break

                    float_r = (entry_p - lows[idx]) / risk_1r
                    close_r = (entry_p - closes[idx]) / risk_1r

                    if float_r >= 1.0:
                        hard_sl = min(hard_sl, entry_p - 0.10 * risk_1r)

                    if (not pyramided) and close_r >= 0.55 and ai_z[idx] <= -z_min:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.30
                        hard_sl = min(hard_sl, entry_p)

                    active_ema = ema20[idx] if float_r >= final_leg_r else ema50[idx]

                    if k >= 2 and closes[idx] > active_ema:
                        exit_p, bars_held, exit_reason = closes[idx], k, "EMA Close Trail"
                        break

            ret1 = cap_w * (direction * ((exit_p - entry_p) / entry_p) - fee)
            ret2 = w2 * (direction * ((exit_p - entry2) / entry2) - fee) if pyramided else 0.0
            net_ret = ret1 + ret2
            r_mult  = direction * ((exit_p - entry_p) / risk_1r) - 0.10

            if net_ret < 0:
                daily_losses += 1

            trades.append({
                'date': day_str,
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'reason': exit_reason,
                'r_mult': r_mult,
                'ret': net_ret
            })
            eq.append(eq[-1] + net_ret * 100.0)
            i += bars_held + (cooldown if net_ret < 0 else 1)
        else:
            i += 1

    eq_arr = np.array(eq)
    dd = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr)) if len(eq_arr) > 1 else 0.0
    return pd.DataFrame(trades), dd

def print_period_stats(tr_df, dd, name):
    if len(tr_df) == 0:
        print(f"{name:<34} | 0 trades")
        return
    n_t = len(tr_df)
    wr = (tr_df['ret'] > 0).mean() * 100.0
    pnl = tr_df['ret'].sum() * 100.0
    gw = tr_df[tr_df['ret'] > 0]['ret'].sum()
    gl = abs(tr_df[tr_df['ret'] < 0]['ret'].sum()) + 1e-10
    pf = gw / gl
    exp_r = tr_df['r_mult'].mean()
    print(f"{name:<34} | Trades: {n_t:<4} | Win%: {wr:<5.1f}% | Expectancy: {exp_r:+.2f}R | PF: {pf:<5.2f} | MaxDD: {dd:<4.2f}% | Net PnL: {pnl:+.2f}%")

def main():
    print("=== PHASE 14: BHATTI (2026) VWAP-EMA CLOSE TRAIL + VOLUME POC RETEST + AI ENGINE ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_paper_poc_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 13)]

    # Target Label: Does a 12-bar (3-hour) structural move achieve positive risk-adjusted continuation?
    HORIZON = 12
    f_high = df['high'].rolling(HORIZON).max().shift(-HORIZON)
    f_low  = df['low'].rolling(HORIZON).min().shift(-HORIZON)
    up_ex  = (f_high - df['close']) / (df['atr'] + 1e-10)
    dn_ex  = (df['close'] - f_low) / (df['atr'] + 1e-10)

    # Label 1 if upside reaches >= 1.5 ATR and beats downside by 1.3x; Label 0 for symmetric bearish move
    df['target'] = np.where((up_ex >= 1.5) & (up_ex > dn_ex * 1.3), 1,
                   np.where((dn_ex >= 1.5) & (dn_ex > up_ex * 1.3), 0, -1))
    df = df.dropna().reset_index(drop=True)

    # STRICT 3-WAY SPLIT (Train <= Aug 31, 2026 | Validation: Sep 1 - Nov 8 | Blind Test: Nov 9 - Nov 15)
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    train_df = df[df['datetime'] <= TRAIN_END].copy()
    train_clean = train_df[train_df['target'] != -1].copy()

    # Exact 50/50 Bull/Bear Class Balance so AI has zero directional bias
    pos_df = train_clean[train_clean['target'] == 1]
    neg_df = train_clean[train_clean['target'] == 0]
    m_len  = min(len(pos_df), len(neg_df))
    bal_train = pd.concat([pos_df.iloc[-m_len:], neg_df.iloc[-m_len:]]).sort_values('datetime')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(bal_train[feature_cols])
    y_train   = bal_train['target']

    print(f"1. Trained AI Ensemble strictly <= Aug 31, 2026 ({len(bal_train)} 50/50 balanced institutional moves)")

    lr  = LogisticRegression(C=0.08, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', alpha=0.4, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    # Rolling 200-Bar Z-Score Calibration across dataset (Zero Regime Drift!)
    X_all_s = scaler.transform(df[feature_cols])
    df['raw_p'] = 0.55 * lr.predict_proba(X_all_s)[:, 1] + 0.45 * mlp.predict_proba(X_all_s)[:, 1]
    r_mean = df['raw_p'].rolling(200).mean()
    r_std  = df['raw_p'].rolling(200).std(ddof=0) + 1e-10
    df['ai_z'] = (df['raw_p'] - r_mean) / r_std

    val_df      = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df     = df[df['datetime'] > VAL_END].copy()
    full_oos_df = df[df['datetime'] > TRAIN_END].copy()

    # Tune structural parameters strictly on Validation Set (Sep 1 - Nov 8)
    best_cfg = None
    best_score = -9999.0

    for z_min in [0.05, 0.15, 0.25]:
        for adx_m in [14.0, 16.0, 18.0]:
            for sl_pad in [0.5, 0.7]:
                v_tr, v_dd = simulate_paper_ai_strategy(
                    val_df, val_df['ai_z'].values, max_hold=24,
                    z_min=z_min, z_strong=z_min + 0.50, adx_min=adx_m,
                    target_r=3.0, final_leg_r=2.0, sl_atr_pad=sl_pad
                )
                if len(v_tr) >= 35:
                    v_pnl = v_tr['ret'].sum() * 100.0
                    gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                    gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                    v_pf = gw / gl
                    score = v_pnl * v_pf
                    if score > best_score:
                        best_score = score
                        best_cfg = {'z_min': z_min, 'adx': adx_m, 'sl_pad': sl_pad}

    print(f"2. Validation-Selected Settings (from Sep 1 - Nov 8): AI_Z_Min={best_cfg['z_min']} | ADX_Min={best_cfg['adx']} | SL_Pad={best_cfg['sl_pad']}x ATR\n")

    val_tr, val_dd   = simulate_paper_ai_strategy(val_df, val_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.5, adx_min=best_cfg['adx'], sl_atr_pad=best_cfg['sl_pad'])
    test_tr, test_dd = simulate_paper_ai_strategy(test_df, test_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.5, adx_min=best_cfg['adx'], sl_atr_pad=best_cfg['sl_pad'])
    full_tr, full_dd = simulate_paper_ai_strategy(full_oos_df, full_oos_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.5, adx_min=best_cfg['adx'], sl_atr_pad=best_cfg['sl_pad'])

    print("=== OUT-OF-SAMPLE PERFORMANCE (BHATTI 2026 CLOSE-TRAIL + POC RETEST + AI) ===")
    print("-" * 108)
    print_period_stats(val_tr,  val_dd,  "1. Sep 01 - Nov 08 (Validation)")
    print_period_stats(test_tr, test_dd, "2. Nov 09 - Nov 15 (Blind Test)")
    print_period_stats(full_tr, full_dd, "3. Sep 01 - Nov 15 (Full OOS Chart)")
    print("-" * 108)

    if len(full_tr) > 0:
        print("\nExit Attribution Breakdown (Full Sep 1 - Nov 15 OOS):")
        print(full_tr.groupby('reason')['ret'].agg(Count='count', WinRate=lambda x: f"{(x>0).mean()*100:.1f}%", NetPnL=lambda x: f"{x.sum()*100:+.2f}%"))

    # =========================================================================
    # 3. COMPILE PINE SCRIPT V5 (AI_Gold_VWAP_POC_Paper_v8.pine)
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold VWAP-POC & Close-EMA Trail v8 (Bhatti 2026 + POC Hybrid)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=45, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005)

// =========================================================================
// 1. EVALUATION WINDOW (DEFAULT: FULL CHART SEP 1 - NOV 16!)
// =========================================================================
use_date_filter = input.bool(false, "Restrict to Specific Date Window (Uncheck for Full Chart)", group="1. Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Window")

// =========================================================================
// 2. BHATTI (2026) + POC RETEST + AI PARAMETERS
// =========================================================================
z_min_val    = input.float({best_cfg['z_min']:.2f}, "Min AI Conviction Z-Score", minval=-0.20, maxval=1.50, step=0.05, group="2. AI & Regime Filters")
z_strong_val = input.float({best_cfg['z_min']+0.50:.2f}, "Tier-2 Strong AI Conviction Z-Score", minval=0.20, maxval=2.00, step=0.05, group="2. AI & Regime Filters")
adx_min_val  = input.float({best_cfg['adx']:.1f}, "Min ADX(14) Trend Strength (Paper Sec 7.3)", minval=10.0, maxval=30.0, step=1.0, group="2. AI & Regime Filters")
sl_pad_atr   = input.float({best_cfg['sl_pad']:.1f}, "Initial Stop Buffer Beyond Signal Low/High (ATR)", minval=0.2, maxval=1.5, step=0.1, group="3. Paper Exit Architecture")
target_r_val = input.float(3.0, "Primary Profit Target (R-Multiple, Paper Sec 4.1)", minval=1.5, maxval=5.0, step=0.5, group="3. Paper Exit Architecture")
final_leg_r  = input.float(2.0, "Switch Trail from EMA50 to EMA20 at R-Multiple (Sec 4.4)", minval=1.0, maxval=3.0, step=0.25, group="3. Paper Exit Architecture")
max_hold_b   = input.int(24, "Max Holding Horizon (Bars)", minval=6, maxval=48, group="3. Paper Exit Architecture")
tier1_pct    = input.float(45.0, "Tier-1 Standard Allocation (%)", minval=10.0, maxval=100.0, step=5.0, group="4. Capital & Pyramiding")
tier2_pct    = input.float(70.0, "Tier-2 High-Conviction Allocation (%)", minval=20.0, maxval=100.0, step=5.0, group="4. Capital & Pyramiding")
enable_pyr   = input.bool(true, "Enable +30% Scale-In at +0.55R Impulse", group="4. Capital & Pyramiding")

// =========================================================================
// 3. INDICATORS: EMA(200, 50, 20), VWAP, VOLUME POC, ADX(14) & CROSS-ASSET
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

ema20  = ta.ema(close, 20)
ema50  = ta.ema(close, 50)
ema200 = ta.ema(close, 200)

// Institutional Rolling Session VWAP (Eq. 1)
typ_p  = (high + low + close) / 3.0
vwap32 = math.sum(typ_p * volume, 32) / (math.sum(volume, 32) + 1e-10)

// Rolling 48-Bar Volume Profile Point of Control (POC) Node
vol_dens = math.pow(volume, 1.5) / ((high - low) + 0.1 * atr14)
poc48    = math.sum(typ_p * vol_dens, 48) / (math.sum(vol_dens, 48) + 1e-10)

vol_ma20  = ta.sma(volume, 20) + 1e-10
vol_ratio = volume / vol_ma20

// ADX(14) Trend Strength
up_m = high - high[1]
dn_m = low[1] - low
p_dm = (up_m > dn_m and up_m > 0) ? up_m : 0.0
m_dm = (dn_m > up_m and dn_m > 0) ? dn_m : 0.0
p_di = 100.0 * pine_rma(p_dm, 14) / (atr14 + 1e-10)
m_di = 100.0 * pine_rma(m_dm, 14) / (atr14 + 1e-10)
dx   = 100.0 * math.abs(p_di - m_di) / (p_di + m_di + 1e-10)
adx14 = pine_rma(dx, 14)

// --- 12 AI FEATURES ---
f1 = ((close - ema200) / close) / atr_pct
f2 = ((ema50 - ema200) / close) / atr_pct
f3 = ((close - vwap32) / close) / atr_pct
f4 = ((close - poc48) / close) / atr_pct
f5 = ((close - ema50) / close) / atr_pct

rng = (high - low) + 1e-10
l_wick = math.min(open, close) - low
u_wick = high - math.max(open, close)
f6 = ((close - open) + 0.5 * (l_wick - u_wick)) / rng

f7 = math.min(4.0, math.max(0.2, vol_ratio)) - 1.0
f8 = math.min(3.5, math.max(0.1, (high - low) / (atr14 + 1e-10))) - 1.0
f9 = ((p_di - m_di) / 50.0) * (adx14 / 25.0)

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f10 = (rsi14 - 50.0) / 50.0

eur_ret4  = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
btc_ret4  = (btc_close - nz(btc_close[4], btc_close[1])) / nz(btc_close[4], btc_close[1])
f11 = nz(eur_ret4, 0.0) * 100.0
f12 = nz(eur_ret4 - gold_ret4 + 0.15 * btc_ret4, 0.0) * 100.0

// =========================================================================
// 4. FROZEN AI INFERENCE + ROLLING 200-BAR Z-SCORE
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
        pine_code += f"z_n{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z_n{j}) - 1.0) / (math.exp(2.0 * z_n{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

    pine_code += f"""raw_p = 0.55 * p_lr + 0.45 * p_mlp
roll_mean = ta.sma(raw_p, 200)
roll_std  = ta.stdev(raw_p, 200) + 1e-10
ai_z      = (raw_p - roll_mean) / roll_std
prob_disp = 1.0 / (1.0 + math.exp(-ai_z * 1.15))

// =========================================================================
// 5. BHATTI (2026) + POC ENTRY & CLOSE-CONDITIONED EMA50/20 EXIT ARCHITECTURE
// =========================================================================
calc_qty(pct_eq) =>
    math.max(0.001, (strategy.equity * (pct_eq / 100.0)) / close)

// Daily 3-Loss Circuit Breaker (Section 5.3)
var int cur_day = -1
var int daily_losses = 0
if dayofmonth(time) != cur_day
    cur_day := dayofmonth(time)
    daily_losses := 0

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        daily_losses += 1

ema200_dist_pct = (math.abs(close - ema200) / ema200) * 100.0
bar_range_atr   = (high - low) / (atr14 + 1e-10)
filters_ok      = in_date_window and (daily_losses < 3) and (ema200_dist_pct >= 0.06) and (adx14 >= adx_min_val) and (bar_range_atr <= 2.4)

// Pullback / Retest of EMA50, EMA20, or Volume POC
bull_zone = (math.min(low, low[1]) <= math.max(ema50, math.max(ema20, poc48)) + 0.35 * atr14) and (close > ema50)
bear_zone = (math.max(high, high[1]) >= math.min(ema50, math.min(ema20, poc48)) - 0.35 * atr14) and (close < ema50)

bull_signal = filters_ok and (close > ema200) and (ema50 >= ema200 * 0.9995) and (close > vwap32 or close > poc48) and bull_zone and (close > open) and (f6 > 0.10) and (ai_z >= z_min_val)
bear_signal = filters_ok and (close < ema200) and (ema50 <= ema200 * 1.0005) and (close < vwap32 or close < poc48) and bear_zone and (close < open) and (f6 < -0.10) and (ai_z <= -z_min_val)

bull_pct = ai_z >= z_strong_val ? tier2_pct : tier1_pct
bear_pct = ai_z <= -z_strong_val ? tier2_pct : tier1_pct

var int   entry_bar  = na
var int   last_loss  = -999
var float entry_p    = na
var float risk_1r    = na
var float hard_sl    = na
var float tp_level   = na
var bool  scaled_in  = false
var float active_pct = 0.0

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_loss := bar_index

cooldown_ok = (bar_index - last_loss) >= 2

if barstate.isconfirmed
    if strategy.position_size == 0 and cooldown_ok
        if bull_signal
            raw_sl   = math.min(low, low[1]) - sl_pad_atr * atr14
            r_dist   = math.max(0.6 * atr14, math.min(2.0 * atr14, close - raw_sl))
            strategy.entry("AI Long", strategy.long, qty=calc_qty(bull_pct), comment="L (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_disp*100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            risk_1r    := r_dist
            hard_sl    := close - r_dist
            tp_level   := close + target_r_val * r_dist
            scaled_in  := false
            active_pct := bull_pct
        else if bear_signal
            raw_sl   = math.max(high, high[1]) + sl_pad_atr * atr14
            r_dist   = math.max(0.6 * atr14, math.min(2.0 * atr14, raw_sl - close))
            strategy.entry("AI Short", strategy.short, qty=calc_qty(bear_pct), comment="S (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round((1.0-prob_disp)*100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            risk_1r    := r_dist
            hard_sl    := close + r_dist
            tp_level   := close - target_r_val * r_dist
            scaled_in  := false
            active_pct := bear_pct

// --- CLOSE-CONDITIONED EMA50 / EMA20 TRAILING STOP & MILESTONE ENGINE (Section 4.1 - 4.4) ---
if strategy.position_size > 0
    float_r = (high - entry_p) / (risk_1r + 1e-10)
    close_r = (close - entry_p) / (risk_1r + 1e-10)

    // 1R Milestone: Protect position at Breakeven+0.1R once +1R is reached
    if float_r >= 1.0
        hard_sl := math.max(hard_sl, entry_p + 0.10 * risk_1r)

    // Pyramid Scale-In at +0.55R if AI still confirms
    if barstate.isconfirmed and enable_pyr and not scaled_in and close_r >= 0.55 and ai_z >= z_min_val
        strategy.entry("AI Long Scale", strategy.long, qty=calc_qty(30.0), comment="+Pyramid (30%)")
        scaled_in  := true
        active_pct := active_pct + 30.0
        hard_sl    := math.max(hard_sl, entry_p)

    strategy.exit("Hard Exit Long", from_entry="AI Long", limit=tp_level, stop=hard_sl)
    strategy.exit("Hard Exit Long Scale", from_entry="AI Long Scale", limit=tp_level, stop=hard_sl)

    // Section 4.2 & 4.4: Close-Only EMA50 Trail (Switches to EMA20 at >= 2.0R)
    active_trail_ema = float_r >= final_leg_r ? ema20 : ema50
    if barstate.isconfirmed and (bar_index - entry_bar) >= 2 and close < active_trail_ema
        strategy.close_all(comment="EMA Close Trail")
    else if (bar_index - entry_bar) >= max_hold_b
        strategy.close_all(comment="Time Horizon")

else if strategy.position_size < 0
    float_r = (entry_p - low) / (risk_1r + 1e-10)
    close_r = (entry_p - close) / (risk_1r + 1e-10)

    if float_r >= 1.0
        hard_sl := math.min(hard_sl, entry_p - 0.10 * risk_1r)

    if barstate.isconfirmed and enable_pyr and not scaled_in and close_r >= 0.55 and ai_z <= -z_min_val
        strategy.entry("AI Short Scale", strategy.short, qty=calc_qty(30.0), comment="+Pyramid (30%)")
        scaled_in  := true
        active_pct := active_pct + 30.0
        hard_sl    := math.min(hard_sl, entry_p)

    strategy.exit("Hard Exit Short", from_entry="AI Short", limit=tp_level, stop=hard_sl)
    strategy.exit("Hard Exit Short Scale", from_entry="AI Short Scale", limit=tp_level, stop=hard_sl)

    active_trail_ema = float_r >= final_leg_r ? ema20 : ema50
    if barstate.isconfirmed and (bar_index - entry_bar) >= 2 and close > active_trail_ema
        strategy.close_all(comment="EMA Close Trail")
    else if (bar_index - entry_bar) >= max_hold_b
        strategy.close_all(comment="Time Horizon")

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. INSTITUTIONAL OVERLAYS & PAPER/POC HUD
// =========================================================================
plot(ema200, "EMA 200 (Regime Classifier)", color=color.new(color.yellow, 10), linewidth=2)
plot(ema50,  "EMA 50 (Pullback & Close-Trail)", color=color.new(color.aqua, 20), linewidth=2)
plot(vwap32, "Session VWAP", color=color.new(color.fuchsia, 30), linewidth=1)
plot(poc48,  "Rolling Volume POC", color=color.new(color.orange, 25), style=plot.style_circles, linewidth=2)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_disp >= 0.5 ? math.round(prob_disp * 100, 1) : math.round((1.0 - prob_disp) * 100, 1)
    reg_txt  = close > ema200 ? "BULLISH (>EMA200)" : "BEARISH (<EMA200)"
    sig_txt  = bull_signal ? "BUY PULLBACK" : bear_signal ? "SHORT PULLBACK" : "WAIT FOR RETEST"
    sig_col  = bull_signal ? color.green : bear_signal ? color.red : color.gray

    table.cell(hud, 0, 0, "Architecture", text_color=color.white)
    table.cell(hud, 1, 0, "Bhatti(2026) + POC + AI", text_color=color.yellow)
    table.cell(hud, 0, 1, "Macro Regime (C1)", text_color=color.white)
    table.cell(hud, 1, 1, reg_txt, text_color=close > ema200 ? color.lime : color.red)
    table.cell(hud, 0, 2, "VWAP / POC Side (C2)", text_color=color.white)
    table.cell(hud, 1, 2, close > vwap32 ? "ABOVE VWAP (BUY)" : "BELOW VWAP (SELL)", text_color=close > vwap32 ? color.lime : color.orange)
    table.cell(hud, 0, 3, "AI Retest Conviction", text_color=color.white)
    table.cell(hud, 1, 3, str.tostring(conf_pct) + "% (ADX: " + str.tostring(math.round(adx14, 1)) + ")", text_color=color.aqua)
    table.cell(hud, 0, 4, "Exit Mechanism", text_color=color.white)
    table.cell(hud, 1, 4, "Close-Only EMA50/20 Trail", text_color=color.lime)
    table.cell(hud, 0, 5, "Live Action", text_color=color.white)
    table.cell(hud, 1, 5, sig_txt, bgcolor=sig_col, text_color=color.white)
"""

    with open("AI_Gold_VWAP_POC_Paper_v8.pine", "w") as f:
        f.write(pine_code)

    print("\n3. Generated 'AI_Gold_VWAP_POC_Paper_v8.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()