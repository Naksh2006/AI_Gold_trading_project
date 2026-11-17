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
        raise FileNotFoundError(f"Missing {cache_file}. Please ensure cached data exists.")
    df = pd.read_csv(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. ICT / SMC LIQUIDITY SWEEP, FVG, ORDER BLOCK & PREMIUM/DISCOUNT FEATURES
# =========================================================================
def build_smc_liquidity_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c, btc_c = df['eur_close'], df['btc_close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df['atr'] = atr14

    # --- TOP-DOWN TIMEFRAME ALIGNMENT (HTF Trend / ITF Range / LTF Sweep) ---
    # 1. HTF Trend (H4 / Daily Draw on Liquidity: EMA 64 & EMA 192 on 15m)
    ema64  = c.ewm(span=64, adjust=False).mean()
    ema192 = c.ewm(span=192, adjust=False).mean()
    df['htf_trend'] = ((ema64 - ema192) / c) / atr_pct

    # 2. ITF Dealing Range (48-bar / 12-Hour High, Low & 50% Equilibrium)
    range_h = h.shift(1).rolling(32).max()
    range_l = l.shift(1).rolling(32).min()
    range_eq = (range_h + range_l) / 2.0
    range_size = (range_h - range_l) + 1e-10
    df['range_h']  = range_h
    df['range_l']  = range_l
    df['range_eq'] = range_eq

    # Local Liquidity Pools (10-bar Swing Highs & Lows where retail stops sit!)
    liq_high_10 = h.shift(1).rolling(10).max()
    liq_low_10  = l.shift(1).rolling(10).min()
    df['liq_h'] = liq_high_10
    df['liq_l'] = liq_low_10

    # --- MECHANICAL LIQUIDITY SWEEP (FAKE BREAKOUT) & MODEL 2 RELOAD DETECTION ---
    # Model 1 Bullish Sweep: Price wicks below 10-bar low (or prior 2-bar sweep), snaps back & closes inside!
    swept_low_now  = (l < liq_low_10) & (c > liq_low_10) & (c > o)
    swept_low_prev = (l.shift(1) < liq_low_10.shift(1)) & (c > liq_low_10.shift(1)) & (c > o) & (c > c.shift(1))

    # Model 1 Bearish Sweep: Price wicks above 10-bar high, snaps back & closes inside!
    swept_high_now  = (h > liq_high_10) & (c < liq_high_10) & (c < o)
    swept_high_prev = (h.shift(1) > liq_high_10.shift(1)) & (c < liq_high_10.shift(1)) & (c < o) & (c < c.shift(1))

    # Fair Value Gap (FVG) & Order Block Displacement
    bull_fvg = (l > h.shift(2)) & (c.shift(1) > o.shift(1))
    bear_fvg = (h < l.shift(2)) & (c.shift(1) < o.shift(1))
    recent_bull_fvg = bull_fvg.rolling(6).max() > 0
    recent_bear_fvg = bear_fvg.rolling(6).max() > 0

    # Premium (> 0.50) vs Discount (< 0.50) Position in Dealing Range
    pd_pos = (c - range_l) / range_size # 0.0 = Range Low, 0.5 = 50% EQ, 1.0 = Range High
    df['pd_pos'] = pd_pos

    # Model 2 Reload Zone (61.8% - 80% retracement into Discount for Longs, Premium for Shorts with FVG/OB)
    bull_reload = (pd_pos <= 0.42) & (pd_pos >= 0.12) & recent_bull_fvg & (c > o) & ((c - l) > 0.5 * (h - l))
    bear_reload = (pd_pos >= 0.58) & (pd_pos <= 0.88) & recent_bear_fvg & (c < o) & ((h - c) > 0.5 * (h - l))

    df['bull_setup'] = (swept_low_now | swept_low_prev | bull_reload).astype(int)
    df['bear_setup'] = (swept_high_now | swept_high_prev | bear_reload).astype(int)

    # --- 12 AI INPUT FEATURES FOR SCORING SWEEP & ORDER-BLOCK QUALITY ---
    rng = (h - l) + 1e-10
    l_wick = np.minimum(o, c) - l
    u_wick = h - np.maximum(o, c)

    # f1: Net Manipulation Wick Rejection Asymmetry
    df['f1'] = (l_wick - u_wick) / rng
    # f2: Candle Closing Conviction (Snap-back strength)
    df['f2'] = (c - o) / rng
    # f3: Premium / Discount Position centered at 50% Equilibrium [-1.0 = Deep Discount, +1.0 = Deep Premium]
    df['f3'] = (pd_pos - 0.50) * 2.0
    # f4: Distance from 10-Bar Liquidity Pool in ATR units
    df['f4'] = ((c - ((liq_high_10 + liq_low_10) / 2.0)) / (atr14 + 1e-10))
    # f5: Fair Value Gap (FVG) 3-Bar Displacement Impulse in ATR units
    df['f5'] = (c - o.shift(2)) / (atr14 + 1e-10)
    # f6: HTF Trend Alignment (EMA 64 vs EMA 192)
    df['f6'] = df['htf_trend']
    # f7: Volume Spike at Liquidity Raid (Confirms Stop-Hunt fuel collected!)
    vol_ma20 = v.rolling(20).mean() + 1e-10
    df['f7'] = np.clip(v / vol_ma20, 0.2, 4.0) - 1.0
    # f8: Wilder RSI(14) Exhaustion at Sweep Extreme
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df['f8'] = (rsi14 - 50.0) / 50.0
    # f9: Institutional VWAP Stretch
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df['f9'] = ((c - vwap24) / c) / atr_pct
    # f10: Range Expansion Ratio (Dealing Range Width in ATR units -> ensures enough room for 2R+ target!)
    df['f10'] = (range_size / (atr14 + 1e-10)) / 5.0 - 1.0
    # f11 & f12: Cross-Asset US Dollar Proxy (EURUSDT) & Liquidity Beta (BTCUSDT)
    eur_ret4  = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4  = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df['f11'] = eur_ret4 * 100.0
    df['f12'] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    return df

# =========================================================================
# 2. MODEL 1 (50% EQ) & MODEL 2 (OPPOSING RANGE + RELOAD PYRAMID) SIMULATOR
# =========================================================================
def simulate_smc_sweep(
    df_slice, ai_z, max_hold=18,
    z_min=-0.05, z_strong=0.45,
    sl_atr_pad=0.35, min_rr_target=2.0,
    cooldown=2, fee=0.0001
):
    closes    = df_slice['close'].values
    opens     = df_slice['open'].values
    highs     = df_slice['high'].values
    lows      = df_slice['low'].values
    atrs      = df_slice['atr'].values
    range_h   = df_slice['range_h'].values
    range_l   = df_slice['range_l'].values
    range_eq  = df_slice['range_eq'].values
    pd_pos    = df_slice['pd_pos'].values
    htf_tr    = df_slice['htf_trend'].values
    bull_s    = df_slice['bull_setup'].values
    bear_s    = df_slice['bear_setup'].values
    dates     = pd.to_datetime(df_slice['datetime']).values

    n = len(closes)
    trades = []
    eq = [0.0]
    i = 2

    while i < n - max_hold - 1:
        dt_obj = pd.Timestamp(dates[i])
        if dt_obj.dayofweek == 6:
            i += 1
            continue

        z = ai_z[i]
        direction = 0
        cap_w = 0.0

        # --- BULLISH SWEEP / RELOAD IN DISCOUNT (Buy Low After Stop Hunt!) ---
        # Must be in Discount (pd_pos < 0.52) and aligned with HTF or strong AI reversal conviction
        if bull_s[i] == 1 and pd_pos[i] < 0.52 and (htf_tr[i] > -0.8 or z >= z_strong) and z >= z_min:
            direction = 1
            cap_w = 0.65 if z >= z_strong else 0.40

        # --- BEARISH SWEEP / RELOAD IN PREMIUM (Sell High After Stop Hunt!) ---
        # Must be in Premium (pd_pos > 0.48) and aligned with HTF or strong AI reversal conviction
        elif bear_s[i] == 1 and pd_pos[i] > 0.48 and (htf_tr[i] < 0.8 or z <= -z_strong) and z <= -z_min:
            direction = -1
            cap_w = 0.65 if z <= -z_strong else 0.40

        if direction != 0:
            entry_p = closes[i]
            cur_atr = atrs[i]

            if direction == 1:
                # Invalidation: Just below the manipulation sweep low!
                sweep_low = min(lows[i], lows[i-1])
                raw_risk  = (entry_p - sweep_low) + sl_atr_pad * cur_atr
                risk_1r   = max(0.55 * cur_atr, min(1.50 * cur_atr, raw_risk))
                sl_price  = entry_p - risk_1r

                # Model 1 Target = 50% Equilibrium of Range (or +1.1R minimum)
                tp1_eq    = max(range_eq[i], entry_p + 1.1 * risk_1r)
                # Model 2 Target = Full Opposing Range High (or +2.2R minimum)
                tp2_full  = max(range_h[i], entry_p + min_rr_target * risk_1r)
            else:
                sweep_high = max(highs[i], highs[i-1])
                raw_risk   = (sweep_high - entry_p) + sl_atr_pad * cur_atr
                risk_1r    = max(0.55 * cur_atr, min(1.50 * cur_atr, raw_risk))
                sl_price   = entry_p + risk_1r

                tp1_eq     = min(range_eq[i], entry_p - 1.1 * risk_1r)
                tp2_full   = min(range_l[i], entry_p - min_rr_target * risk_1r)

            hit_tp1 = False
            pyramided = False
            entry2 = 0.0
            w2 = 0.0
            exit_p = closes[i + max_hold]
            bars_held = max_hold
            outcome_tag = "Time Exit"

            for k in range(1, max_hold + 1):
                idx = i + k
                if direction == 1:
                    # 1. Check Invalidation Stop
                    if lows[idx] <= sl_price:
                        exit_p, bars_held = sl_price, k
                        outcome_tag = "Model1+BE Win" if hit_tp1 else "Stop Invalidation"
                        break
                    # 2. Check Model 2 Full Opposing Range Target!
                    if highs[idx] >= tp2_full:
                        exit_p, bars_held = tp2_full, k
                        outcome_tag = "Model2 Full Range Win"
                        break
                    # 3. Check Model 1 (50% Equilibrium Objective): Lock in +0.5R floor once 50% EQ is hit!
                    if (not hit_tp1) and highs[idx] >= tp1_eq:
                        hit_tp1 = True
                        sl_price = max(sl_price, entry_p + 0.45 * risk_1r)
                    # 4. Model 2 Reload Zone Pyramid: Scale-in if price holds above sweep low and pushes
                    if (not pyramided) and closes[idx] >= (entry_p + 0.40 * risk_1r) and ai_z[idx] >= z_min:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = max(sl_price, entry_p)
                else:
                    if highs[idx] >= sl_price:
                        exit_p, bars_held = sl_price, k
                        outcome_tag = "Model1+BE Win" if hit_tp1 else "Stop Invalidation"
                        break
                    if lows[idx] <= tp2_full:
                        exit_p, bars_held = tp2_full, k
                        outcome_tag = "Model2 Full Range Win"
                        break
                    if (not hit_tp1) and lows[idx] <= tp1_eq:
                        hit_tp1 = True
                        sl_price = min(sl_price, entry_p - 0.45 * risk_1r)
                    if (not pyramided) and closes[idx] <= (entry_p - 0.40 * risk_1r) and ai_z[idx] <= -z_min:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = min(sl_price, entry_p)

            # Blended return: 50% exited at Model 1 (50% EQ) + 50% held for Model 2 (Full Opposing Range)
            if hit_tp1 and exit_p != tp2_full:
                blended_exit = 0.50 * tp1_eq + 0.50 * exit_p
            else:
                blended_exit = exit_p

            ret1 = cap_w * (direction * ((blended_exit - entry_p) / entry_p) - fee)
            ret2 = w2 * (direction * ((blended_exit - entry2) / entry2) - fee) if pyramided else 0.0
            net_ret = ret1 + ret2
            r_mult  = direction * ((blended_exit - entry_p) / risk_1r) - 0.08

            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'outcome': outcome_tag,
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

def print_stats(tr_df, dd, label):
    if len(tr_df) == 0:
        print(f"{label:<34} | 0 trades")
        return
    n_t = len(tr_df)
    wr  = (tr_df['ret'] > 0).mean() * 100.0
    pnl = tr_df['ret'].sum() * 100.0
    gw  = tr_df[tr_df['ret'] > 0]['ret'].sum()
    gl  = abs(tr_df[tr_df['ret'] < 0]['ret'].sum()) + 1e-10
    pf  = gw / gl
    avg_win_r = tr_df[tr_df['r_mult'] > 0]['r_mult'].mean() if (tr_df['r_mult'] > 0).any() else 0.0
    print(f"{label:<34} | Trades: {n_t:<4} | Win%: {wr:<5.1f}% | AvgWin: +{avg_win_r:.2f}R | PF: {pf:<5.2f} | MaxDD: {dd:<4.2f}% | Net PnL: {pnl:+.2f}%")

def main():
    print("=== PHASE 15: ICT/SMC LIQUIDITY SWEEP (FAKE BREAKOUT) + MODEL 1 (50% EQ) & MODEL 2 ENGINE ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_smc_liquidity_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 13)]

    # Target: After a candle, does price snap higher or lower over the next 10 bars (2.5 hours)?
    HORIZON = 10
    f_high = df['high'].rolling(HORIZON).max().shift(-HORIZON)
    f_low  = df['low'].rolling(HORIZON).min().shift(-HORIZON)
    up_ex  = (f_high - df['close']) / (df['atr'] + 1e-10)
    dn_ex  = (df['close'] - f_low) / (df['atr'] + 1e-10)

    df['target'] = np.where((up_ex >= 1.2) & (up_ex > dn_ex * 1.25), 1,
                   np.where((dn_ex >= 1.2) & (dn_ex > up_ex * 1.25), 0, -1))
    df = df.dropna().reset_index(drop=True)

    # STRICT 3-WAY CHRONOLOGICAL SPLIT (Train <= Aug 31 | Val: Sep 1 - Nov 8 | Blind Test: Nov 9 - Nov 15)
    TRAIN_END = pd.Timestamp("2026-08-31 23:59:59")
    VAL_END   = pd.Timestamp("2026-11-08 23:59:59")

    # Train specifically on candidate Sweep / Reload setups <= Aug 31, 2026!
    train_df = df[(df['datetime'] <= TRAIN_END) & (df['target'] != -1)].copy()
    pos_df = train_df[train_df['target'] == 1]
    neg_df = train_df[train_df['target'] == 0]
    m_len  = min(len(pos_df), len(neg_df))
    bal_train = pd.concat([pos_df.iloc[-m_len:], neg_df.iloc[-m_len:]]).sort_values('datetime')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(bal_train[feature_cols])
    y_train   = bal_train['target']

    print(f"1. Trained AI Liquidity-Sweep Scorer <= Aug 31, 2026 ({len(bal_train)} balanced setups)")

    lr  = LogisticRegression(C=0.1, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh', alpha=0.35, max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    mlp.fit(X_train_s, y_train)

    X_all_s = scaler.transform(df[feature_cols])
    df['raw_p'] = 0.55 * lr.predict_proba(X_all_s)[:, 1] + 0.45 * mlp.predict_proba(X_all_s)[:, 1]
    r_mean = df['raw_p'].rolling(200).mean()
    r_std  = df['raw_p'].rolling(200).std(ddof=0) + 1e-10
    df['ai_z'] = (df['raw_p'] - r_mean) / r_std

    val_df      = df[(df['datetime'] > TRAIN_END) & (df['datetime'] <= VAL_END)].copy()
    test_df     = df[df['datetime'] > VAL_END].copy()
    full_oos_df = df[df['datetime'] > TRAIN_END].copy()

    # Optimize strictly on Validation Set (Sep 1 - Nov 8)
    best_cfg = None
    best_score = -9999.0

    for z_m in [-0.15, -0.05, 0.05, 0.15]:
        for rr_m in [1.8, 2.0, 2.3]:
            for sl_pad in [0.30, 0.45]:
                v_tr, v_dd = simulate_smc_sweep(
                    val_df, val_df['ai_z'].values, max_hold=18,
                    z_min=z_m, z_strong=z_m + 0.45, sl_atr_pad=sl_pad, min_rr_target=rr_m
                )
                if len(v_tr) >= 90:
                    v_pnl = v_tr['ret'].sum() * 100.0
                    v_wr  = (v_tr['ret'] > 0).mean() * 100.0
                    gw = v_tr[v_tr['ret'] > 0]['ret'].sum()
                    gl = abs(v_tr[v_tr['ret'] < 0]['ret'].sum()) + 1e-10
                    v_pf = gw / gl
                    score = v_pnl * v_pf * (v_wr / 50.0)
                    if score > best_score:
                        best_score = score
                        best_cfg = {'z_min': z_m, 'rr': rr_m, 'sl_pad': sl_pad}

    print(f"2. Validation-Selected Settings (Sep 1 - Nov 8): AI_Z_Min={best_cfg['z_min']} | Full_Range_RR={best_cfg['rr']}R | Sweep_SL_Pad={best_cfg['sl_pad']}x ATR\n")

    val_tr, val_dd   = simulate_smc_sweep(val_df, val_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.45, sl_atr_pad=best_cfg['sl_pad'], min_rr_target=best_cfg['rr'])
    test_tr, test_dd = simulate_smc_sweep(test_df, test_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.45, sl_atr_pad=best_cfg['sl_pad'], min_rr_target=best_cfg['rr'])
    full_tr, full_dd = simulate_smc_sweep(full_oos_df, full_oos_df['ai_z'].values, z_min=best_cfg['z_min'], z_strong=best_cfg['z_min']+0.45, sl_atr_pad=best_cfg['sl_pad'], min_rr_target=best_cfg['rr'])

    print("=== OUT-OF-SAMPLE PERFORMANCE: LIQUIDITY SWEEP (MODEL 1: 50% EQ + MODEL 2: FULL RANGE) ===")
    print("-" * 112)
    print_stats(val_tr,  val_dd,  "1. Sep 01 - Nov 08 (Validation)")
    print_stats(test_tr, test_dd, "2. Nov 09 - Nov 15 (Blind Test)")
    print_stats(full_tr, full_dd, "3. Sep 01 - Nov 15 (Full OOS Chart)")
    print("-" * 112)

    if len(full_tr) > 0:
        print("\nModel 1 (50% EQ) & Model 2 (Full Range) Outcome Attribution:")
        print(full_tr.groupby('outcome')['ret'].agg(Trades='count', WinRate=lambda x: f"{(x>0).mean()*100:.1f}%", NetPnL=lambda x: f"{x.sum()*100:+.2f}%"))

    # =========================================================================
    # 3. COMPILE PINE SCRIPT V5 (AI_Gold_Liquidity_Sweep_AMD_v9.pine)
    # =========================================================================
    mean, scale = scaler.mean_, scaler.scale_
    w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
    w0, b0 = mlp.coefs_[0], mlp.intercepts_[0]
    w1, b1 = mlp.coefs_[1], mlp.intercepts_[1]

    pine_code = f"""//@version=5
strategy("AI Gold Liquidity Sweep & Fake-Breakout AMD v9 (Model 1: 50% EQ + Model 2: Full Range)", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=45, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.005, max_labels_count=500)

// =========================================================================
// 1. EVALUATION WINDOW (DEFAULT: FULL CHART SEP 1 - NOV 16!)
// =========================================================================
use_date_filter = input.bool(false, "Restrict to Custom Date Range (Uncheck for Full Chart)", group="1. Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Window")

// =========================================================================
// 2. ICT / SMC LIQUIDITY SWEEP & MODEL 1 / MODEL 2 SETTINGS
// =========================================================================
range_len    = input.int(32,   "ITF Dealing Range Lookback (Bars)", minval=16, maxval=96, group="2. Top-Down Structure & Sweep")
sweep_len    = input.int(10,   "LTF Stop-Hunt Liquidity Pool Lookback (Bars)", minval=5, maxval=30, group="2. Top-Down Structure & Sweep")
z_min_val    = input.float({best_cfg['z_min']:.2f}, "Min AI Sweep Conviction Z-Score", minval=-0.50, maxval=1.50, step=0.05, group="2. Top-Down Structure & Sweep")
z_strong_val = input.float({best_cfg['z_min']+0.45:.2f}, "Tier-2 High Conviction Z-Score", minval=0.0, maxval=2.00, step=0.05, group="2. Top-Down Structure & Sweep")
sl_pad_atr   = input.float({best_cfg['sl_pad']:.2f}, "Invalidation Stop Buffer Beyond Sweep Wick (ATR)", minval=0.10, maxval=1.00, step=0.05, group="3. Model 1 (50% EQ) & Model 2 (Full Range)")
min_rr_val   = input.float({best_cfg['rr']:.1f},  "Model 2 Full Range Minimum R:R Target", minval=1.5, maxval=4.0, step=0.1, group="3. Model 1 (50% EQ) & Model 2 (Full Range)")
max_hold_b   = input.int(18,   "Max Holding Horizon (Bars)", minval=6, maxval=40, group="3. Model 1 (50% EQ) & Model 2 (Full Range)")
tier1_pct    = input.float(45.0, "Tier-1 Standard Allocation (%)", minval=10.0, maxval=100.0, step=5.0, group="4. Capital & Reload Zone Pyramid")
tier2_pct    = input.float(65.0, "Tier-2 Strong Allocation (%)", minval=20.0, maxval=100.0, step=5.0, group="4. Capital & Reload Zone Pyramid")
enable_pyr   = input.bool(true, "Enable Model 2 Reload Zone Pyramid (+35%)", group="4. Capital & Reload Zone Pyramid")

// =========================================================================
// 3. TOP-DOWN STRUCTURE (HTF TREND, ITF RANGE EQ, LTF LIQUIDITY SWEEP & FVG)
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

// 1. HTF Trend (H4 / Daily Draw on Liquidity)
ema64     = ta.ema(close, 64)
ema192    = ta.ema(close, 192)
htf_trend = ((ema64 - ema192) / close) / atr_pct

// 2. ITF Dealing Range (Range High, Range Low, and 50% Equilibrium Midpoint!)
range_h    = ta.highest(high[1], range_len)
range_l    = ta.lowest(low[1], range_len)
range_eq   = (range_h + range_l) / 2.0
range_size = (range_h - range_l) + 1e-10
pd_pos     = (close - range_l) / range_size // < 0.50 = Discount, > 0.50 = Premium

// 3. LTF Liquidity Pools (Stop-Hunt Highs & Lows)
liq_high_10 = ta.highest(high[1], sweep_len)
liq_low_10  = ta.lowest(low[1], sweep_len)

// Model 1 Manipulation Sweep Detection (Fake Breakout + Snap Back Inside Range!)
swept_low_now   = (low < liq_low_10) and (close > liq_low_10) and (close > open)
swept_low_prev  = (low[1] < liq_low_10[1]) and (close > liq_low_10[1]) and (close > open) and (close > close[1])
swept_high_now  = (high > liq_high_10) and (close < liq_high_10) and (close < open)
swept_high_prev = (high[1] > liq_high_10[1]) and (close < liq_high_10[1]) and (close < open) and (close < close[1])

// Fair Value Gap (FVG) & Order Block Reload Zone (61.8% - 80% Retracement in Discount/Premium)
bull_fvg = (low > high[2]) and (close[1] > open[1])
bear_fvg = (high < low[2]) and (close[1] < open[1])
recent_bull_fvg = ta.highest(bull_fvg ? 1.0 : 0.0, 6) > 0
recent_bear_fvg = ta.highest(bear_fvg ? 1.0 : 0.0, 6) > 0

bull_reload = (pd_pos <= 0.42) and (pd_pos >= 0.12) and recent_bull_fvg and (close > open) and ((close - low) > 0.5 * (high - low))
bear_reload = (pd_pos >= 0.58) and (pd_pos <= 0.88) and recent_bear_fvg and (close < open) and ((high - close) > 0.5 * (high - low))

bull_setup_raw = swept_low_now or swept_low_prev or bull_reload
bear_setup_raw = swept_high_now or swept_high_prev or bear_reload

// --- 12 AI FEATURES ---
rng    = (high - low) + 1e-10
l_wick = math.min(open, close) - low
u_wick = high - math.max(open, close)

f1 = (l_wick - u_wick) / rng
f2 = (close - open) / rng
f3 = (pd_pos - 0.50) * 2.0
f4 = (close - ((liq_high_10 + liq_low_10) / 2.0)) / (atr14 + 1e-10)
f5 = (close - nz(open[2], open)) / (atr14 + 1e-10)
f6 = htf_trend

vol_ma20 = ta.sma(volume, 20) + 1e-10
f7 = math.min(4.0, math.max(0.2, volume / vol_ma20)) - 1.0

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs14 = pine_rma(loss, 14) == 0 ? 0.0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi14 = 100.0 - (100.0 / (1.0 + rs14))
f8 = (rsi14 - 50.0) / 50.0

typ_p  = (high + low + close) / 3.0
vwap24 = math.sum(typ_p * volume, 24) / (math.sum(volume, 24) + 1e-10)
f9 = ((close - vwap24) / close) / atr_pct
f10 = (range_size / (atr14 + 1e-10)) / 5.0 - 1.0

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
// 5. LIQUIDITY SWEEP ENTRY & MODEL 1 (50% EQ) + MODEL 2 (FULL RANGE) EXITS
// =========================================================================
calc_qty(pct_eq) =>
    math.max(0.001, (strategy.equity * (pct_eq / 100.0)) / close)

// Buy Sweeps in Discount (< 52% of Range); Sell Sweeps in Premium (> 48% of Range)
is_bull_sweep = in_date_window and bull_setup_raw and (pd_pos < 0.52) and (htf_trend > -0.8 or ai_z >= z_strong_val) and (ai_z >= z_min_val)
is_bear_sweep = in_date_window and bear_setup_raw and (pd_pos > 0.48) and (htf_trend < 0.8 or ai_z <= -z_strong_val) and (ai_z <= -z_min_val)

bull_pct = ai_z >= z_strong_val ? tier2_pct : tier1_pct
bear_pct = ai_z <= -z_strong_val ? tier2_pct : tier1_pct

var int   entry_bar  = na
var int   last_loss  = -999
var float entry_p    = na
var float risk_1r    = na
var float sl_level   = na
var float tp1_eq     = na
var float tp2_full   = na
var bool  hit_tp1    = false
var bool  scaled_in  = false
var float active_pct = 0.0

if strategy.position_size == 0 and strategy.position_size[1] != 0
    if strategy.closedtrades > 0 and strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        last_loss := bar_index

cooldown_ok = (bar_index - last_loss) >= 2

if barstate.isconfirmed
    if strategy.position_size == 0 and cooldown_ok
        if is_bull_sweep
            sweep_low  = math.min(low, low[1])
            raw_r      = (close - sweep_low) + sl_pad_atr * atr14
            r_dist     = math.max(0.55 * atr14, math.min(1.50 * atr14, raw_r))
            strategy.entry("Sweep Long", strategy.long, qty=calc_qty(bull_pct), comment="SWEEP BUY (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_disp*100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            risk_1r    := r_dist
            sl_level   := close - r_dist
            tp1_eq     := math.max(range_eq, close + 1.1 * r_dist)
            tp2_full   := math.max(range_h, close + min_rr_val * r_dist)
            hit_tp1    := false
            scaled_in  := false
            active_pct := bull_pct
        else if is_bear_sweep
            sweep_high = math.max(high, high[1])
            raw_r      = (sweep_high - close) + sl_pad_atr * atr14
            r_dist     = math.max(0.55 * atr14, math.min(1.50 * atr14, raw_r))
            strategy.entry("Sweep Short", strategy.short, qty=calc_qty(bear_pct), comment="SWEEP SELL (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round((1.0-prob_disp)*100)) + "%")
            entry_bar  := bar_index
            entry_p    := close
            risk_1r    := r_dist
            sl_level   := close + r_dist
            tp1_eq     := math.min(range_eq, close - 1.1 * r_dist)
            tp2_full   := math.min(range_l, close - min_rr_val * r_dist)
            hit_tp1    := false
            scaled_in  := false
            active_pct := bear_pct

// --- MANAGE OPEN TRADES: MODEL 1 (50% EQ PARTIAL) + MODEL 2 (FULL RANGE RUN) ---
if strategy.position_size > 0
    if not hit_tp1 and high >= tp1_eq
        hit_tp1  := true
        sl_level := math.max(sl_level, entry_p + 0.45 * risk_1r)

    if barstate.isconfirmed and enable_pyr and not scaled_in and close >= (entry_p + 0.40 * risk_1r) and ai_z >= z_min_val
        strategy.entry("Reload Long", strategy.long, qty=calc_qty(35.0), comment="+Model2 Reload (35%)")
        scaled_in  := true
        active_pct := active_pct + 35.0
        sl_level   := math.max(sl_level, entry_p)

    // Exit 50% at Model 1 (50% Equilibrium) and 50% at Model 2 (Full Opposing Range)
    strategy.exit("Model1 50% EQ", from_entry="Sweep Long", qty_percent=50, limit=tp1_eq, stop=sl_level)
    strategy.exit("Model2 Full Range", from_entry="Sweep Long", limit=tp2_full, stop=sl_level)
    strategy.exit("Exit Reload Long", from_entry="Reload Long", limit=tp2_full, stop=sl_level)

    if (bar_index - entry_bar) >= max_hold_b
        strategy.close_all(comment="Range Time Exit")

else if strategy.position_size < 0
    if not hit_tp1 and low <= tp1_eq
        hit_tp1  := true
        sl_level := math.min(sl_level, entry_p - 0.45 * risk_1r)

    if barstate.isconfirmed and enable_pyr and not scaled_in and close <= (entry_p - 0.40 * risk_1r) and ai_z <= -z_min_val
        strategy.entry("Reload Short", strategy.short, qty=calc_qty(35.0), comment="+Model2 Reload (35%)")
        scaled_in  := true
        active_pct := active_pct + 35.0
        sl_level   := math.min(sl_level, entry_p)

    strategy.exit("Model1 50% EQ Short", from_entry="Sweep Short", qty_percent=50, limit=tp1_eq, stop=sl_level)
    strategy.exit("Model2 Full Range Short", from_entry="Sweep Short", limit=tp2_full, stop=sl_level)
    strategy.exit("Exit Reload Short", from_entry="Reload Short", limit=tp2_full, stop=sl_level)

    if (bar_index - entry_bar) >= max_hold_b
        strategy.close_all(comment="Range Time Exit")

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. VISUAL RANGE HIGH, RANGE LOW, 50% EQUILIBRIUM & AMD HUD
// =========================================================================
p_rh = plot(range_h,  "ITF Range High (Premium Liquidity)", color=color.new(color.red, 35), linewidth=1)
p_eq = plot(range_eq, "50% Range Equilibrium (Model 1 Target)", color=color.new(color.yellow, 20), style=plot.style_circles, linewidth=2)
p_rl = plot(range_l,  "ITF Range Low (Discount Liquidity)", color=color.new(color.lime, 35), linewidth=1)
fill(p_rh, p_eq, color=color.new(color.red, 94), title="Premium Zone (Sell Sweeps)")
fill(p_eq, p_rl, color=color.new(color.green, 94), title="Discount Zone (Buy Sweeps)")

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_disp >= 0.5 ? math.round(prob_disp * 100, 1) : math.round((1.0 - prob_disp) * 100, 1)
    zone_txt = pd_pos < 0.50 ? "DISCOUNT (" + str.tostring(math.round(pd_pos*100)) + "%)" : "PREMIUM (" + str.tostring(math.round(pd_pos*100)) + "%)"
    sig_txt  = is_bull_sweep ? "SWEEP BUY (MODEL 1&2)" : is_bear_sweep ? "SWEEP SELL (MODEL 1&2)" : "WAITING FOR SWEEP"
    sig_col  = is_bull_sweep ? color.green : is_bear_sweep ? color.red : color.gray

    table.cell(hud, 0, 0, "Strategy Logic", text_color=color.white)
    table.cell(hud, 1, 0, "ICT Liquidity Sweep + AI", text_color=color.yellow)
    table.cell(hud, 0, 1, "HTF Draw on Liq", text_color=color.white)
    table.cell(hud, 1, 1, htf_trend > 0 ? "BULLISH DRAW" : "BEARISH DRAW", text_color=htf_trend > 0 ? color.lime : color.red)
    table.cell(hud, 0, 2, "ITF Range Zone", text_color=color.white)
    table.cell(hud, 1, 2, zone_txt, text_color=pd_pos < 0.50 ? color.lime : color.orange)
    table.cell(hud, 0, 3, "FVG / Displacement", text_color=color.white)
    table.cell(hud, 1, 3, recent_bull_fvg ? "BULL FVG ACTIVE" : recent_bear_fvg ? "BEAR FVG ACTIVE" : "NO GAP", text_color=color.aqua)
    table.cell(hud, 0, 4, "AI Sweep Score", text_color=color.white)
    table.cell(hud, 1, 4, str.tostring(conf_pct) + "% Conviction", text_color=color.white)
    table.cell(hud, 0, 5, "Live Action", text_color=color.white)
    table.cell(hud, 1, 5, sig_txt, bgcolor=sig_col, text_color=color.white)
"""

    with open("AI_Gold_Liquidity_Sweep_AMD_v9.pine", "w") as f:
        f.write(pine_code)

    print("\n3. Generated 'AI_Gold_Liquidity_Sweep_AMD_v9.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()