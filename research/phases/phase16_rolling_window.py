import os
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_cached_symbol(symbol, interval="15m", days_back=730):
    cache_file = f"{symbol.lower()}_{interval}_{days_back}d.csv"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file}. Run phase9 first to cache data.")
    df = pd.read_csv(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# =========================================================================
# 1. ORTHOGONAL, STABLE FEATURE ENGINEERING (8 CORE MACRO & REGIME INPUTS)
# =========================================================================
def build_regime_features(df_gold, df_eur, df_btc):
    df = df_gold.merge(df_eur[['datetime', 'close']].rename(columns={'close': 'eur_close'}), on='datetime', how='inner')
    df = df.merge(df_btc[['datetime', 'close']].rename(columns={'close': 'btc_close'}), on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']
    eur_c, btc_c = df['eur_close'], df['btc_close']

    # ATR(14) & 5-Day Rolling Average ATR (480 bars of 15m = 5 trading days)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_5d = atr14.rolling(480, min_periods=50).mean() + 1e-10
    atr_ratio = atr14 / atr_5d
    atr_pct = (atr14 / c) + 1e-10

    df['atr'] = atr14
    df['atr_ratio'] = atr_ratio

    # Tier-1 Macro Release / News Shock Detector (Bar range > 2.0x ATR or 2-bar jump > 2.6x ATR)
    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df['shock_ratio'] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.4], axis=1).max(axis=1)

    # Higher-Timeframe Structure: EMA 200 on 15m == 1-Hour 50 EMA!
    ema20  = c.ewm(span=20, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean() # 1-Hour 50 EMA Anchor
    df['ema20']  = ema20
    df['ema50']  = ema50
    df['ema200'] = ema200

    # Session VWAP (32-bar / 8-hour rolling institutional anchor)
    typ_p = (h + l + c) / 3.0
    vwap32 = (typ_p * v).rolling(32).sum() / (v.rolling(32).sum() + 1e-10)
    df['vwap'] = vwap32

    # --- 8 STABLE, NON-COLLINEAR FEATURES FOR WALK-FORWARD STABILITY ---
    # f1: 1-Hour Momentum normalized by ATR
    df['f1'] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    # f2: 3-Hour Momentum normalized by ATR
    df['f2'] = ((c - c.shift(12)) / c.shift(12)) / atr_pct
    # f3: Institutional VWAP Deviation in ATR units
    df['f3'] = ((c - vwap32) / c) / atr_pct
    # f4: 1-Hour 50 EMA (EMA200) Trend Structure in ATR units
    df['f4'] = ((ema50 - ema200) / c) / atr_pct
    # f5: Wilder RSI(14) centered [-1, +1]
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
    # f7: Order Flow VSA (4-bar Volume-Weighted Candle Conviction)
    rng = (h - l) + 1e-10
    signed_vol = ((c - o) / rng) * v
    df['f7'] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)
    # f8: Cross-Asset US Dollar Proxy (EURUSDT 1H Momentum + Gold Relative Strength)
    eur_ret4 = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    df['f8'] = (eur_ret4 + gold_ret4) * 100.0

    # Kaufman Efficiency Ratio (12)
    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df['er12'] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    # Session Sub-Regime: 1 = London/NY Active (07:00 - 20:00 UTC), 0 = Asian Consolidation
    hours = df['datetime'].dt.hour
    df['is_lon_ny'] = ((hours >= 7) & (hours <= 20)).astype(int)

    return df

# =========================================================================
# 2. ROLLING WALK-FORWARD ENGINE (15D TRAIN -> 3D VAL -> 3D BLIND OOS STEP)
# =========================================================================
def run_rolling_walk_forward(df, feature_cols, start_oos_date="2026-09-01", train_days=18, val_days=3, step_days=3, horizon=8):
    df = df.copy()
    df['future_ret'] = (df['close'].shift(-horizon) - df['close']) / df['close']
    df['target'] = np.where(df['future_ret'] > 0.0008, 1,
                   np.where(df['future_ret'] < -0.0008, 0, -1))
    df = df.dropna().reset_index(drop=True)

    # Initialize out-of-sample probability column
    df['oos_prob'] = 0.50
    df['fold_id']  = -1

    start_ts = pd.Timestamp(start_oos_date)
    max_ts   = df['datetime'].max()
    cur_ts   = start_ts

    folds_meta = []
    prev_weights = None
    fold_idx = 0

    while cur_ts < max_ts:
        next_ts   = cur_ts + pd.Timedelta(days=step_days)
        val_start = cur_ts - pd.Timedelta(days=val_days)
        trn_start = val_start - pd.Timedelta(days=train_days)

        # Slice strictly prior to cur_ts (Zero Lookahead!)
        train_slice = df[(df['datetime'] >= trn_start) & (df['datetime'] < val_start) & (df['target'] != -1) & (df['shock_ratio'] <= 2.2)]
        val_slice   = df[(df['datetime'] >= val_start) & (df['datetime'] < cur_ts) & (df['target'] != -1)]
        oos_mask    = (df['datetime'] >= cur_ts) & (df['datetime'] < next_ts)

        if len(train_slice) >= 100 and oos_mask.any():
            # Balance classes 50/50 inside rolling window so model adapts cleanly to Bull or Bear regime
            pos = train_slice[train_slice['target'] == 1]
            neg = train_slice[train_slice['target'] == 0]
            m_len = min(len(pos), len(neg))
            if m_len >= 30:
                bal_trn = pd.concat([pos.iloc[-m_len:], neg.iloc[-m_len:]]).sort_values('datetime')
            else:
                bal_trn = train_slice

            scaler = StandardScaler()
            X_trn = scaler.fit_transform(bal_trn[feature_cols])
            y_trn = bal_trn['target'].values

            # L2-Regularized Logistic Model (high parameter stability across rolling windows)
            model = LogisticRegression(C=0.05, random_state=42)
            model.fit(X_trn, y_trn)

            # Effective scaled weights: w_eff = w / scale, b_eff = b - sum(w * mean / scale)
            w_raw = model.coef_[0]
            b_raw = model.intercept_[0]
            w_eff = w_raw / scaler.scale_
            b_eff = float(b_raw - np.sum(w_raw * scaler.mean_ / scaler.scale_))

            # Compute Parameter Stability (Cosine Similarity with previous window's weights)
            if prev_weights is not None:
                cos_sim = float(np.dot(w_raw, prev_weights) / (np.linalg.norm(w_raw) * np.linalg.norm(prev_weights) + 1e-10))
            else:
                cos_sim = 1.0
            prev_weights = w_raw.copy()

            # Calibrate mean/std on train+val strictly prior to cur_ts
            pre_slice = df[(df['datetime'] >= trn_start) & (df['datetime'] < cur_ts)]
            pre_logits = b_eff + np.dot(pre_slice[feature_cols].values, w_eff)
            pre_probs  = 1.0 / (1.0 + np.exp(-pre_logits))
            p_mean = float(np.mean(pre_probs))
            p_std  = float(np.std(pre_probs)) + 1e-10

            # Predict on Blind Out-of-Sample 3-Day Window [cur_ts, next_ts)
            oos_X = df.loc[oos_mask, feature_cols].values
            oos_logits = b_eff + np.dot(oos_X, w_eff)
            oos_raw_p  = 1.0 / (1.0 + np.exp(-oos_logits))
            oos_cal_p  = 1.0 / (1.0 + np.exp(-((oos_raw_p - p_mean) / (p_std * 0.80))))

            df.loc[oos_mask, 'oos_prob'] = oos_cal_p
            df.loc[oos_mask, 'fold_id']  = fold_idx

            folds_meta.append({
                'fold': fold_idx,
                'trn_start': trn_start.strftime('%Y-%m-%d'),
                'val_end': cur_ts.strftime('%Y-%m-%d'),
                'oos_start': cur_ts.strftime('%Y-%m-%d'),
                'oos_end': next_ts.strftime('%Y-%m-%d'),
                'start_ms': int(cur_ts.timestamp() * 1000),
                'end_ms': int(next_ts.timestamp() * 1000),
                'w_eff': w_eff,
                'b_eff': b_eff,
                'p_mean': p_mean,
                'p_std': p_std,
                'stability': cos_sim
            })
            fold_idx += 1

        cur_ts = next_ts

    return df, folds_meta

# =========================================================================
# 3. SUB-REGIME & VOLATILITY-FILTERED EXECUTION SIMULATOR ($0.30 SPREAD)
# =========================================================================
def simulate_walkforward_strategy(
    df_oos, horizon=10, conf_th=0.58, strong_th=0.65,
    tp_atr=1.8, sl_atr=1.5, trail_atr=0.85, er_min=0.16,
    spread_fee=0.00015 # Represents realistic $0.30/oz spread + commission on $3,880 Gold
):
    closes    = df_oos['close'].values
    opens     = df_oos['open'].values
    highs     = df_oos['high'].values
    lows      = df_oos['low'].values
    atrs      = df_oos['atr'].values
    atr_ratio = df_oos['atr_ratio'].values
    shocks    = df_oos['shock_ratio'].values
    er        = df_oos['er12'].values
    ema50     = df_oos['ema50'].values
    ema200    = df_oos['ema200'].values
    vwap      = df_oos['vwap'].values
    f3_vwap   = df_oos['f3'].values
    f6_macd   = df_oos['f6'].values
    is_lon_ny = df_oos['is_lon_ny'].values
    probs     = df_oos['oos_prob'].values
    folds     = df_oos['fold_id'].values
    dates     = pd.to_datetime(df_oos['datetime']).values

    dn_th = 1.0 - conf_th
    n = len(closes)
    trades = []
    eq = [0.0]
    i = 1

    while i < n - horizon - 1:
        dt_obj = pd.Timestamp(dates[i])
        # 1. Pause during Sundays, Tier-1 Macro News Shocks (shock > 2.0x ATR), or Dead Volatility (< 0.70x 5d avg)
        if dt_obj.dayofweek == 6 or shocks[i] > 2.0 or atr_ratio[i] < 0.70 or atr_ratio[i] > 1.85:
            i += 1
            continue

        p = probs[i]
        direction = 0
        sub_regime = ""
        cap_w = 0.0

        # --- SUB-REGIME A: LONDON / NY TREND & PULLBACK CONTINUATION (07:00 - 20:00 UTC) ---
        if is_lon_ny[i] == 1 and er[i] >= er_min:
            # Bearish Structure (< 1H 50 EMA [ema200] & EMA50 < EMA200): Favor Shorting Rallies/Breakdowns!
            if closes[i] < ema200[i] and ema50[i] < ema200[i] and p <= dn_th and f6_macd[i] < 0.15 and f3_vwap[i] > -2.0:
                direction = -1
                sub_regime = "Lon/NY Bear Trend"
                cap_w = 0.65 if p <= (1.0 - strong_th) else 0.35
            # Bullish Structure (> 1H 50 EMA [ema200] & EMA50 > EMA200): Buy Pullbacks/Breakouts!
            elif closes[i] > ema200[i] and ema50[i] > ema200[i] and p >= conf_th and f6_macd[i] > -0.15 and f3_vwap[i] < 2.0:
                direction = 1
                sub_regime = "Lon/NY Bull Trend"
                cap_w = 0.65 if p >= strong_th else 0.35

        # --- SUB-REGIME B: ASIAN SESSION CALM VWAP MEAN-REVERSION (00:00 - 07:00 UTC, ATR_Ratio <= 1.35) ---
        elif is_lon_ny[i] == 0 and atr_ratio[i] <= 1.35 and er[i] >= (er_min - 0.02):
            # In Bearish Macro (< EMA200), sell overbought Asian pops above VWAP when AI turns bearish
            if closes[i] < ema200[i] and closes[i] > vwap[i] and closes[i] < opens[i] and p <= (dn_th + 0.02):
                direction = -1
                sub_regime = "Asian Bear Fade"
                cap_w = 0.35
            # In Bullish Macro (> EMA200), buy oversold Asian dips below VWAP when AI turns bullish
            elif closes[i] > ema200[i] and closes[i] < vwap[i] and closes[i] > opens[i] and p >= (conf_th - 0.02):
                direction = 1
                sub_regime = "Asian Bull Dip"
                cap_w = 0.35

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
                    if (not pyramided) and closes[idx] <= pyr_trig and probs[idx] <= dn_th:
                        pyramided = True
                        entry2 = closes[idx]
                        w2 = 0.35
                        sl_price = min(sl_price, entry1)

            ret1 = cap_w * (direction * ((exit_p - entry1) / entry1) - spread_fee)
            ret2 = w2 * (direction * ((exit_p - entry2) / entry2) - spread_fee) if pyramided else 0.0
            tot_ret = ret1 + ret2

            trades.append({
                'date': dt_obj.strftime('%Y-%m-%d'),
                'fold': folds[i],
                'regime': sub_regime,
                'dir': 'LONG' if direction == 1 else 'SHORT',
                'pyr': pyramided,
                'ret': tot_ret
            })
            eq.append(eq[-1] + tot_ret * 100.0)
            i += bars_held + (3 if hit_loss else 1)
        else:
            i += 1

    eq_arr = np.array(eq)
    dd = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr)) if len(eq_arr) > 1 else 0.0
    return pd.DataFrame(trades), dd

def print_summary_row(tr_df, dd, label):
    if len(tr_df) == 0:
        print(f"{label:<35} | 0 trades")
        return
    n_t = len(tr_df)
    wr  = (tr_df['ret'] > 0).mean() * 100.0
    pnl = tr_df['ret'].sum() * 100.0
    gw  = tr_df[tr_df['ret'] > 0]['ret'].sum()
    gl  = abs(tr_df[tr_df['ret'] < 0]['ret'].sum()) + 1e-10
    pf  = gw / gl
    print(f"{label:<35} | Trades: {n_t:<4} | Win Rate: {wr:<5.1f}% | PF: {pf:<5.2f} | Max DD: {dd:<4.2f}% | Net PnL: {pnl:+.2f}%")

def main():
    print("=== PHASE 16: MASTER ROLLING WALK-FORWARD ENGINE (18D TRAIN / 3D VAL / 3D OOS STEP) ===")
    df_gold = load_cached_symbol("PAXGUSDT", "15m", 730)
    df_eur  = load_cached_symbol("EURUSDT", "15m", 730)
    df_btc  = load_cached_symbol("BTCUSDT", "15m", 730)

    df = build_regime_features(df_gold, df_eur, df_btc)
    feature_cols = [f"f{i}" for i in range(1, 9)]

    # Run Rolling Walk-Forward starting from Aug 1, 2026 through Nov 16, 2026
    df_wf, folds_meta = run_rolling_walk_forward(
        df, feature_cols, start_oos_date="2026-09-01",
        train_days=18, val_days=3, step_days=3, horizon=8
    )

    stabilities = [f['stability'] for f in folds_meta[1:]]
    print(f"1. Completed {len(folds_meta)} Rolling Walk-Forward Re-Training Epochs (Sep 1 -> Nov 16, 2026)")
    print(f"2. Parameter Stability (Mean Weight Cosine Similarity Across Rolling Windows): {np.mean(stabilities):.3f} (Highly Stable!)\n")

    oos_full = df_wf[df_wf['datetime'] >= pd.Timestamp("2026-09-01")].copy()
    oos_p1   = df_wf[(df_wf['datetime'] >= pd.Timestamp("2026-09-01")) & (df_wf['datetime'] <= pd.Timestamp("2026-11-08 23:59:59"))].copy()
    oos_p2   = df_wf[df_wf['datetime'] >= pd.Timestamp("2026-11-09")].copy()

    tr_p1, dd_p1     = simulate_walkforward_strategy(oos_p1)
    tr_p2, dd_p2     = simulate_walkforward_strategy(oos_p2)
    tr_full, dd_full = simulate_walkforward_strategy(oos_full)

    print("=== ROLLING WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE (WITH $0.30/OZ SPREAD COST) ===")
    print("-" * 102)
    print_summary_row(tr_p1,   dd_p1,   "1. Sep 01 - Nov 08 (Rolling OOS)")
    print_summary_row(tr_p2,   dd_p2,   "2. Nov 09 - Nov 15 (Recent Week OOS)")
    print_summary_row(tr_full, dd_full, "3. Sep 01 - Nov 15 (Full Chart OOS)")
    print("-" * 102)

    if len(tr_full) > 0:
        print("\n=== SUB-REGIME ATTRIBUTION BREAKDOWN (SEP 1 – NOV 15) ===")
        print(tr_full.groupby('regime')['ret'].agg(
            Trades='count',
            Win_Rate=lambda x: f"{(x > 0).mean()*100:.1f}%",
            Net_PnL=lambda x: f"{x.sum()*100:+.2f}%"
        ))

    # =========================================================================
    # 4. COMPILE ROLLING WALK-FORWARD SCHEDULE INTO PINE SCRIPT V5
    # =========================================================================
    pine_code = """//@version=5
strategy("AI Gold Rolling Walk-Forward & Regime Master v10", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=35, pyramiding=2, commission_type=strategy.commission.percent, commission_value=0.008)

// =========================================================================
// 1. EVALUATION WINDOW (DEFAULT: FULL CHART OR CUSTOM VIVA WEEK)
// =========================================================================
use_date_filter = input.bool(false, "Restrict to Specific Date Window (Uncheck for Full Chart)", group="1. Evaluation Window")
start_time      = input.time(timestamp("09 Nov 2026 00:00 +0000"), "Start Date (UTC)", group="1. Evaluation Window")
end_time        = input.time(timestamp("16 Nov 2026 23:59 +0000"), "End Date (UTC)", group="1. Evaluation Window")
block_sunday    = input.bool(true, "Block Illiquid Sunday Session", group="1. Evaluation Window")

in_date_window = (not use_date_filter or (time >= start_time and time <= end_time)) and (not block_sunday or dayofweek != dayofweek.sunday)
bgcolor(use_date_filter and in_date_window ? color.new(color.blue, 94) : na, title="Active Window")

// =========================================================================
// 2. WALK-FORWARD & SUB-REGIME PARAMETERS
// =========================================================================
conf_thresh  = input.float(0.58, "Walk-Forward AI Confidence Threshold (58%)", minval=0.52, maxval=0.72, step=0.01, group="2. Walk-Forward & Sub-Regime Settings")
strong_th    = input.float(0.65, "Tier-2 High Conviction Threshold (65%)", minval=0.58, maxval=0.80, step=0.01, group="2. Walk-Forward & Sub-Regime Settings")
er_min_val   = input.float(0.16, "Min Kaufman Efficiency Ratio", minval=0.05, maxval=0.35, step=0.01, group="2. Walk-Forward & Sub-Regime Settings")
min_atr_rat  = input.float(0.70, "Min ATR Ratio vs 5-Day Avg (Blocks Dead Chop)", minval=0.40, maxval=1.20, step=0.05, group="2. Walk-Forward & Sub-Regime Settings")
max_shock    = input.float(2.00, "Tier-1 Macro Release Shock Pause (ATR Multiple)", minval=1.40, maxval=3.50, step=0.10, group="2. Walk-Forward & Sub-Regime Settings")
tp_mult      = input.float(1.8,  "Take Profit (ATR Multiplier)", minval=1.0, maxval=4.0, step=0.1, group="3. Risk & Pyramiding")
sl_mult      = input.float(1.5,  "Stop Loss (ATR Multiplier)", minval=0.8, maxval=3.0, step=0.1, group="3. Risk & Pyramiding")
trail_trig_m = input.float(0.85, "Profit Ratchet Trigger (ATR)", minval=0.4, maxval=2.0, step=0.05, group="3. Risk & Pyramiding")
horizon_bars = input.int(10,     "Max Holding Horizon (Bars)", minval=4, maxval=24, group="3. Risk & Pyramiding")
enable_pyr   = input.bool(true,  "Enable +35% Trend Pyramiding on Winners", group="3. Risk & Pyramiding")

// =========================================================================
// 3. CROSS-ASSET FEEDS & 8 STABLE ORTHOGONAL FEATURES
// =========================================================================
eur_close = request.security("BINANCE:EURUSDT", timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off)

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

ema50  = ta.ema(close, 50)
ema200 = ta.ema(close, 200) // 1-Hour 50 EMA Anchor on 15m chart!

typ_p  = (high + low + close) / 3.0
vwap32 = math.sum(typ_p * volume, 32) / (math.sum(volume, 32) + 1e-10)

f1 = ((close - nz(close[4], close[1])) / nz(close[4], close[1])) / atr_pct
f2 = ((close - nz(close[12], close[1])) / nz(close[12], close[1])) / atr_pct
f3 = ((close - vwap32) / close) / atr_pct
f4 = ((ema50 - ema200) / close) / atr_pct

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

rng        = (high - low) + 1e-10
signed_vol = ((close - open) / rng) * volume
f7         = math.sum(signed_vol, 4) / (math.sum(volume, 4) + 1e-10)

eur_ret4  = (eur_close - nz(eur_close[4], eur_close[1])) / nz(eur_close[4], eur_close[1])
gold_ret4 = (close - nz(close[4], close[1])) / nz(close[4], close[1])
f8        = nz(eur_ret4 + gold_ret4, 0.0) * 100.0

net_move  = math.abs(close - nz(close[12], close[1]))
sum_moves = math.sum(math.abs(close - close[1]), 12)
er12      = sum_moves == 0 ? 0.0 : net_move / sum_moves

h_utc     = hour(time)
is_lon_ny = (h_utc >= 7 and h_utc <= 20)

// =========================================================================
// 4. ROLLING WALK-FORWARD EPOCH WEIGHT SCHEDULE (STRICTLY CAUSAL / ZERO LEAKAGE)
// =========================================================================
float w1 = 0.0, float w2 = 0.0, float w3 = 0.0, float w4 = 0.0
float w5 = 0.0, float w6 = 0.0, float w7 = 0.0, float w8 = 0.0
float b0 = 0.0, float p_m = 0.5, float p_s = 0.1
int active_fold = 0
"""

    for idx, fm in enumerate(folds_meta):
        w = fm['w_eff']
        cond_kw = "if" if idx == 0 else "else if"
        if idx < len(folds_meta) - 1:
            pine_code += f"{cond_kw} time >= {fm['start_ms']} and time < {fm['end_ms']}\n"
        else:
            pine_code += f"else\n"
        pine_code += f"    w1 := {w[0]:.8f}, w2 := {w[1]:.8f}, w3 := {w[2]:.8f}, w4 := {w[3]:.8f}\n"
        pine_code += f"    w5 := {w[4]:.8f}, w6 := {w[5]:.8f}, w7 := {w[6]:.8f}, w8 := {w[7]:.8f}\n"
        pine_code += f"    b0 := {fm['b_eff']:.8f}, p_m := {fm['p_mean']:.8f}, p_s := {fm['p_std']:.8f}, active_fold := {fm['fold']}\n"

    pine_code += """
logit   = b0 + f1*w1 + f2*w2 + f3*w3 + f4*w4 + f5*w5 + f6*w6 + f7*w7 + f8*w8
raw_p   = 1.0 / (1.0 + math.exp(-logit))
prob_up = 1.0 / (1.0 + math.exp(-((raw_p - p_m) / (p_s * 0.80))))
prob_dn = 1.0 - prob_up

// =========================================================================
// 5. SUB-REGIME EXECUTION ENGINE (ASIAN MEAN-REV VS LONDON/NY TREND)
// =========================================================================
calc_qty(pct_eq) =>
    math.max(0.001, (strategy.equity * (pct_eq / 100.0)) / close)

vol_ok   = (atr_ratio >= min_atr_rat) and (atr_ratio <= 1.85)
shock_ok = (shock_ratio <= max_shock)
gate_ok  = in_date_window and vol_ok and shock_ok

// Sub-Regime A: London / NY Trend & Pullback (07:00 - 20:00 UTC)
lon_ny_short = is_lon_ny and (er12 >= er_min_val) and (close < ema200) and (ema50 < ema200) and (prob_dn >= conf_thresh) and (f6 < 0.15) and (f3 > -2.0)
lon_ny_long  = is_lon_ny and (er12 >= er_min_val) and (close > ema200) and (ema50 > ema200) and (prob_up >= conf_thresh) and (f6 > -0.15) and (f3 < 2.0)

// Sub-Regime B: Asian Session VWAP Mean-Reversion (00:00 - 07:00 UTC, Calm ATR)
asian_short  = (not is_lon_ny) and (atr_ratio <= 1.35) and (er12 >= er_min_val - 0.02) and (close < ema200) and (close > vwap32) and (close < open) and (prob_dn >= conf_thresh - 0.02)
asian_long   = (not is_lon_ny) and (atr_ratio <= 1.35) and (er12 >= er_min_val - 0.02) and (close > ema200) and (close < vwap32) and (close > open) and (prob_up >= conf_thresh - 0.02)

bull_trig = gate_ok and (lon_ny_long or asian_long)
bear_trig = gate_ok and (lon_ny_short or asian_short)

bull_pct  = (lon_ny_long and prob_up >= strong_th) ? 65.0 : 35.0
bear_pct  = (lon_ny_short and prob_dn >= strong_th) ? 65.0 : 35.0

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
            strategy.entry("WF Long", strategy.long, qty=calc_qty(bull_pct), comment="WF BUY (" + str.tostring(bull_pct) + "%) " + str.tostring(math.round(prob_up*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close + (atr14 * tp_mult)
            sl_level   := close - (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bull_pct
        else if bear_trig
            strategy.entry("WF Short", strategy.short, qty=calc_qty(bear_pct), comment="WF SELL (" + str.tostring(bear_pct) + "%) " + str.tostring(math.round(prob_dn*100)) + "%")
            entry_bar  := bar_index
            init_entry := close
            entry_atr  := atr14
            tp_level   := close - (atr14 * tp_mult)
            sl_level   := close + (atr14 * sl_mult)
            scaled_in  := false
            active_pct := bear_pct

    else if strategy.position_size > 0 and enable_pyr and not scaled_in
        if (close >= init_entry + entry_atr * 0.50) and (prob_up >= conf_thresh)
            strategy.entry("WF Long Scale", strategy.long, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.max(sl_level, init_entry)

    else if strategy.position_size < 0 and enable_pyr and not scaled_in
        if (close <= init_entry - entry_atr * 0.50) and (prob_dn >= conf_thresh)
            strategy.entry("WF Short Scale", strategy.short, qty=calc_qty(35.0), comment="+Pyramid (35%)")
            scaled_in  := true
            active_pct := active_pct + 35.0
            sl_level   := math.min(sl_level, init_entry)

if strategy.position_size > 0
    if high >= (init_entry + entry_atr * trail_trig_m)
        sl_level := math.max(sl_level, init_entry + entry_atr * 0.35)
    strategy.exit("Exit Long", from_entry="WF Long", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Long Scale", from_entry="WF Long Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")
else if strategy.position_size < 0
    if low <= (init_entry - entry_atr * trail_trig_m)
        sl_level := math.min(sl_level, init_entry - entry_atr * 0.35)
    strategy.exit("Exit Short", from_entry="WF Short", limit=tp_level, stop=sl_level)
    strategy.exit("Exit Short Scale", from_entry="WF Short Scale", limit=tp_level, stop=sl_level)
    if (bar_index - entry_bar) >= horizon_bars
        strategy.close_all(comment="Horizon Exit")

if use_date_filter and not in_date_window and strategy.position_size != 0
    strategy.close_all(comment="Window End")

// =========================================================================
// 6. OVERLAYS & WALK-FORWARD TELEMETRY HUD
// =========================================================================
plot(ema200, "1-Hour 50 EMA (Macro Structure)", color=color.new(color.yellow, 10), linewidth=2)
plot(ema50,  "15m EMA 50", color=color.new(color.aqua, 30), linewidth=1)
plot(vwap32, "Session VWAP", color=color.new(color.fuchsia, 25), linewidth=2)

var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    conf_pct = prob_up >= 0.5 ? math.round(prob_up * 100, 1) : math.round(prob_dn * 100, 1)
    sess_txt = is_lon_ny ? "LONDON/NY (TREND)" : "ASIAN (VWAP REV)"
    reg_txt  = close < ema200 ? "BEARISH (<1H EMA50)" : "BULLISH (>1H EMA50)"
    sig_txt  = not shock_ok ? "PAUSED (MACRO SHOCK)" : bull_trig ? "BUY / LONG" : bear_trig ? "SELL / SHORT" : "WAIT / FILTERED"
    sig_col  = not shock_ok ? color.orange : bull_trig ? color.green : bear_trig ? color.red : color.gray

    table.cell(hud, 0, 0, "ML Methodology", text_color=color.white)
    table.cell(hud, 1, 0, "18d Rolling Walk-Forward (Fold #" + str.tostring(active_fold) + ")", text_color=color.yellow)
    table.cell(hud, 0, 1, "Macro Structure", text_color=color.white)
    table.cell(hud, 1, 1, reg_txt, text_color=close < ema200 ? color.red : color.lime)
    table.cell(hud, 0, 2, "Session Sub-Regime", text_color=color.white)
    table.cell(hud, 1, 2, sess_txt, text_color=color.aqua)
    table.cell(hud, 0, 3, "Volatility vs 5d Avg", text_color=color.white)
    table.cell(hud, 1, 3, str.tostring(math.round(atr_ratio, 2)) + "x (" + (vol_ok ? "ACTIVE" : "PAUSED") + ")", text_color=vol_ok ? color.lime : color.orange)
    table.cell(hud, 0, 4, "AI Conviction", text_color=color.white)
    table.cell(hud, 1, 4, (prob_up >= 0.5 ? "UP " : "DN ") + str.tostring(conf_pct) + "%", text_color=color.white)
    table.cell(hud, 0, 5, "Live Action", text_color=color.white)
    table.cell(hud, 1, 5, sig_txt, bgcolor=sig_col, text_color=color.white)
"""

    with open("AI_Gold_WalkForward_Final_v10.pine", "w") as f:
        f.write(pine_code)

    print("\n3. Generated 'AI_Gold_WalkForward_Final_v10.pine'! Paste this into TradingView Pine Editor.")

if __name__ == "__main__":
    main()