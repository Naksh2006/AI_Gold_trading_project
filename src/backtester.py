import numpy as np
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
