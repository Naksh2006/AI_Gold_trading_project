def compile_to_pine(artifacts, cfg, output_path="pine/AI_Gold_Quant_Strategy.pine"):
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
        pine_code += f"z{j} = {eq}\n"
        pine_code += f"h{j} = (math.exp(2.0 * z{j}) - 1.0) / (math.exp(2.0 * z{j}) + 1.0)\n"

    mlp_out = f"{b1[0]:.8f}"
    for j in range(len(b0)):
        mlp_out += f" + (h{j} * {w1[j][0]:.8f})"
    pine_code += f"logit_mlp = {mlp_out}\n"
    pine_code += "p_mlp = 1.0 / (1.0 + math.exp(-logit_mlp))\n\n"

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
