import pickle

def generate_pine_strategy():
    print("Loading model weights...")
    with open('model_weights.pkl', 'rb') as f:
        model = pickle.load(f)
        
    w0 = model['weights'][0] 
    b0 = model['biases'][0]  
    w1 = model['weights'][1] 
    b1 = model['biases'][1]  
    mean = model['scaler_mean']
    scale = model['scaler_scale']
    
    # 1. Base Pine Script Strategy Template (Includes 0.05% Fees)
    pine_code = """//@version=5
strategy("AI Gold Backtest (PAXG/XAU)", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.05)

// --- 1. FEATURE ENGINEERING ---
f1 = math.log(close / close[1])
f2 = math.log(close / nz(close[4], close[1]))

pine_rma(src, length) =>
    alpha = 1.0 / length
    sum = 0.0
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])
    sum

delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs = pine_rma(loss, 14) == 0 ? 0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi_val = 100.0 - (100.0 / (1.0 + rs))
f3 = (rsi_val - 50.0) / 50.0

macd_line = ta.ema(close, 12) - ta.ema(close, 26)
f4 = ((macd_line - ta.ema(macd_line, 9)) / close) * 1000.0

sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20)
upper_band = sma20 + (std20 * 2)
lower_band = sma20 - (std20 * 2)
f5 = upper_band == lower_band ? 0.0 : (close - lower_band) / (upper_band - lower_band)

f6 = (pine_rma(ta.tr(true), 14) / close) * 100.0
ema50 = ta.ema(close, 50)
f7 = (close - ema50) / ema50

buy_vol = (high == low) ? 0.0 : volume * ((close - low) / (high - low))
f8 = volume == 0 ? 0.5 : buy_vol / volume

h = hour(time)
f9 = (h >= 8 and h <= 16) ? 1.0 : 0.0
f10 = (h >= 13 and h <= 21) ? 1.0 : 0.0

// --- 2. MODEL SCALING ---
"""
    
    # 2. Inject StandardScaler parameters
    for i in range(10):
        pine_code += f"s{i+1} = (f{i+1} - {mean[i]}) / {scale[i]}\n"
        
    pine_code += "\n// --- 3. NEURAL NETWORK FORWARD PASS ---\n"
    
    # 3. Inject Hidden Layer (ReLU)
    for j in range(8):
        equation = f"{b0[j]:.6f}"
        for i in range(10):
            weight = w0[i][j]
            equation += f" + (s{i+1} * {weight:.6f})"
        pine_code += f"h{j}_raw = {equation}\n"
        pine_code += f"h{j} = math.max(0.0, h{j}_raw)\n"
        
    pine_code += "\n// Output Layer\n"
    output_eq = f"{b1[0]:.6f}"
    for j in range(8):
        weight = w1[j][0]
        output_eq += f" + (h{j} * {weight:.6f})"
    
    pine_code += f"out_raw = {output_eq}\n"
    pine_code += "prob_up = 1.0 / (1.0 + math.exp(-out_raw))\n\n"
    
    # 4. Trading Logic for Strategy Tester
    pine_code += """// --- 4. BACKTEST EXECUTION ---
UP_THRESHOLD = 0.65
DOWN_THRESHOLD = 0.70

is_up = prob_up >= UP_THRESHOLD
is_down = prob_up <= DOWN_THRESHOLD

// Track how many bars we have been in a trade
var int bars_in_trade = 0
if strategy.position_size != 0
    bars_in_trade += 1
else
    bars_in_trade := 0

// Execution Logic
if barstate.isconfirmed
    if is_up
        strategy.entry("Long", strategy.long)
    if is_down
        strategy.entry("Short", strategy.short)

// Exit after our 4-candle training horizon
if bars_in_trade >= 4
    strategy.close_all(comment="Time Exit")
"""
    
    with open('AI_Strategy.pine', 'w') as f:
        f.write(pine_code)
        
    print("Success! 'AI_Strategy.pine' has been generated.")

if __name__ == "__main__":
    generate_pine_strategy()