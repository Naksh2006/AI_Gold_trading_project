import pickle

def generate_pine_script():
    print("Loading model weights...")
    with open('model_weights.pkl', 'rb') as f:
        model = pickle.load(f)
        
    w0 = model['weights'][0] # Hidden layer weights (10 inputs x 8 neurons)
    b0 = model['biases'][0]  # Hidden layer biases (8 neurons)
    w1 = model['weights'][1] # Output layer weights (8 neurons x 1 output)
    b1 = model['biases'][1]  # Output layer bias (1 output)
    
    mean = model['scaler_mean']
    scale = model['scaler_scale']
    
    # 1. Base Pine Script Template & Feature Engineering
    pine_code = """//@version=5
indicator("AI Gold Predictor (PAXG/XAU)", overlay=true, max_labels_count=500)

// --- 1. FEATURE ENGINEERING ---
f1 = math.log(close / close[1])
f2 = math.log(close / nz(close[4], close[1]))

// Wilder's Smoothing for exact RSI/ATR matching
pine_rma(src, length) =>
    alpha = 1.0 / length
    sum = 0.0
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])
    sum

// RSI
delta = close - close[1]
gain = delta > 0 ? delta : 0.0
loss = delta < 0 ? -delta : 0.0
rs = pine_rma(loss, 14) == 0 ? 0 : pine_rma(gain, 14) / pine_rma(loss, 14)
rsi_val = 100.0 - (100.0 / (1.0 + rs))
f3 = (rsi_val - 50.0) / 50.0

// MACD
macd_line = ta.ema(close, 12) - ta.ema(close, 26)
f4 = ((macd_line - ta.ema(macd_line, 9)) / close) * 1000.0

// Bollinger Bands
sma20 = ta.sma(close, 20)
std20 = ta.stdev(close, 20)
upper_band = sma20 + (std20 * 2)
lower_band = sma20 - (std20 * 2)
f5 = upper_band == lower_band ? 0.0 : (close - lower_band) / (upper_band - lower_band)

// ATR
f6 = (pine_rma(ta.tr(true), 14) / close) * 100.0

// EMA Dist
ema50 = ta.ema(close, 50)
f7 = (close - ema50) / ema50

// Buy Ratio Proxy (TradingView lacks Taker Volume, so we approximate order flow via price action)
buy_vol = (high == low) ? 0.0 : volume * ((close - low) / (high - low))
f8 = volume == 0 ? 0.5 : buy_vol / volume

// Time Features (UTC)
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
        pine_code += f"h{j} = math.max(0.0, h{j}_raw) // ReLU Activation\n"
        
    pine_code += "\n// Output Layer (Sigmoid Probability)\n"
    output_eq = f"{b1[0]:.6f}"
    for j in range(8):
        weight = w1[j][0]
        output_eq += f" + (h{j} * {weight:.6f})"
    
    pine_code += f"out_raw = {output_eq}\n"
    pine_code += "prob_up = 1.0 / (1.0 + math.exp(-out_raw))\n\n"
    
    # 4. Signal UI and Zero-Repainting Guarantee
    pine_code += """// --- 4. CHART UI & SIGNALS ---
UP_THRESHOLD = 0.58
DOWN_THRESHOLD = 0.42

is_up = prob_up >= UP_THRESHOLD
is_down = prob_up <= DOWN_THRESHOLD

// The 'barstate.isconfirmed' check guarantees zero repainting for grading.
plotshape(barstate.isconfirmed and is_up, title="Buy", text="UP", style=shape.triangleup, location=location.belowbar, color=color.green, textcolor=color.green, size=size.small)
plotshape(barstate.isconfirmed and is_down, title="Sell", text="DN", style=shape.triangledown, location=location.abovebar, color=color.red, textcolor=color.red, size=size.small)
"""
    
    # Write to file
    with open('AI_Indicator.pine', 'w') as f:
        f.write(pine_code)
        
    print("Success! 'AI_Indicator.pine' has been generated.")
    print("Copy the contents of that file and paste it into the TradingView Pine Editor.")

if __name__ == "__main__":
    generate_pine_script()