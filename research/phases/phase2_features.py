import pandas as pd
import numpy as np

def calculate_features(input_file="paxg_15m_1year.csv", output_file="paxg_features.csv"):
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # ---------------------------------------------------------
    # Helper Function: TradingView's exact RMA (Wilder's Smoothing)
    # ---------------------------------------------------------
    def pine_rma(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    # ---------------------------------------------------------
    # 1. Price Action & Momentum Features
    # ---------------------------------------------------------
    print("Calculating Price Action Features...")
    
    # Feature 1 & 2: Log Returns (1-candle and 4-candle/1-hour)
    df['f1_ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['f2_ret_4'] = np.log(df['close'] / df['close'].shift(4))
    
    # Feature 3: TradingView Exact RSI (14)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pine_rma(pd.Series(gain), 14)
    avg_loss = pine_rma(pd.Series(loss), 14)
    rs = avg_gain / (avg_loss + 1e-10) # Avoid division by zero
    df['f3_rsi'] = 100 - (100 / (1 + rs))
    # Scale RSI from [0, 100] to [-1, 1] for the neural network
    df['f3_rsi_scaled'] = (df['f3_rsi'] - 50) / 50.0 

    # Feature 4: Normalized MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    # Normalize by close price so it works across any price regime
    df['f4_macd_norm'] = (macd_hist / df['close']) * 1000 

    # Feature 5: Bollinger Band %B (20, 2)
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)
    df['f5_bb_pos'] = (df['close'] - lower_band) / (upper_band - lower_band + 1e-10)

    # ---------------------------------------------------------
    # 2. Volatility & Order Flow Features
    # ---------------------------------------------------------
    print("Calculating Volatility & Microstructure Features...")
    
    # Feature 6: Normalized ATR (14)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr14 = pine_rma(true_range, 14)
    df['f6_atr_norm'] = (atr14 / df['close']) * 100 

    # Feature 7: Distance from 50 EMA
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    df['f7_ema_dist'] = (df['close'] - ema50) / ema50

    # Feature 8: Order Flow (Taker Buy Ratio)
    # Measures aggressive market buying vs total volume. > 0.5 means buyers dominate.
    df['f8_buy_ratio'] = df['taker_buy_base_asset_volume'] / (df['volume'] + 1e-10)

    # ---------------------------------------------------------
    # 3. Session & Macro Time Features
    # ---------------------------------------------------------
    # Gold moves differently depending on which global banks are awake.
    # UTC hours: London opens ~08:00, NY opens ~13:00
    df['hour'] = df['datetime'].dt.hour
    df['f9_is_london'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
    df['f9_is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)

    # ---------------------------------------------------------
    # 4. TARGET LABEL (What we want to predict)
    # ---------------------------------------------------------
    print("Generating Target Labels...")
    # Target: Will the price be strictly HIGHER 4 candles (1 hour) from now?
    # shift(-4) looks 4 rows into the future. 
    df['future_return'] = (df['close'].shift(-4) - df['close']) / df['close']
    
    # Label 1 = UP, Label 0 = DOWN / FLAT
    # We require a minimum move of 0.05% to count as an "UP" to overcome trading fees
    fee_threshold = 0.0005 
    df['target'] = (df['future_return'] > fee_threshold).astype(int)

    # Clean up: Drop NaN values caused by rolling windows and future shifts
    df = df.dropna().reset_index(drop=True)
    
    # Select final columns for the ML model
    feature_cols = [
        'f1_ret_1', 'f2_ret_4', 'f3_rsi_scaled', 'f4_macd_norm', 
        'f5_bb_pos', 'f6_atr_norm', 'f7_ema_dist', 'f8_buy_ratio',
        'f9_is_london', 'f9_is_ny'
    ]
    
    final_df = df[['datetime', 'close'] + feature_cols + ['target']]
    
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Extracted {len(feature_cols)} features.")
    print(f"Dataset ready for ML: {output_file} ({len(final_df)} valid rows)")
    
    # Quick sanity check on class balance
    print("\nTarget Class Balance:")
    print(final_df['target'].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    calculate_features()