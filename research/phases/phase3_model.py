import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train_walk_forward_model(input_file="paxg_features.csv"):
    print(f"Loading engineered dataset from {input_file}...\n")
    df = pd.read_csv(input_file)
    
    # ---------------------------------------------------------
    # 1. Chronological Split (NO RANDOM SHUFFLING)
    # ---------------------------------------------------------
    # 70% Train, 15% Validation, 15% Out-of-Sample Test
    train_idx = int(len(df) * 0.70)
    val_idx = int(len(df) * 0.85)
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    # Define our feature columns
    feature_cols = [
        'f1_ret_1', 'f2_ret_4', 'f3_rsi_scaled', 'f4_macd_norm', 
        'f5_bb_pos', 'f6_atr_norm', 'f7_ema_dist', 'f8_buy_ratio',
        'f9_is_london', 'f9_is_ny'
    ]
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_val, y_val = val_df[feature_cols], val_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    print(f"Train size: {len(X_train)} candles")
    print(f"Val size:   {len(X_val)} candles")
    print(f"Test size:  {len(X_test)} candles\n")

    # ---------------------------------------------------------
    # 2. Feature Scaling (Crucial for Neural Networks)
    # ---------------------------------------------------------
    # We strictly FIT the scaler on the Training set only to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # 3. Train the Lightweight Neural Network
    # ---------------------------------------------------------
    # Architecture: 1 hidden layer with 8 neurons. 
    # This keeps the weight matrices small enough to hardcode into Pine Script later.
    print("Training Neural Network (MLPClassifier)...")
    model = MLPClassifier(
        hidden_layer_sizes=(8,), 
        activation='relu', 
        solver='adam', 
        max_iter=500,
        early_stopping=True,     # Use validation set to stop training before overfitting
        validation_fraction=0.1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    print("Training complete!\n")
    
    # ---------------------------------------------------------
    # 4. Confidence Filtering & Live Simulation (The 40-Mark Requirement)
    # ---------------------------------------------------------
    # Get raw probabilities on the unseen Out-of-Sample Test set
    y_test_probs = model.predict_proba(X_test_scaled)
    
    # y_test_probs[:, 1] is the probability of class 1 (UP)
    prob_up = y_test_probs[:, 1]
    
    # Define our thresholds
    # If P(UP) > 0.58 -> Strong Buy
    # If P(UP) < 0.42 -> Strong Sell (meaning P(DOWN) > 0.58)
    # Everything in between is "No Signal" (Chop)
    UP_THRESHOLD = 0.58
    DOWN_THRESHOLD = 0.42
    
    test_df = test_df.copy()
    test_df['prob_up'] = prob_up
    
    # Generate Trading Signals
    conditions = [
        (test_df['prob_up'] >= UP_THRESHOLD),
        (test_df['prob_up'] <= DOWN_THRESHOLD)
    ]
    choices = [1, 0] # 1 = Predict UP, 0 = Predict DOWN
    test_df['signal'] = np.select(conditions, choices, default=-1)
    
    # Filter only the rows where the model was confident
    confident_trades = test_df[test_df['signal'] != -1]
    
    total_candles = len(test_df)
    total_signals = len(confident_trades)
    
    if total_signals > 0:
        confident_accuracy = accuracy_score(confident_trades['target'], confident_trades['signal'])
    else:
        confident_accuracy = 0.0
        
    print("--- OUT-OF-SAMPLE TEST RESULTS (Live Simulation) ---")
    print(f"Total Evaluation Candles: {total_candles}")
    print(f"Total Confident Signals Fired: {total_signals} ({round((total_signals/total_candles)*100, 1)}% of time)")
    print(f"Accuracy of Confident Signals: {round(confident_accuracy * 100, 2)}%\n")
    
    # ---------------------------------------------------------
    # 5. Save the Weights & Scaler for Pine Script compilation
    # ---------------------------------------------------------
    # We save these locally so Phase 4 can read them and generate the Pine Code
    model_data = {
        'weights': model.coefs_,
        'biases': model.intercepts_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': feature_cols
    }
    
    with open('model_weights.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model weights and scaler parameters saved to 'model_weights.pkl'.")
    print("Ready for Phase 4: Pine Script Compilation!")

if __name__ == "__main__":
    train_walk_forward_model()