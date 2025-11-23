import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
LOOK_BACK = 60           # Hours of history to look at
TRAIN_WINDOW = 2000      # Sliding window size for training
INITIAL_TRAIN_RATIO = 0.6
RETRAIN_FREQUENCY = 100  # How often to retrain (e.g., every 100 hours)
NUM_EPOCHS = 50
LR = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
SEED = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 2. FEATURE ENGINEERING (IMPROVED)
# ==========================================
def add_improved_features(df):
    """
    Generates robust, stationary features for Crypto LSTM.
    Avoids raw prices to prevent scaling issues during bull/bear runs.
    """
    df = df.copy()
    
    # A. Log Returns (The Target & Momentum)
    # Measures the speed of price change
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # B. SMA Distance (Stationary Trend)
    # Instead of raw price ($60k), we use % distance from 20-hour average.
    # This works the same whether BTC is $100 or $100k.
    sma = df['close'].rolling(window=20).mean()
    df['sma_dist'] = (df['close'] - sma) / sma
    
    # C. Volatility Squeeze (Bollinger Width)
    # Low values imply a breakout is coming. High values imply chaos.
    rolling_std = df['close'].rolling(window=20).std()
    upper = sma + (rolling_std * 2)
    lower = sma - (rolling_std * 2)
    # Normalized width
    df['bb_width'] = (upper - lower) / sma
    
    # D. Volume Force (Log Volume Change)
    # "Fuel" for the move. We use log change to dampen massive spikes.
    # Replace 0 volume with 1 to avoid log(0) errors
    vol = df['volume'].replace(0, 1)
    df['vol_change'] = np.log(vol / vol.shift(1))
    
    # E. RSI (Relative Strength Index)
    # Standard momentum oscillator (0-100), normalized to 0-1
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100.0 # Normalize immediately to 0-1 range
    
    # F. Time Encoding (Cyclical)
    # Helps model learn daily patterns (e.g., Asian Open vs US Close)
    # We use sin/cos so Hour 23 is mathematically close to Hour 0
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Drop NaNs created by rolling windows (first ~20 rows)
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)                 
        # Take the last time step output
        last = out[:, -1, :]                 
        last = self.bn(last)                 
        return self.fc(last)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def run_test():
    # --- Load Data or Generate Dummy ---
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found. Generating dummy data for testing...")
        dates = pd.date_range('2020-01-01', periods=5000, freq='H')
        df = pd.DataFrame({
            'close': np.random.lognormal(10, 0.02, 5000), 
            'volume': np.random.randint(100, 1000, 5000)
        }, index=dates)
        df['open'] = df['close']; df['high'] = df['close']; df['low'] = df['close']
    else:
        print(f"=== Processing {os.path.basename(FILE_PATH)} ===")
        df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    
    # Basic Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # --- APPLY IMPROVED FEATURES ---
    df = add_improved_features(df)
    
    # Define Columns
    # Target: log_ret
    # Features: log_ret + improved indicators
    feature_cols = ['log_ret', 'sma_dist', 'bb_width', 'vol_change', 'rsi', 'hour_sin', 'hour_cos']
    
    print(f"Using {len(feature_cols)} Features: {feature_cols}")
    
    # Split DataFrames
    X_df = df[feature_cols].copy()
    Y_df = df[['log_ret']].copy()

    n = len(df)
    split_idx = int(n * INITIAL_TRAIN_RATIO)
    
    # Result Containers
    pred_returns, base_prices, actual_prices, timestamps = [], [], [], []
    
    # Scalers (initialized outside)
    X_scaler = MinMaxScaler((-1, 1))
    Y_scaler = MinMaxScaler((-1, 1))
    model = None

    # --- TRAIN LOOP (Walk-Forward) ---
    print("--- Starting Walk-Forward Training ---")
    
    for t_start in range(LOOK_BACK, split_idx - LOOK_BACK, RETRAIN_FREQUENCY):
        start_win = max(0, t_start - TRAIN_WINDOW)
        
        # Slice
        X_train_raw = X_df.iloc[start_win:t_start].values
        Y_train_raw = Y_df.iloc[start_win:t_start].values
        
        # Fit Scaler on TRAIN slice only
        X_train_slice = X_scaler.fit_transform(X_train_raw)
        Y_train_slice = Y_scaler.fit_transform(Y_train_raw)
        
        if len(X_train_slice) < LOOK_BACK: continue

        # Sequence Generation
        X_train_list, y_train_list = [], []
        for i in range(len(X_train_slice) - LOOK_BACK):
            X_train_list.append(X_train_slice[i:i+LOOK_BACK])
            y_train_list.append(Y_train_slice[i+LOOK_BACK]) # Predicting next step
            
        X_train_np = np.array(X_train_list)
        y_train_np = np.array(y_train_list)
        
        if len(X_train_np) == 0: continue
            
        # Tensor conversion
        X_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
        
        # Model Init (Dynamic Input Dim based on features)
        input_dim = len(feature_cols)
        model = LSTM(input_dim, HIDDEN_DIM, NUM_LAYERS, 1, DROPOUT).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            out = model(X_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()
            
        if t_start % 500 == 0:
            print(f"Step {t_start}/{split_idx} | Train Loss: {loss.item():.6f}")

    # --- TEST LOOP ---
    print("--- Starting Testing ---")
    if model is None: return

    # Prepare Test Data
    # We need a small history buffer before split_idx to form the first test sequence
    test_slice = X_df.iloc[split_idx:]
    history_needed = X_df.iloc[split_idx - LOOK_BACK : split_idx]
    full_test_input = pd.concat([history_needed, test_slice])
    
    # CRITICAL: Transform using the scaler fitted on the LAST training window
    # Do NOT fit_transform here, or you leak future info.
    test_data_scaled = X_scaler.transform(full_test_input.values)
    
    model.eval()
    with torch.no_grad():
        for i in range(LOOK_BACK, len(test_data_scaled)):
            # Sequence
            seq = test_data_scaled[i-LOOK_BACK : i]
            input_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            pred_val = model(input_seq).item()
            pred_returns.append(pred_val)
            
            # Reconstruct Index
            curr_idx = split_idx + (i - LOOK_BACK)
            if curr_idx < len(df):
                base_prices.append(df.iloc[curr_idx-1]['close'])
                actual_prices.append(df.iloc[curr_idx]['close'])
                timestamps.append(df.index[curr_idx])

    # --- EVALUATION ---
    # Inverse Transform Predictions
    pred_returns_np = np.array(pred_returns).reshape(-1, 1)
    pred_rets_real = Y_scaler.inverse_transform(pred_returns_np).flatten()
    
    # Calculate Actual Returns
    actual_rets_real = np.log(np.array(actual_prices) / np.array(base_prices))
    
    # MSE
    mse = mean_squared_error(actual_rets_real, pred_rets_real)
    print(f"Final MSE (Log Returns): {mse:.8f}")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, actual_rets_real, label='Actual Returns', color='black', alpha=0.5, linewidth=1)
    plt.plot(timestamps, pred_rets_real, label='Predicted Returns', color='red', alpha=0.7, linewidth=1)
    plt.title(f"LSTM Walk-Forward Prediction (Improved Features)\nMSE: {mse:.8f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_test()