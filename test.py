import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configuration
FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
LOOK_BACK = 60
TRAIN_WINDOW = 1000
INITIAL_TRAIN_RATIO = 0.6
RETRAIN_FREQUENCY = 100
NUM_EPOCHS = 100
LR = 0.001
HIDDEN_DIM = 16
NUM_LAYERS = 2
SEED = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
DROUPOUT = 0.2
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_test():
    if not os.path.exists(FILE_PATH): return print("File not found.")
    print(f"=== Processing {os.path.basename(FILE_PATH)} ===")
    
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    if 'returns' not in df.columns: df['returns'] = df['close'].pct_change().fillna(0)
    
    n = len(df)
    split_idx = int(n * INITIAL_TRAIN_RATIO)
    
    
    
    

    X_df = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'returns']) 
    Y_df = df['returns'].copy()


    

    
    pred_returns, base_prices, actual_prices, timestamps = [], [], [], []

    # --- Walk-forward on first 0.6 ---
    for t_start in range(LOOK_BACK, split_idx - LOOK_BACK, RETRAIN_FREQUENCY):
        start_win = max(0, t_start - TRAIN_WINDOW)
        X_train_raw = X_df.iloc[start_win:t_start].values
        Y_train_raw = Y_df.iloc[start_win:t_start].values.reshape(-1, 1)
        X_scaler = MinMaxScaler((-1, 1)).fit(X_train_raw)
        Y_scaler = MinMaxScaler((-1, 1)).fit(Y_train_raw)
        
        X_train_slice = X_scaler.transform(X_train_raw)
        Y_train_slice = Y_scaler.transform(Y_train_raw)   

        if len(X_train_slice) < LOOK_BACK:
            continue
        X_train_list, y_train = [], []
        for i in range(len(X_train_slice) - LOOK_BACK):
            X_train_list.append(X_train_slice[i:i+LOOK_BACK])
            y_train.append(Y_train_slice[i+LOOK_BACK])

        # FIX: Convert the list to a NumPy array before checking shape
        X_train_np = np.array(X_train_list)
        
        if X_train_np.shape[0] == 0:
             continue  # skip if nothing to train
             
        X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
        
        model = LSTM(X_tensor.shape[2], HIDDEN_DIM, NUM_LAYERS, 1 , DROUPOUT)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        print(f"Walk-forward step {t_start} | Train Loss: {loss.item():.6f}")

    # --- Predict remaining 0.4 using last trained model ---
    test_slice = X_df.iloc[split_idx:]
    for i in range(len(test_slice) - LOOK_BACK ):
        input_seq = test_slice.iloc[i:i+LOOK_BACK].values
        
        # FIX: Ensure model is defined if the training loop was skipped or used 'model'
        try:
             model.eval()
        except UnboundLocalError:
             print("Error: Model was not trained. Check initial data volume.")
             return

        test_seq = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            pred_ret = model(test_seq).item()
            pred_returns.append(pred_ret)

        prev_idx = split_idx + i + LOOK_BACK - 1
        curr_idx = prev_idx + 1
        base_prices.append(df.iloc[prev_idx]['close'])
        actual_prices.append(df.iloc[curr_idx]['close'])
        timestamps.append(df.index[curr_idx])

    # --- Evaluation ---
    pred_returns_np = np.array(pred_returns).reshape(-1, 1)
    
    # Use Y_scaler to get the real returns
    pred_rets_real = Y_scaler.inverse_transform(pred_returns_np).flatten()

    # Calculate actual returns using prices (as before, since Y_scaler can only inverse one column)
    actual_rets_real = (np.array(actual_prices) - np.array(base_prices)) / np.array(base_prices)
    mse = mean_squared_error(actual_rets_real, pred_rets_real)
    print(f"Final MSE (Returns): {mse:.8f}")

    plt.figure(figsize=(14, 7))
    
    # Plot Actual Returns
    plt.plot(timestamps, actual_rets_real, 
             label='Actual Returns', 
             color='black', 
             linewidth=1, 
             alpha=0.6)
    
    # Plot Predicted Returns
    plt.plot(timestamps, pred_rets_real, 
             label='Predicted Returns', 
             color='red', 
             linewidth=1, 
             alpha=0.8)
             
    plt.title(f"Predicted vs. Actual Returns (MSE: {mse:.8f})")
    plt.ylabel("Daily Return (Ratio)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='-') # Horizontal line at zero for visualization
    plt.show()

    # --- Optional: Zoomed Plot to see Directional Hits ---
    
    plt.figure(figsize=(14, 4))
    zoom_slice = slice(-100, None)
    
    plt.plot(timestamps[zoom_slice], actual_rets_real[zoom_slice], 
             label='Actual Returns', 
             color='black', 
             marker='.', 
             linestyle='-', 
             alpha=0.6)
             
    plt.plot(timestamps[zoom_slice], pred_rets_real[zoom_slice], 
             label='Predicted Returns', 
             color='red', 
             marker='x', 
             linestyle='--', 
             alpha=0.8)
             
    plt.title("Zoomed View: Last 100 Days (Directional Comparison)")
    plt.ylabel("Daily Return (Ratio)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='-')
    plt.show()
if __name__ == "__main__":
    run_test()