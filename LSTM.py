import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configuration
CLEAN_DIR = './cleaned'
LOOK_BACK = 60
INITIAL_TRAIN_RATIO = 0.7
NUM_EPOCHS = 2
LR = 0.01
HIDDEN_DIM = 32
NUM_LAYERS = 2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # PyTorch defaults h0, c0 to zeros if not provided
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

for fp in sorted(glob.glob(os.path.join(CLEAN_DIR, 'binance_*.csv'))):
    filename = os.path.basename(fp)
    print(f"\n=== Processing {filename} ===")
    
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    n = len(df)
    split_idx = int(n * INITIAL_TRAIN_RATIO)

    # Scaling
    scaler = MinMaxScaler((-1, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    data_vals = df_scaled.values
    
    preds, actuals = [], []

    # Walk-forward Validation
    for t in range(split_idx, n - LOOK_BACK):
        # Prepare window data
        window = data_vals[:t]
        X, y = [], []
        for i in range(len(window) - LOOK_BACK):
            X.append(window[i:i+LOOK_BACK-1])
            # Target is 'returns' column
            y.append(window[i+LOOK_BACK-1, df.columns.get_loc('returns')])
        
        X_train = torch.tensor(np.array(X), dtype=torch.float32)
        y_train = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

        # Init Model & Opt
        model = LSTM(X_train.shape[2], HIDDEN_DIM, NUM_LAYERS, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        # Train
        model.train()
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()
            
            # --- REQUESTED LOGGING ---
            print(f"File: {filename} | Step: {t}/{n-LOOK_BACK} | Epoch: {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.6f}", end='\r')

        # Predict
        model.eval()
        with torch.no_grad():
            test_seq = torch.tensor(data_vals[t:t+LOOK_BACK-1][np.newaxis, :, :], dtype=torch.float32)
            preds.append(model(test_seq).item())
            actuals.append(df_scaled.iloc[t+LOOK_BACK-1]['close']) # Tracking close price for metric

    # Inverse Transform logic for evaluation
    col_idx = df.columns.get_loc('close')
    dummy = np.zeros((len(preds), df.shape[1]))
    
    dummy[:, col_idx] = preds
    preds_inv = scaler.inverse_transform(dummy)[:, col_idx]
    
    dummy[:, col_idx] = actuals
    actuals_inv = scaler.inverse_transform(dummy)[:, col_idx]

    rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))
    print(f"\n{filename} RMSE: {rmse:.6f}")

    # Plot
    plt.figure(figsize=(10, 4))
    valid_idx = df.index[split_idx + LOOK_BACK - 1 : n]
    plt.plot(valid_idx, actuals_inv, label='Real Close')
    plt.plot(valid_idx, preds_inv, label='Predicted')
    plt.title(f"{filename} Walk-forward Results")
    plt.legend()
    plt.show()