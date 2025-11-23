import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
LOOK_BACK = 60
TRAIN_WINDOW = 2000
INITIAL_TRAIN_RATIO = 0.6
RETRAIN_FREQUENCY = 100
NUM_EPOCHS = 200
LR = 0.01
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
SEED = 200
PRINT_EVERY_EPOCHS = 5
PRINT_TEST_STEP = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

# ===========================
# MODEL
# ===========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.bn(last)
        return self.fc(last)

# ===========================
# LOAD DATA
# ===========================

print(f"=== Loading {os.path.basename(FILE_PATH)} ===")
df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

feature_cols = ['log_ret', 'ema_slope_ratio_6_21', 'rsi_6','rsi_14','rsi_24',
                'stoch_k_14','stoch_d_14','trend_strength','price_zscore_20',
                'bb_position','range_position','macd_histogram']
print(f"Using {len(feature_cols)} features")

X_df = df[feature_cols].copy()
Y_df = df[['log_ret']].copy()

n = len(df)
split_idx = int(n * INITIAL_TRAIN_RATIO)

X_scaler = MinMaxScaler((-1, 1))
Y_scaler = MinMaxScaler((-1, 1))
model = None

# ===========================
# WALK-FORWARD TRAINING
# ===========================
print("--- Walk-forward training ---")
t0 = time.time()
for t_start in range(LOOK_BACK, split_idx - LOOK_BACK, RETRAIN_FREQUENCY):
    start_win = max(0, t_start - TRAIN_WINDOW)
    X_train_raw = X_df.iloc[start_win:t_start].values
    Y_train_raw = Y_df.iloc[start_win:t_start].values

    if len(X_train_raw) < LOOK_BACK + 1:
        print(f"  skip window ending at {t_start} (too small)")
        continue

    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    Y_train_scaled = Y_scaler.fit_transform(Y_train_raw)

    
    X_seq = np.stack([X_train_scaled[i:i+LOOK_BACK] for i in range(seq_count)], axis=0)
    y_seq = np.stack([Y_train_scaled[i+LOOK_BACK] for i in range(seq_count)], axis=0)

    X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_seq, dtype=torch.float32).to(device)

    model = LSTMModel(len(feature_cols), HIDDEN_DIM, NUM_LAYERS, 1, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
        # if epoch % PRINT_EVERY_EPOCHS == 0 or epoch == 1 or epoch == NUM_EPOCHS:
        #     print(f"  window_end={t_start:6d} | epoch {epoch:3d}/{NUM_EPOCHS} | loss={loss.item():.6e}")

    elapsed = time.time() - t0
    print(f"Trained window end {t_start}  — elapsed {elapsed:.1f}s | loss={loss.item():.6e}")

if model is None:
    raise RuntimeError("No trained model produced (insufficient data)")

# ===========================
# TESTING / INFERENCE
# ===========================
print("--- Testing ---")
test_slice = X_df.iloc[split_idx:]
history_needed = X_df.iloc[split_idx - LOOK_BACK: split_idx]
full_test = pd.concat([history_needed, test_slice])
test_scaled = X_scaler.transform(full_test.values)

pred_returns, base_prices, actual_prices, timestamps = [], [], [], []

model.eval()
with torch.no_grad():
    total_steps = len(test_scaled) - LOOK_BACK
    for i in range(LOOK_BACK, len(test_scaled)):
        seq = test_scaled[i-LOOK_BACK:i]
        x_in = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_in).cpu().numpy().ravel()[0]
        pred_returns.append(pred)

        curr_idx = split_idx + (i - LOOK_BACK)
        if curr_idx < len(df):
            base_prices.append(df.iloc[curr_idx-1]['close'])
            actual_prices.append(df.iloc[curr_idx]['close'])
            timestamps.append(df.index[curr_idx])

        step = i - LOOK_BACK + 1
        if step % PRINT_TEST_STEP == 0 or step == 1 or step == total_steps:
            print(f"  test step {step}/{total_steps}")

pred_np = np.array(pred_returns).reshape(-1, 1)
pred_real = Y_scaler.inverse_transform(pred_np).flatten()
actual_rets = np.log(np.array(actual_prices) / np.array(base_prices))
mse = mean_squared_error(actual_rets, pred_real)
print(f"Final MSE (log returns): {mse:.8f}")

plt.figure(figsize=(12, 6))
plt.plot(timestamps, actual_rets, label='Actual Returns', linewidth=1)
plt.plot(timestamps, pred_real, label='Predicted Returns', linewidth=1, alpha=0.8)
plt.title(f"LSTM Walk-Forward (MSE {mse:.8f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
