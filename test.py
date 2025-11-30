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
LR = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
SEED = 200
PRINT_EVERY_EPOCHS = 5    # print epoch loss every N epochs
PRINT_TEST_STEP = 200     # print testing progress every N steps
VAL_RATIO = 0.1           # fraction of each training window used for validation

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
        # LayerNorm is safer than BatchNorm for variable / small batch sizes
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last = out[:, -1, :]            # (batch, hidden)
        last = self.norm(last)
        return self.fc(last)            # (batch, out_dim)

# ===========================
# LOAD DATA (top-level, no run())
# ===========================
if os.path.exists(FILE_PATH):
    print(f"=== Loading {os.path.basename(FILE_PATH)} ===")
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
else:
    print(f"File not found: {FILE_PATH} — generating dummy data.")
    dates = pd.date_range('2020-01-01', periods=5000, freq='H')
    df = pd.DataFrame({
        'close': np.random.lognormal(10, 0.02, 5000),
        'volume': np.random.randint(100, 1000, 5000)
    }, index=dates)
    df['open'] = df['high'] = df['low'] = df['close']

# basic cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Features (must exist in your csv). If not, script will error — ensure preprocessing created them.
feature_cols = [
    'log_ret', 'ema_slope_ratio_6_21', 'rsi_6', 'rsi_14', 'rsi_24',
    'stoch_k_14', 'stoch_d_14', 'trend_strength', 'price_zscore_20',
    'bb_position', 'range_position', 'macd_histogram'
]
print(f"Using {len(feature_cols)} features")

X_df = df[feature_cols].copy()
Y_df = df[['log_ret']].copy()

n = len(df)
split_idx = int(n * INITIAL_TRAIN_RATIO)

# scalers (will be refit every window; keep last fitted for test inverse transform)
X_scaler = MinMaxScaler((-1, 1))
Y_scaler = MinMaxScaler((-1, 1))
model = None

# Containers for Option B diagnostics
window_train_losses = []
window_val_losses = []

# ===========================
# WALK-FORWARD TRAINING
# ===========================
print("--- Walk-forward training ---")
t0 = time.time()
window_idx = 0
for t_start in range(LOOK_BACK, split_idx - LOOK_BACK, RETRAIN_FREQUENCY):
    start_win = max(0, t_start - TRAIN_WINDOW)
    X_train_raw = X_df.iloc[start_win:t_start].values
    Y_train_raw = Y_df.iloc[start_win:t_start].values

    if len(X_train_raw) < LOOK_BACK + 2:
        print(f"  skip window ending at {t_start} (too small: {len(X_train_raw)})")
        continue

    # fit scalers on train slice only
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    Y_train_scaled = Y_scaler.fit_transform(Y_train_raw)

    seq_count = len(X_train_scaled) - LOOK_BACK
    if seq_count <= 0:
        print(f"  skip window ending at {t_start} (seq_count={seq_count})")
        continue

    # build sequences
    X_seq = np.stack([X_train_scaled[i:i+LOOK_BACK] for i in range(seq_count)], axis=0)  # (seq_count, LOOK_BACK, feat)
    y_seq = np.stack([Y_train_scaled[i+LOOK_BACK] for i in range(seq_count)], axis=0)    # (seq_count, 1)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

    # train/val split (last VAL_RATIO fraction for validation)
    cut = int(len(X_tensor) * (1 - VAL_RATIO))
    if cut < 1:
        cut = max(1, len(X_tensor) - 1)  # ensure at least one sample in train if possible
    X_train_t, y_train_t = X_tensor[:cut], y_tensor[:cut]
    X_val_t, y_val_t = X_tensor[cut:], y_tensor[cut:]

    # init model for this window
    model = LSTMModel(len(feature_cols), HIDDEN_DIM, NUM_LAYERS, 1, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    final_train_loss = None
    final_val_loss = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = loss_fn(val_out, y_val_t)

        final_train_loss = loss.item()
        final_val_loss = val_loss.item()

        # if epoch % PRINT_EVERY_EPOCHS == 0 or epoch == 1 or epoch == NUM_EPOCHS:
        #     print(f"  window#{window_idx} end={t_start:6d} | epoch {epoch:3d}/{NUM_EPOCHS} | train={final_train_loss:.6e} | val={final_val_loss:.6e}")

    window_train_losses.append(final_train_loss)
    window_val_losses.append(final_val_loss)
    print(f"Finished window#{window_idx} @ {t_start}: train={final_train_loss:.6e}, val={final_val_loss:.6e}")

    window_idx += 1
    # optional quick time summary
    print(f"  elapsed {time.time() - t0:.1f}s")

if model is None:
    raise RuntimeError("No trained model produced (insufficient data)")

# ===========================
# Option B PLOT: per-window losses
# ===========================
if len(window_train_losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(window_train_losses, '-o', label="Train Loss per Window")
    plt.plot(window_val_losses, '-o', label="Validation Loss per Window")
    plt.xlabel("Retrain Window Index")
    plt.ylabel("MSE Loss")
    plt.title("Walk-Forward Window Training & Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No window losses collected to plot (window_train_losses empty).")

# ===========================
# TESTING / INFERENCE
# ===========================
print("--- Testing ---")
test_slice = X_df.iloc[split_idx:]
history_needed = X_df.iloc[split_idx - LOOK_BACK: split_idx]
full_test = pd.concat([history_needed, test_slice])
test_scaled = X_scaler.transform(full_test.values)  # use scaler from last trained window

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

# inverse transform preds and compute metrics
pred_np = np.array(pred_returns).reshape(-1, 1)
pred_real = Y_scaler.inverse_transform(pred_np).flatten()
actual_rets = np.log(np.array(actual_prices) / np.array(base_prices))

mse = mean_squared_error(actual_rets, pred_real)
print(f"Final MSE (log returns): {mse:.8f}")

# plot predictions vs actual returns
plt.figure(figsize=(12, 6))
plt.plot(timestamps, actual_rets, label='Actual Returns', linewidth=1)
plt.plot(timestamps, pred_real, label='Predicted Returns', linewidth=1, alpha=0.8)
plt.title(f"LSTM Walk-Forward (MSE {mse:.8f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
