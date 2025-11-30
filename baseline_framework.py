# baseline_framework.py
# Full integrated baseline framework for walk-forward evaluation and tuning

import os
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import optuna

# ==========================
# MODEL DEFINITIONS
# ==========================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, depth=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq, feat)
        x = x[:, -1, :]
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=(dropout if num_layers>1 else 0.0),
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.fc(last)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=(dropout if num_layers>1 else 0.0)
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(conv_channels, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1,2)  # (batch, feat, seq)
        x = self.conv(x)
        x = x.transpose(1,2)  # (batch, seq, channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==========================
# MODEL FACTORY
# ==========================

def build_model(model_type, input_dim, hidden_dim=64, num_layers=1, dropout=0.0, bidirectional=False, **kwargs):
    if model_type == 'mlp':
        return MLP(input_dim, hidden_dim, depth=kwargs.get('depth', 2))
    if model_type == 'lstm':
        return LSTMModel(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=False)
    if model_type == 'stacked_lstm':
        return LSTMModel(input_dim, hidden_dim, num_layers=max(2, num_layers), dropout=dropout, bidirectional=False)
    if model_type == 'bilstm':
        return LSTMModel(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
    if model_type == 'gru':
        return GRUModel(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_type == 'cnn_lstm':
        return CNNLSTM(input_dim, hidden_dim, conv_channels=kwargs.get('conv_channels', 32))
    raise ValueError('Unknown model type')


# ==========================
# WALK-FORWARD TRAINING (generic)
# ==========================

def walk_forward_train_generic(
    df,
    feature_cols,
    model_type,
    LOOK_BACK=60,
    TRAIN_WINDOW=2000,
    INITIAL_TRAIN_RATIO=0.6,
    RETRAIN_FREQUENCY=100,
    VAL_RATIO=0.1,
    NUM_EPOCHS=100,
    HIDDEN_DIM=64,
    NUM_LAYERS=1,
    DROPOUT=0.0,
    LR=1e-3,
    WEIGHT_DECAY=1e-5,
    SEED=200,
    device='cpu',
    trial=None,
    extra_kwargs=None
):
    """
    Generic walk-forward training that builds model with build_model(...).
    Includes Gradient Clipping for stability and Optuna reporting/pruning.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    extra_kwargs = extra_kwargs or {}

    X_df = df[feature_cols].copy()
    Y_df = df[['log_ret']].copy()

    n = len(df)
    split_idx = int(n * INITIAL_TRAIN_RATIO)

    X_scaler = MinMaxScaler((-1,1))
    Y_scaler = MinMaxScaler((-1,1))

    window_train_losses = []
    window_val_losses = []

    model = None
    
    # Define reporting interval (e.g., every 20% of epochs)
    report_interval = max(1, NUM_EPOCHS // 5)

    for window_idx, t_start in enumerate(range(LOOK_BACK, split_idx - LOOK_BACK, RETRAIN_FREQUENCY)):
        start_win = max(0, t_start - TRAIN_WINDOW)
        X_train_raw = X_df.iloc[start_win:t_start].values
        Y_train_raw = Y_df.iloc[start_win:t_start].values

        if len(X_train_raw) < LOOK_BACK + 2:
            continue

        X_train_scaled = X_scaler.fit_transform(X_train_raw)
        Y_train_scaled = Y_scaler.fit_transform(Y_train_raw)

        seq_count = len(X_train_scaled) - LOOK_BACK
        if seq_count <= 0:
            continue

        X_seq = np.stack([X_train_scaled[i:i+LOOK_BACK] for i in range(seq_count)], axis=0)
        y_seq = np.stack([Y_train_scaled[i+LOOK_BACK] for i in range(seq_count)], axis=0)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

        cut = int(len(X_tensor) * (1 - VAL_RATIO))
        if cut < 1:
            cut = max(1, len(X_tensor)-1)

        X_train_t, y_train_t = X_tensor[:cut], y_tensor[:cut]
        X_val_t, y_val_t = X_tensor[cut:], y_tensor[cut:]

        # Assuming build_model is defined elsewhere in your framework
        model = __import__('baseline_framework').build_model(
            model_type, input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, dropout=DROPOUT, **extra_kwargs
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        final_train_loss = None
        final_val_loss = None
        
        current_window_step_base = window_idx * NUM_EPOCHS

        for epoch in range(1, NUM_EPOCHS+1):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = loss_fn(out, y_train_t)
            loss.backward()
            
            # === CRITICAL FIX: GRADIENT CLIPPING to prevent INF loss ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ==========================================================

            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = loss_fn(val_out, y_val_t)

            final_train_loss = loss.item()
            final_val_loss = val_loss.item()
            
            # Report progress for Optuna pruning
            if trial is not None and epoch % report_interval == 0:
                step_val = current_window_step_base + epoch
                trial.report(final_val_loss, step=step_val)
                if trial.should_prune():
                    if optuna:
                        raise optuna.TrialPruned()
                    else:
                        raise Exception("Pruned")

        window_train_losses.append(final_train_loss)
        window_val_losses.append(final_val_loss)

        # Report at the end of window ONLY if it wasn't covered by the loop (Fix for reported value warnings)
        if trial is not None and (NUM_EPOCHS % report_interval != 0):
            step_val = (window_idx + 1) * NUM_EPOCHS
            trial.report(final_val_loss, step=step_val)
            if trial.should_prune():
                if optuna:
                    raise optuna.TrialPruned()
                else:
                    raise Exception("Pruned")

    if model is None:
        raise RuntimeError('No trained model produced (insufficient data)')

    return model, X_scaler, Y_scaler, window_train_losses, window_val_losses, split_idx

# ==========================
# EVALUATION / TESTING
# ==========================

def evaluate_model(model, df, X_scaler, Y_scaler, feature_cols, LOOK_BACK, split_idx, device='cpu'):
    test_slice = df.iloc[split_idx:]
    history_needed = df.iloc[split_idx - LOOK_BACK: split_idx]
    full_test = pd.concat([history_needed[feature_cols], test_slice[feature_cols]])
    test_scaled = X_scaler.transform(full_test.values)

    pred_returns = []
    base_prices = []
    actual_prices = []
    timestamps = []

    model.eval()
    with torch.no_grad():
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

    pred_np = np.array(pred_returns).reshape(-1,1)
    if len(pred_np) == 0:
        return float('inf'), timestamps, np.array(actual_prices), np.array(pred_returns)

    pred_real = Y_scaler.inverse_transform(pred_np).flatten()
    actual_rets = np.log(np.array(actual_prices) / np.array(base_prices))

    L = min(len(actual_rets), len(pred_real))
    mse = mean_squared_error(actual_rets[:L], pred_real[:L])
    return mse, timestamps[:L], actual_rets[:L], pred_real[:L]


# ==========================
# SAVE / LOAD HELPERS
# ==========================

def save_trial_artifacts(base_dir, trial_number, model, X_scaler, Y_scaler, summary):
    trial_dir = os.path.join(base_dir, f'trial_{trial_number}')
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    try:
        torch.save(model.state_dict(), os.path.join(trial_dir, 'model.pt'))
        joblib.dump(X_scaler, os.path.join(trial_dir, 'X_scaler.pkl'))
        joblib.dump(Y_scaler, os.path.join(trial_dir, 'Y_scaler.pkl'))
    except Exception as e:
        print('Warning saving artifacts:', e)


def copy_best_artifacts(src_trial_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in ['model.pt', 'X_scaler.pkl', 'Y_scaler.pkl', 'summary.json']:
        s = os.path.join(src_trial_dir, fname)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dest_dir, fname))


# ==========================
# BATCH EVALUATION (simple runner)
# ==========================

def evaluate_models_list(models_list, df, feature_cols, train_params, eval_params, save_dir='baseline_results'):
    """
    models_list: list of model_type strings
    train_params: dict of training params (LOOK_BACK, TRAIN_WINDOW, ... default values applied)
    eval_params: dict with device etc.
    Saves results per model in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    summary = {}

    for m in models_list:
        print(f'-- Evaluating model: {m} --')
        try:
            model, X_scaler, Y_scaler, train_losses, val_losses, split_idx = walk_forward_train_generic(
                df=df,
                feature_cols=feature_cols,
                model_type=m,
                device=eval_params.get('device','cpu'),
                **train_params
            )
        except Exception as e:
            print('Training failed for', m, 'error:', e)
            summary[m] = {'error': str(e)}
            continue

        mse, timestamps, actual_rets, pred_rets = evaluate_model(
            model, df, X_scaler, Y_scaler, feature_cols, LOOK_BACK=train_params.get('LOOK_BACK',60), split_idx=split_idx, device=eval_params.get('device','cpu')
        )

        model_dir = os.path.join(save_dir, m)
        os.makedirs(model_dir, exist_ok=True)
        # save artifacts
        summary[m] = {
            'mse': float(mse),
            'train_losses_per_window': train_losses,
            'val_losses_per_window': val_losses,
            'n_windows': len(train_losses)
        }
        save_trial_artifacts(save_dir, m, model, X_scaler, Y_scaler, summary[m])

        # also save predictions
        np.save(os.path.join(model_dir, 'preds.npy'), pred_rets)
        np.save(os.path.join(model_dir, 'actuals.npy'), actual_rets)

        print(f"Saved results for {m} to {model_dir} (MSE={mse:.6e})")

    # write overall summary
    with open(os.path.join(save_dir, 'summary_all.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
