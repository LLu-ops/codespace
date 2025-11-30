

import os

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna


FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
FEATURE_COLS = [
    'log_ret','ema_slope_ratio_6_21','rsi_6','rsi_14','rsi_24',
    'stoch_k_14','stoch_d_14','trend_strength','price_zscore_20',
    'bb_position','range_position','macd_histogram'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED_BASE = 200
RESULTS_DIR = "optuna_tuning_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Defaults that Optuna can override
LOOK_BACK = 60
TRAIN_WINDOW = 2000
INITIAL_TRAIN_RATIO = 0.6
VAL_RATIO = 0.1

# Optuna study settings
STORAGE_URL = f"sqlite:///{os.path.join(RESULTS_DIR, 'optuna_study.db')}"
STUDY_NAME = "lstm_walkforward_tuning"
N_TRIALS = 20  # change to a larger number when you have more compute
N_JOBS = 1     # parallel trials (keep 1 for deterministic GPU usage)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.fc(last)

# ===========================
# DATA LOADING (top-level)
# ===========================
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
else:
    # dummy data
    dates = pd.date_range('2020-01-01', periods=5000, freq='H')
    df = pd.DataFrame({
        'close': np.random.lognormal(10, 0.02, 5000),
        'volume': np.random.randint(100, 1000, 5000)
    }, index=dates)
    df['open'] = df['high'] = df['low'] = df['close']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# check features exist
for c in FEATURE_COLS:
    if c not in df.columns:
        raise RuntimeError(f"Feature column missing: {c}")

# ===========================
# WALK-FORWARD TRAINING (supports OPTUNA pruning)
# ===========================
def walk_forward_train(
    df,
    feature_cols,
    LOOK_BACK,
    TRAIN_WINDOW,
    INITIAL_TRAIN_RATIO,
    RETRAIN_FREQUENCY,
    VAL_RATIO,
    NUM_EPOCHS,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    LR,
    WEIGHT_DECAY,
    seed,
    trial=None,                   # optuna.trial.Trial or None
    device=DEVICE
):
    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_df = df[feature_cols].copy()
    Y_df = df[['log_ret']].copy()

    n = len(df)
    split_idx = int(n * INITIAL_TRAIN_RATIO)

    X_scaler = MinMaxScaler((-1, 1))
    Y_scaler = MinMaxScaler((-1, 1))

    window_train_losses = []
    window_val_losses = []

    model = None

    # walk forward windows
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
            cut = max(1, len(X_tensor) - 1)

        X_train_t, y_train_t = X_tensor[:cut], y_tensor[:cut]
        X_val_t, y_val_t = X_tensor[cut:], y_tensor[cut:]

        model = LSTMModel(len(feature_cols), HIDDEN_DIM, NUM_LAYERS, 1, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        final_train_loss = None
        final_val_loss = None

        # training loop for this window
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = loss_fn(out, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = loss_fn(val_out, y_val_t)

            final_train_loss = loss.item()
            final_val_loss = val_loss.item()

            # Optional: report epoch-level intermediate values to Optuna (coarse)
            if trial is not None and epoch % max(1, NUM_EPOCHS // 5) == 0:
                # report the current validation loss (smaller is better)
                trial.report(final_val_loss, step=window_idx * NUM_EPOCHS + epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        window_train_losses.append(final_train_loss)
        window_val_losses.append(final_val_loss)

        # Report per-window value to Optuna (useful for window-level pruning)
        if trial is not None:
            trial.report(final_val_loss, step= (window_idx+1) * NUM_EPOCHS)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if model is None:
        raise RuntimeError("No trained model produced (insufficient data)")

    return model, X_scaler, Y_scaler, window_train_losses, window_val_losses, split_idx

# ===========================
# TEST / EVALUATION
# ===========================
def evaluate_model(model, df, X_scaler, Y_scaler, feature_cols, LOOK_BACK, split_idx, device=DEVICE):
    test_slice = df.iloc[split_idx:]
    history_needed = df.iloc[split_idx - LOOK_BACK: split_idx]
    full_test = pd.concat([history_needed[feature_cols], test_slice[feature_cols]])
    test_scaled = X_scaler.transform(full_test.values)  # use scaler from last trained window

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

    pred_np = np.array(pred_returns).reshape(-1, 1)
    if len(pred_np) == 0:
        return float('inf'), timestamps, np.array(actual_prices), np.array(pred_returns)

    pred_real = Y_scaler.inverse_transform(pred_np).flatten()
    actual_rets = np.log(np.array(actual_prices) / np.array(base_prices))

    # align lengths if needed
    L = min(len(actual_rets), len(pred_real))
    mse = mean_squared_error(actual_rets[:L], pred_real[:L])
    return mse, timestamps[:L], actual_rets[:L], pred_real[:L]

# ===========================
# OPTUNA OBJECTIVE
# ===========================
def objective(trial):
    # sample hyperparameters
    HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    NUM_LAYERS = trial.suggest_int("num_layers", 1, 3)
    DROPOUT = trial.suggest_float("dropout", 0.0, 0.5)
    LR = trial.suggest_loguniform("lr", 1e-5, 5e-3)
    WEIGHT_DECAY = trial.suggest_loguniform("weight_decay", 1e-7, 1e-4)
    NUM_EPOCHS = trial.suggest_categorical("num_epochs", [50, 100, 150])
    RETRAIN_FREQUENCY = trial.suggest_categorical("retrain_frequency", [50, 100, 200])
    seed = SEED_BASE + trial.number

    # call walk-forward train with trial for pruning
    try:
        model, X_scaler, Y_scaler, train_losses, val_losses, split_idx = walk_forward_train(
            df=df,
            feature_cols=FEATURE_COLS,
            LOOK_BACK=LOOK_BACK,
            TRAIN_WINDOW=TRAIN_WINDOW,
            INITIAL_TRAIN_RATIO=INITIAL_TRAIN_RATIO,
            RETRAIN_FREQUENCY=RETRAIN_FREQUENCY,
            VAL_RATIO=VAL_RATIO,
            NUM_EPOCHS=NUM_EPOCHS,
            HIDDEN_DIM=HIDDEN_DIM,
            NUM_LAYERS=NUM_LAYERS,
            DROPOUT=DROPOUT,
            LR=LR,
            WEIGHT_DECAY=WEIGHT_DECAY,
            seed=seed,
            trial=trial,
            device=DEVICE
        )
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        # if training fails for some combination, report large value
        print(f"Trial {trial.number} failed during training: {e}")
        return float('inf')

    mse, timestamps, actual_rets, pred_rets = evaluate_model(
        model, df, X_scaler, Y_scaler, FEATURE_COLS, LOOK_BACK, split_idx, device=DEVICE
    )

    # save trial artifacts
    trial_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # save small summary JSON
    summary = {
        "trial_number": trial.number,
        "params": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "retrain_frequency": RETRAIN_FREQUENCY
        },
        "train_losses_per_window": train_losses,
        "val_losses_per_window": val_losses,
        "test_mse": float(mse)
    }
    with open(os.path.join(trial_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save model state + scalers
    try:
        torch.save(model.state_dict(), os.path.join(trial_dir, "model.pt"))
        joblib.dump(X_scaler, os.path.join(trial_dir, "X_scaler.pkl"))
        joblib.dump(Y_scaler, os.path.join(trial_dir, "Y_scaler.pkl"))
    except Exception as e:
        print(f"Warning: failed to save artifacts for trial {trial.number}: {e}")

    # report final objective
    return float(mse)

# ===========================
# RUN STUDY
# ===========================
if __name__ == "__main__":
    # create/load study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    print(f"Starting optimization: study_name={STUDY_NAME}, storage={STORAGE_URL}")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # save study results summary
    best = study.best_trial
    best_params = best.params if best is not None else {}
    best_value = best.value if best is not None else None

    summary_all = {
        "best_value": float(best_value) if best_value is not None else None,
        "best_params": best_params,
        "n_trials": len(study.trials)
    }
    with open(os.path.join(RESULTS_DIR, "study_summary.json"), "w") as f:
        json.dump(summary_all, f, indent=2)

    print("Optimization finished.")
    print("Best MSE:", summary_all["best_value"])
    print("Best params:", summary_all["best_params"])

    # Save best model artifacts to central location
    if best is not None:
        best_trial_dir = os.path.join(RESULTS_DIR, f"trial_{best.number}")
        central_model = os.path.join(RESULTS_DIR, "best_model.pt")
        central_xsc = os.path.join(RESULTS_DIR, "best_X_scaler.pkl")
        central_ysc = os.path.join(RESULTS_DIR, "best_Y_scaler.pkl")
        try:
            if os.path.exists(os.path.join(best_trial_dir, "model.pt")):
                joblib.copy = None  # silence linters about joblib.copy not existing
                # use simple copy
                import shutil
                shutil.copy(os.path.join(best_trial_dir, "model.pt"), central_model)
                if os.path.exists(os.path.join(best_trial_dir, "X_scaler.pkl")):
                    shutil.copy(os.path.join(best_trial_dir, "X_scaler.pkl"), central_xsc)
                if os.path.exists(os.path.join(best_trial_dir, "Y_scaler.pkl")):
                    shutil.copy(os.path.join(best_trial_dir, "Y_scaler.pkl"), central_ysc)
                print(f"Best model/artifacts copied to {RESULTS_DIR}")
        except Exception as e:
            print("Warning copying best artifacts:", e)

    # optional: print top-5 trials
    print("\nTop-5 trials by value:")
    for t in sorted(study.trials, key=lambda tr: (tr.value if tr.value is not None else float('inf')))[:5]:
        print(f"  #{t.number} value={t.value} params={t.params}")
