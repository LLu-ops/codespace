import os
import json
import optuna
import numpy as np
import pandas as pd
import torch



from baseline_framework import walk_forward_train_generic, evaluate_model, save_trial_artifacts, build_model

FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
FEATURE_COLS = [
    'log_ret','ema_slope_ratio_6_21','rsi_6','rsi_14','rsi_24',
    'stoch_k_14','stoch_d_14','trend_strength','price_zscore_20',
    'bb_position','range_position','macd_histogram'
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = 'model_tuning_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
df.replace([np.inf,-np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

models_to_tune = ['lstm']
N_TRIALS = 20



def get_model_hyperparameters(trial, model_type):
    """Suggests model-specific hyperparameters for the given trial."""
    params = {}
    
    # Common Parameters
    params['LOOK_BACK'] = trial.suggest_categorical('LOOK_BACK', [30, 60])
    params['TRAIN_WINDOW'] = trial.suggest_categorical('TRAIN_WINDOW', [1000, 2000])
    params['RETRAIN_FREQUENCY'] = trial.suggest_categorical('RETRAIN_FREQUENCY', [50, 100])
    params['HIDDEN_DIM'] = trial.suggest_categorical('hidden', [32, 64, 128])
    params['NUM_EPOCHS'] = trial.suggest_categorical('epochs', [50, 100])
    
    # Reduced max LR to 5e-4 for stability with LSTMs
    params['LR'] = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    params['WEIGHT_DECAY'] = trial.suggest_float('wd', 1e-7, 1e-4, log=True)

    params['EXTRA_KWARGS'] = {}
    params['NUM_LAYERS'] = 1
    params['DROPOUT'] = 0.0

    if model_type == 'mlp':
        params['DEPTH'] = trial.suggest_int('depth', 1, 3)
        params['EXTRA_KWARGS']['depth'] = params['DEPTH']
    
    elif model_type in ['lstm', 'stacked_lstm', 'bilstm', 'gru']:
        params['NUM_LAYERS'] = trial.suggest_int('num_layers', 1, 3)
        params['DROPOUT'] = trial.suggest_float('dropout', 0.0, 0.4)
    
    elif model_type == 'cnn_lstm':
        params['CONV_CH'] = trial.suggest_categorical('conv_ch', [16, 32, 64])
        params['EXTRA_KWARGS']['conv_channels'] = params['CONV_CH']
        params['LR'] = trial.suggest_float('lr', 1e-5, 5e-4, log=True) # Max LR reset for CNN

    return params


def make_objective(model_type):
    """Factory function to create the Optuna objective function."""
    
    def objective(trial):
        # 1. Parameter Suggestion
        hparams = get_model_hyperparameters(trial, model_type)
        
        # Fixed parameters
        INITIAL_TRAIN_RATIO = 0.6
        VAL_RATIO = 0.1

        model = None
        X_scaler, Y_scaler = None, None
        
        try:
            # 2. Training (Walk-Forward)
            model, X_scaler, Y_scaler, train_losses, val_losses, split_idx = walk_forward_train_generic(
                df=df,
                feature_cols=FEATURE_COLS,
                model_type=model_type,
                LOOK_BACK=hparams['LOOK_BACK'],
                TRAIN_WINDOW=hparams['TRAIN_WINDOW'],
                INITIAL_TRAIN_RATIO=INITIAL_TRAIN_RATIO,
                RETRAIN_FREQUENCY=hparams['RETRAIN_FREQUENCY'],
                VAL_RATIO=VAL_RATIO,
                NUM_EPOCHS=hparams['NUM_EPOCHS'],
                HIDDEN_DIM=hparams['HIDDEN_DIM'],
                NUM_LAYERS=hparams['NUM_LAYERS'],
                DROPOUT=hparams['DROPOUT'],
                LR=hparams['LR'],
                WEIGHT_DECAY=hparams['WEIGHT_DECAY'],
                SEED=200 + trial.number,
                device=DEVICE,
                trial=trial,
                extra_kwargs=hparams['EXTRA_KWARGS']
            )

            # 3. Evaluation
            mse, _, _, _ = evaluate_model(
                model, df, X_scaler, Y_scaler, FEATURE_COLS, hparams['LOOK_BACK'], split_idx, device=DEVICE
            )
            mse_value = float(mse)

        except optuna.TrialPruned:
            # Propagate pruning signal
            raise
        except Exception as e:
            print(f'Trial {trial.number} failed for {model_type}: {e}')
            return float('inf')
        finally:
            # 4. Memory Cleanup (CRITICAL FOR EFFICIENCY)
            del X_scaler
            del Y_scaler
            
            # Ensure model and associated tensors are deleted
            if model is not None:
                del model
                
            # Clear unused memory from the PyTorch cache if using GPU
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        # 5. Result Logging (Minimal logging for non-best trials)
        trial_dir = os.path.join(RESULTS_DIR, model_type, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)
        summary = {'model': model_type, 'trial': trial.number, 'mse': mse_value, 'params': trial.params}
        with open(os.path.join(trial_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        return mse_value
    
    return objective


def copy_best_artifacts(study, model_type):
    """Finds the best trial and saves its full artifacts."""
    best_trial = study.best_trial
    best_params = get_model_hyperparameters(best_trial, model_type)
    
    print(f"Retraining and saving best model (Trial {best_trial.number})...")

    # Retrain the best model fully (no Optuna trial context needed)
    final_model, X_scaler, Y_scaler, _, _, _ = walk_forward_train_generic(
        df=df,
        feature_cols=FEATURE_COLS,
        model_type=model_type,
        LOOK_BACK=best_params['LOOK_BACK'],
        TRAIN_WINDOW=best_params['TRAIN_WINDOW'],
        INITIAL_TRAIN_RATIO=0.6,
        RETRAIN_FREQUENCY=best_params['RETRAIN_FREQUENCY'],
        VAL_RATIO=0.1,
        NUM_EPOCHS=best_params['NUM_EPOCHS'],
        HIDDEN_DIM=best_params['HIDDEN_DIM'],
        NUM_LAYERS=best_params['NUM_LAYERS'],
        DROPOUT=best_params['DROPOUT'],
        LR=best_params['LR'],
        WEIGHT_DECAY=best_params['WEIGHT_DECAY'],
        SEED=200 + best_trial.number,
        device=DEVICE,
        trial=None, # No trial object passed for final training
        extra_kwargs=best_params['EXTRA_KWARGS']
    )
    
    # Save the final artifacts to the main model directory
    save_trial_artifacts(os.path.join(RESULTS_DIR, model_type), 'best', final_model, X_scaler, Y_scaler, {'params': best_trial.params})
    
    # Cleanup after saving
    del final_model
    del X_scaler
    del Y_scaler
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    for m in models_to_tune:
        print('\n=== Tuning model:', m, '===')
        
        # Configure the Pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=50,
            interval_steps=20
        )
        
        study = optuna.create_study(
            direction='minimize', 
            sampler=optuna.samplers.TPESampler(seed=100), 
            pruner=pruner,
            study_name=f'crypto_forecasting_{m}',
            storage='sqlite:///optuna_studies.db',
            load_if_exists=True
        )
        
        study.optimize(make_objective(m), n_trials=N_TRIALS)

        best = study.best_trial
        model_dir = os.path.join(RESULTS_DIR, m)
        
        # Save best parameters
        with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
            json.dump({'value': best.value, 'params': best.params}, f, indent=2)
            
        print('Best for', m, '=> MSE:', best.value, best.params)
        
        # Only save the artifacts of the single best trial
        copy_best_artifacts(study, m)

    print('\nTuning complete. Results saved under', RESULTS_DIR)