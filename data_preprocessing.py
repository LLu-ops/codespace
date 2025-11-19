import os
import glob
import pandas as pd
import numpy as np

OUT_DIR = "./cleaned"
FREQ = "h"  
os.makedirs(OUT_DIR, exist_ok=True)

for fp in sorted(glob.glob("./binance_*.csv")):
    try:
        df = pd.read_csv(fp, index_col=0, parse_dates=True, dtype=float, low_memory=False)
    except Exception as e:
        print(f"Skipping {fp}: {e}")
        continue

    
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]

    # compute log returns
    if 'Close' in df.columns:
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # fill missing values
    df = df.asfreq(FREQ).fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(axis=1, how='all')

    out_path = os.path.join(OUT_DIR, os.path.basename(fp))
    df.to_csv(out_path, index=True)
    print(f"Saved cleaned -> {out_path} (rows={len(df)}, cols={len(df.columns)})")