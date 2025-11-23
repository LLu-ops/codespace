import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load Data
# --------------------------
FILE_PATH = './cleaned/binance_BTC_USDT_2020_2024.csv'
OUT_DIR = './eda_plots'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

# --------------------------
# Feature Engineering (direct inline, no functions)
# --------------------------

# Log return

df.dropna(inplace=True)

print("Final dataset length:", len(df))

# --------------------------
# PLOTS
# --------------------------

# 1. Price
plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], linewidth=1)
plt.title("BTC Price (2020–2024)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/price.png", dpi=300)
plt.close()

# 2. Log returns
plt.figure(figsize=(14,6))
plt.plot(df.index, df['log_ret'], linewidth=0.8)
plt.title("Log Returns")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/log_returns.png", dpi=300)
plt.close()

# 3. Histogram of log returns
plt.figure(figsize=(8,5))
plt.hist(df['log_ret'], bins=120, alpha=0.7)
plt.title("Log Return Distribution")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/log_returns_hist.png", dpi=300)
plt.close()

# 4. Rolling volatility
plt.figure(figsize=(14,6))
plt.plot(df.index, df['log_ret'].rolling(48).std(), linewidth=1)
plt.title("Rolling Volatility (48h std)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/rolling_volatility.png", dpi=300)
plt.close()

# 5. Volume + Volume change
plt.figure(figsize=(14,6))
plt.plot(df.index, df['volume'], linewidth=1)
plt.title("Volume")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/volume.png", dpi=300)
plt.close()


# 6. Technical indicators
# plt.figure(figsize=(14,4))
# plt.plot(df.index, df['sma_dist'])
# plt.title("SMA Distance")
# plt.grid(alpha=0.3)
# plt.savefig(f"{OUT_DIR}/sma_dist.png", dpi=300)
# plt.close()

# plt.figure(figsize=(14,4))
# plt.plot(df.index, df['bb_width'])
# plt.title("Bollinger Band Width")
# plt.grid(alpha=0.3)
# plt.savefig(f"{OUT_DIR}/bb_width.png", dpi=300)
# plt.close()

# plt.figure(figsize=(14,4))
# plt.plot(df.index, df['rsi'])
# plt.title("RSI (0–1 normalized)")
# plt.grid(alpha=0.3)
# plt.savefig(f"{OUT_DIR}/rsi.png", dpi=300)
# plt.close()

# 7. Correlation
corr_cols = [
    'log_ret',
    'volatility_30',
    'volatility_7',
    'volatility_ratio',
    'realized_vol',
    'rsi_6',
    'rsi_14',
    'rsi_24',
    'stoch_k_14',
    'stoch_d_14',
    'ema_slope_ratio_6_21',
    'ema_slope_ratio_12_26',
    'volume_ratio',
    'price_vwap_deviation',
    'obv_momentum',
    'high_vol_regime',
    'trend_strength',
    'price_zscore_20',
    'bb_position',
    'intraday_volatility',
    'range_position',
    'macd_histogram',
    'obv_momentum'
]

plt.figure(figsize=(14, 12))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{OUT_DIR}/correlation.png", dpi=300)
plt.close()


print("EDA completed. Files saved to", OUT_DIR)
