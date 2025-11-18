import ccxt
import pandas as pd
from datetime import datetime
import time
import numpy as np
# Initialize Binance exchange
exchange = ccxt.binance()

# Define parameters
symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'LTC/USDT', 'AVAX/USDT',
    'MATIC/USDT', 'LINK/USDT', 'BCH/USDT', 'TRX/USDT', 'UNI/USDT'
] # Add more cryptos as needed
timeframe = '1h'  # Daily candles
start_date = exchange.parse8601('2020-01-01T00:00:00Z')  # Start: Jan 1, 2020
end_date = exchange.parse8601('2024-12-31T23:59:59Z')  # End: Dec 31, 2024
limit = 1000  # Max candles per API call

def true_range(df):
    h = df['high']
    l = df['low']
    c_prev = df['close'].shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
# Function to fetch OHLCV for a symbol
def fetch_ohlcv(symbol, timeframe, start, end):
    all_ohlcv = []
    current_start = start
    while current_start < end:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_start, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_start = ohlcv[-1][0] + 24 * 60 * 60 * 1000  # Next day
            time.sleep(1)  # Avoid rate limits
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(5)
    return all_ohlcv

# Fetch and save data for each symbol
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    ohlcv_data = fetch_ohlcv(symbol, timeframe, start_date, end_date)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    
    # Calculate micro indicators
    df['returns'] = df['close'].pct_change()
    
    df['volatility'] = df['returns'].rolling(window=30).std() * (252 ** 0.5)  # Annualized
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_25'] = df['close'].rolling(window=25).mean()
    # df['sma_99'] = df['close'].rolling(window=99).mean()
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_25'] = df['close'].ewm(span=25, adjust=False).mean()
    # df['ema_99'] = df['close'].ewm(span=99, adjust=False).mean()
    
    df['macd'] = df['ema_7'] - df['ema_25']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    df['rsi'] = 100 - (100 / (1 + df['returns'].rolling(window=14).mean() / df['returns'].rolling(window=14).std()))
    
    df['stochastic_k'] = (df['close'] - df['low']) / (df['high'] - df['low']) * 100
    df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

    df['tr'] = true_range(df)
    df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()

        # Bollinger width
    ma20 = df['close'].rolling(20, min_periods=1).mean()
    sd20 = df['close'].rolling(20, min_periods=1).std().fillna(0)
    df['bb_upper'] = ma20 + 2 * sd20
    df['bb_lower'] = ma20 - 2 * sd20
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # OBV
    df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()

        # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum().replace(0, np.nan)

        # VWMA
    df['vwma_20'] = (df['close'] * df['volume']).rolling(20, min_periods=1).sum() / df['volume'].rolling(20, min_periods=1).sum()

        # ADX-ish
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=1).sum() / df['atr_14'].replace(0, 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=1).sum() / df['atr_14'].replace(0, 1e-9)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    df['adx_14'] = dx.rolling(14, min_periods=1).mean()

        # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    tp_ma = tp.rolling(20, min_periods=1).mean()
    tp_md = tp.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci_20'] = (tp - tp_ma) / (0.015 * tp_md.replace(0, 1e-9))

        # Stochastic
    low_min = df['low'].rolling(14, min_periods=1).min()
    high_max = df['high'].rolling(14, min_periods=1).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()

        # ROC, MOM, WillR
    df['roc_12'] = 100 * (df['close'] - df['close'].shift(12)) / df['close'].shift(12).replace(0, np.nan)
    df['mom_10'] = df['close'] - df['close'].shift(10)
    df['willr_14'] = -100 * ((df['high'].rolling(14, min_periods=1).max() - df['close']) /
                                 (df['high'].rolling(14, min_periods=1).max() - df['low'].rolling(14, min_periods=1).min()).replace(0, np.nan))

    # Save to CSV
    df.to_csv(f'binance_{symbol.replace("/", "_")}_2020_2024.csv')
    print(f"Saved data for {symbol}")



# Combine all symbols into one DataFrame (optional)
combined_df = pd.concat(
    [pd.read_csv(f'binance_{s.replace("/", "_")}_2020_2024.csv', index_col='date', parse_dates=True)[['close', 'returns', 'volatility']]
     for s in symbols],
    axis=1, keys=symbols, names=['symbol', 'metric']
)
combined_df.to_csv('binance_combined_2020_2024.csv')