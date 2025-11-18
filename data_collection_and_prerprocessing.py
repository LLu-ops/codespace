import ccxt
import pandas as pd
from datetime import datetime
import time

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
    
    # Save to CSV
    df.to_csv(f'binance_{symbol.replace("/", "_")}_2020_2024.csv')
    print(f"Saved data for {symbol}")


