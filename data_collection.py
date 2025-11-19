import ccxt
import pandas as pd
from datetime import datetime
import time
import numpy as np


exchange = ccxt.binance()


# symbols = [
#     'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
#     'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'LTC/USDT', 'AVAX/USDT',
#     'MATIC/USDT', 'LINK/USDT', 'BCH/USDT', 'TRX/USDT', 'UNI/USDT'
# ] 
symbols = [
    'BTC/USDT'
] 
timeframe = '1h' 
start_date = exchange.parse8601('2020-01-01T00:00:00Z')  
end_date = exchange.parse8601('2024-12-31T23:59:59Z')  
limit = 1000  

def true_range(df):
    h = df['high']
    l = df['low']
    c_prev = df['close'].shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
def calculate_rsi(series, period=14):
    """Calculate RSI with proper handling"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(df, period=14):
    """Calculate ADX for trend strength"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movement
    up = high - high.shift()
    down = low.shift() - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    
    return adx

def fetch_ohlcv(symbol, timeframe, start, end):
    all_ohlcv = []
    current_start = start
    while current_start < end:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_start, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_start = ohlcv[-1][0] + 24 * 60 * 60 * 1000 
            time.sleep(1)  
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(5)
    return all_ohlcv


for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    ohlcv_data = fetch_ohlcv(symbol, timeframe, start_date, end_date)
    
    
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    
    
    df['returns'] = df['close'].pct_change()
    
    # 1. PRICE-BASED FEATURES (Normalized)
    df['price_change'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['norm_close'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
    
    # 2. VOLATILITY FEATURES (Stationary)
    df['volatility_30'] = df['log_return'].rolling(30).std()
    df['volatility_7'] = df['log_return'].rolling(7).std()
    df['volatility_ratio'] = df['volatility_7'] / df['volatility_30']
    
    # Realized volatility (more predictive)
    df['realized_vol'] = df['log_return'].rolling(24).std() * np.sqrt(365)  # Daily annualized
    
    # 3. MOMENTUM OSCILLATORS (Normalized)
    # RSI with multiple timeframes
    for period in [6, 14, 24]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
    
    # Stochastic with normalization
    df['stoch_k_14'] = 100 * (
        (df['close'] - df['low'].rolling(14).min()) / 
        (df['high'].rolling(14).max() - df['low'].rolling(14).min())
    )
    df['stoch_d_14'] = df['stoch_k_14'].rolling(3).mean()
    
    # 4. TREND STRENGTH INDICATORS
    # ADX for trend strength
    df['adx_14'] = calculate_adx(df, 14)
    
    # EMA slope indicators
    for fast, slow in [(6, 21), (12, 26)]:
        df[f'ema_slope_ratio_{fast}_{slow}'] = (
            df['close'].ewm(span=fast).mean() / 
            df['close'].ewm(span=slow).mean() - 1
        )
    
    # 5. VOLUME-PRICE CONFIRMATION
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Volume-weighted features
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['price_vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    
    # OBV momentum
    df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
    df['obv_momentum'] = df['obv'].pct_change(5)
    
    # 6. MARKET REGIME FEATURES
    # Volatility regime
    vol_ma = df['realized_vol'].rolling(100).mean()
    df['high_vol_regime'] = (df['realized_vol'] > vol_ma * 1.2).astype(int)
    
    # Trend regime using multiple EMAs
    ema_9 = df['close'].ewm(span=9).mean()
    ema_21 = df['close'].ewm(span=21).mean() 
    ema_50 = df['close'].ewm(span=50).mean()
    
    df['trend_strength'] = (
        (ema_9 > ema_21).astype(int) + 
        (ema_21 > ema_50).astype(int) +
        (df['close'] > ema_50).astype(int)
    ) / 3.0
    
    # 7. MEAN REVERSION POTENTIAL
    # Z-score for mean reversion
    df['price_zscore_20'] = (
        (df['close'] - df['close'].rolling(20).mean()) / 
        df['close'].rolling(20).std()
    )
    
    # Bollinger Band position
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)
    
    # 8. CRYPTO-SPECIFIC FEATURES
    # Overnight/weekend effects (important for crypto 24/7 market)
    df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
    
    # Price range position
    df['range_position'] = (
        (df['close'] - df['low'].rolling(24).min()) / 
        (df['high'].rolling(24).max() - df['low'].rolling(24).min())
    )
    
    # 9. MACD FEATURES (Normalized)
    macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    macd_signal = macd_line.ewm(span=9).mean()
    df['macd_histogram'] = (macd_line - macd_signal) / df['close'] * 1000  # Normalized
    

    
    # Remove temporary columns and handle NaN values
    cols_to_keep = [col for col in df.columns if col not in ['volume_sma_20', 'vwap', 'obv']]
    df = df[cols_to_keep].copy()
    
    # Forward fill then backward fill to handle NaNs
    df = df.ffill().bfill()

    # Save to CSV
    df.to_csv(f'binance_{symbol.replace("/", "_")}_2020_2024.csv')
    print(f"Saved data for {symbol}")


