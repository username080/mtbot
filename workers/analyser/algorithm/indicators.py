import pandas as pd
import numpy as np
import sys

def compute_live_ta(bar_queue) -> dict:
    df = pd.DataFrame(list(bar_queue))
    if len(df) < 50:
        return {}
        
    try:
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['ema_cross'] = (df['ema_20'] - df['ema_50']) / df['close']
        df['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']
        df['momentum_20'] = df['close'].pct_change(20).fillna(0.0)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_percentile'] = df['atr_14'].rank(pct=True)
            
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        df['bollinger_width'] = (upper_band - lower_band) / rolling_mean
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # New Features
        ret_series = df['close'].pct_change()
        
        # Amihud Illiquidity
        dollar_vol = df['volume'] * df['close']
        df['amihud_illiquidity'] = ret_series.abs() / dollar_vol.replace(0, 1)
        
        # Volatility Ratio
        df['volatility'] = ret_series.ewm(span=100, adjust=False).std().fillna(0.005)
        vol_10 = ret_series.ewm(span=10, adjust=False).std().fillna(0.005)
        df['volatility_ratio'] = vol_10 / df['volatility'].replace(0, 0.005)
        
        last_row = df.iloc[-1].to_dict()
        return last_row
        
    except Exception as e:
        print(f"CRITICAL ERROR in indicators missing data: {e}")
        sys.exit(1)
