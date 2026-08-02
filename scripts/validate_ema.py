import os
import pandas as pd
import numpy as np
import sys
import json

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtest/data"))

def test():
    sym = "BTCUSDT"
    path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values("close_time").reset_index(drop=True)
    
    # Start after warm-up cutoff to have data
    warmup_cutoff = 1567958400000 + (200 * 4 * 3600 * 1000)
    df = df[df["close_time"] >= warmup_cutoff - (500 * 4 * 3600 * 1000)].reset_index(drop=True)
    df = df.head(100000) # test 100000 rows
    
    current_4h_bucket = 0
    ema_4h = 0.0
    regime = 0
    regime_duration = 0
    
    results = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["close"])
        close_time = int(row["close_time"])
        
        bucket_index = close_time // 14400000
        if current_4h_bucket == 0:
            current_4h_bucket = bucket_index
            ema_4h = close
        elif bucket_index > current_4h_bucket:
            # use previous close
            last_close = float(df.iloc[i-1]["close"]) if i > 0 else close
            alpha = 2.0 / 201.0
            ema_4h = (last_close * alpha) + (ema_4h * (1 - alpha))
            
            new_regime = 1 if last_close > ema_4h else -1
            if new_regime == regime:
                regime_duration += 1
            else:
                regime = new_regime
                regime_duration = 1
                
            current_4h_bucket = bucket_index
            
        results.append({
            "close_time": close_time,
            "live_ema": ema_4h,
            "live_regime": regime
        })
        
    res_df = pd.DataFrame(results)
    
    # Offline calculation
    sym_base = sym.replace("USDT", "")
    path_4h = os.path.join(DATA_DIR, sym_base, f"{sym}_4h_klines.parquet")
    df_4h = pd.read_parquet(path_4h)
    df_4h['open_time'] = df_4h['open_time'].astype(np.int64)
    df_4h['close'] = df_4h['close'].astype(float)
    df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
    df_4h['ema_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['regime'] = np.sign(df_4h['close'] - df_4h['ema_4h'])
    df_4h['join_time'] = df_4h['open_time'] + (4 * 3600 * 1000)
    
    # Merge
    merged = pd.merge_asof(
        res_df,
        df_4h[['join_time', 'ema_4h', 'regime']],
        left_on='close_time',
        right_on='join_time',
        direction='backward'
    )
    merged = merged.dropna(subset=['ema_4h', 'regime']).reset_index(drop=True)
    
    # Ensure warmup period passes
    merged = merged[merged['close_time'] >= warmup_cutoff]
    
    merged['diff'] = np.abs(merged['live_ema'] - merged['ema_4h']) / merged['ema_4h']
    print(merged[['close_time', 'live_ema', 'ema_4h', 'diff', 'live_regime', 'regime']].head(20))
    print("Max diff:", merged['diff'].max())
    print("Mean diff:", merged['diff'].mean())

if __name__ == "__main__":
    test()
