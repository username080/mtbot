"""
Purpose: The main data-prep script. Iterates through historical Dollar Bars, computes all ML features (Microstructure & TA), and applies the Triple-Barrier Method to create Long/Short training labels.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from collections import deque
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from workers.analyser.features.assembler import FeatureAssembler
from shared.features import PRIMARY_FEATURES, META_FEATURES
from tqdm import tqdm
from numba import njit, prange
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

with open(os.path.join(os.path.dirname(__file__), "../../config.json"), "r") as f:
    config = json.load(f)
algo_config = config.get("algorithm", {})

if HAS_CUDA:
    @cuda.jit
    def _triple_barrier_kernel(closes, vols, tp_mult, sl_mult, N, out_long, out_short):
        i = cuda.grid(1)
        if i >= len(closes) - N:
            return
            
        entry = closes[i]
        vol_i = vols[i]
        tp_i = max(0.0010, vol_i * tp_mult)
        sl_i = -max(0.0010, vol_i * sl_mult)
        
        label_l = 0
        label_s = 0
        
        for j in range(1, N + 1):
            ret = (closes[i + j] - entry) / entry
            
            # Long Logic
            if label_l == 0:
                if ret >= tp_i:
                    label_l = 1
                elif ret <= sl_i:
                    label_l = -1  # Dead
                    
            # Short Logic: mirrors Long's difficulty structure (favorable=down here).
            # Win requires the far tp_mult move (down); a near sl_mult move (up) kills it -
            # matches how risk_manager.py actually exits a short trade (sign-corrected ret,
            # same tp_mult/sl_mult distances for both directions).
            if label_s == 0:
                if ret <= -tp_i:
                    label_s = 1
                elif ret >= -sl_i:
                    label_s = -1  # Dead

            if label_l != 0 and label_s != 0:
                break

        out_long[i] = 1 if label_l == 1 else 0
        out_short[i] = 1 if label_s == 1 else 0

@njit(parallel=True)
def _triple_barrier_cpu(closes, vols, tp_mult, sl_mult, N, out_long, out_short):
    n = len(closes)
    for i in prange(n - N):
        entry = closes[i]
        vol_i = vols[i]
        tp_i = max(0.0010, vol_i * tp_mult)
        sl_i = -max(0.0010, vol_i * sl_mult)
        
        label_l = 0
        label_s = 0
        
        for j in range(1, N + 1):
            ret = (closes[i + j] - entry) / entry
            
            # Long Logic
            if label_l == 0:
                if ret >= tp_i:
                    label_l = 1
                elif ret <= sl_i:
                    label_l = -1  # Dead
                    
            # Short Logic: mirrors Long's difficulty structure (favorable=down here).
            # Win requires the far tp_mult move (down); a near sl_mult move (up) kills it -
            # matches how risk_manager.py actually exits a short trade (sign-corrected ret,
            # same tp_mult/sl_mult distances for both directions).
            if label_s == 0:
                if ret <= -tp_i:
                    label_s = 1
                elif ret >= -sl_i:
                    label_s = -1  # Dead

            if label_l != 0 and label_s != 0:
                break

        out_long[i] = 1 if label_l == 1 else 0
        out_short[i] = 1 if label_s == 1 else 0

def compute_ta_features(df):
    logger.info("Computing TA features vectorized...")
    try:
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_cross'] = (ema_20 - df['ema_50']) / df['close']
        
        # ATR 14
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_percentile'] = df['atr_14'].rolling(500).rank(pct=True)
        
        # Bollinger Width
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        df['bollinger_width'] = (upper_band - lower_band) / rolling_mean
        
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
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
        
        return df
    except Exception as e:
        logger.error(f"CRITICAL ERROR in indicators missing data: {e}")
        sys.exit(1)

def build_features(symbol):
    logger.info(f"Building training features for {symbol}")
    bars_path = os.path.join(DATA_DIR, f"dollar_bars_{symbol}_all.parquet")
    funding_path = os.path.join(DATA_DIR, f"funding_rate_{symbol}.parquet")
    
    if not os.path.exists(bars_path):
        logger.warning(f"No bars data found for {symbol}")
        return
        
    bars_df = pd.read_parquet(bars_path)
    
    if os.path.exists(funding_path):
        funding_df = pd.read_parquet(funding_path)
        # Guarantee matching int64 timestamp dtype (Unix milliseconds) for merge_asof
        if pd.api.types.is_datetime64_any_dtype(funding_df["fundingTime"]):
            funding_df["fundingTime"] = ((funding_df["fundingTime"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64")
        if pd.api.types.is_datetime64_any_dtype(bars_df["close_time"]):
            bars_df["close_time"] = ((bars_df["close_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64")
            
        funding_df = funding_df.sort_values("fundingTime")
        bars_df = bars_df.sort_values("close_time")
        
        bars_df = pd.merge_asof(
            bars_df, 
            funding_df[["fundingTime", "fundingRate"]], 
            left_on="close_time", 
            right_on="fundingTime", 
            direction="backward"
        )
        bars_df["fundingRate"] = bars_df["fundingRate"].fillna(0.0)
    else:
        bars_df["fundingRate"] = 0.0

    assembler = FeatureAssembler(algo_config)
    queue = deque(maxlen=algo_config.get("bar_queue_size", 200))
    
    from collections import defaultdict
    import gc
    cols_data = defaultdict(list)
    
    for i, row in enumerate(tqdm(bars_df.itertuples(), total=len(bars_df), desc=f"[{symbol}] Assembling Features", unit="bar")):
        bar = row._asdict()
        if "Index" in bar: del bar["Index"]
        
        queue.append(bar)
        funding = bar.get("fundingRate", 0.0)
        assembler.update(bar, queue, funding_rate=funding)
        
        feats = assembler.get_features()
        feats["symbol_id"] = 0 if symbol == "BTCUSDT" else 1
        
        feats["close_time"] = bar["close_time"]
        feats["close"] = bar["close"]
        feats["high"] = bar["high"]
        feats["low"] = bar["low"]
        feats["volume"] = bar["volume"]
        
        for k, v in feats.items():
            cols_data[k].append(v)
            
    # Free raw bars dataframe immediately to release ~1GB RAM before DataFrame creation
    del bars_df
    gc.collect()
    
    logger.info("Converting column dictionaries to DataFrame...")
    feat_df = pd.DataFrame(cols_data)
    del cols_data
    gc.collect()
    
    feat_df = compute_ta_features(feat_df)
    
    # Drop rows with NaN (from lags) before labeling
    feat_df = feat_df.dropna().reset_index(drop=True)
    
    # Downcast float64 to float32 to reduce RAM usage by 50% for Numba/XGBoost
    for col in feat_df.select_dtypes(include=['float64']).columns:
        if col != 'close_time':
            feat_df[col] = feat_df[col].astype('float32')
    gc.collect()
    
    N = algo_config.get("forecast_horizon_bars", 50)
    tp_mult = algo_config.get("tp_vol_mult", 2.0)
    sl_mult = algo_config.get("sl_vol_mult", 2.0)
    
    closes = feat_df["close"].values.astype(np.float64)
    vols = feat_df["volatility"].values.astype(np.float64)
    n_rows = len(closes)
    
    out_long = np.full(n_rows, -1, dtype=np.int8)
    out_short = np.full(n_rows, -1, dtype=np.int8)
    
    logger.info(f"Applying Lopez de Prado's Triple Barrier labeling (span=100, tp_mult={tp_mult}, sl_mult={sl_mult}, symmetric barriers, horizon={N} bars)...")
    
    if HAS_CUDA:
        logger.info("⚡ Using GPU Acceleration via Numba CUDA for Triple Barrier Labeling!")
        d_closes = cuda.to_device(closes)
        d_vols = cuda.to_device(vols)
        d_long = cuda.to_device(out_long)
        d_short = cuda.to_device(out_short)
        
        threadsperblock = 256
        blockspergrid = (n_rows + (threadsperblock - 1)) // threadsperblock
        _triple_barrier_kernel[blockspergrid, threadsperblock](
            d_closes, d_vols, float(tp_mult), float(sl_mult), int(N), d_long, d_short
        )
        out_long = d_long.copy_to_host()
        out_short = d_short.copy_to_host()
    else:
        logger.info("Using Multi-Core CPU Acceleration via Numba prange for Triple Barrier Labeling...")
        _triple_barrier_cpu(closes, vols, float(tp_mult), float(sl_mult), int(N), out_long, out_short)
        
    feat_df["label_long"] = out_long
    feat_df["label_short"] = out_short
    feat_df = feat_df[(feat_df["label_long"] != -1) & (feat_df["label_short"] != -1)].copy()
    feat_df["label_long"] = feat_df["label_long"].astype(int)
    feat_df["label_short"] = feat_df["label_short"].astype(int)
    
    # Filter columns to only keep active PRIMARY_FEATURES plus essential metadata and target labels
    keep_cols = list(set(PRIMARY_FEATURES + META_FEATURES + [
        "close_time", "close", "high", "low", "volume", 
        "volatility", "label_long", "label_short", "symbol_id"
    ]))
    existing_keep_cols = [c for c in feat_df.columns if c in keep_cols]
    feat_df = feat_df[existing_keep_cols]
    
    out_path = os.path.join(DATA_DIR, f"training_{symbol}.parquet")
    feat_df.to_parquet(out_path)
    logger.info(f"Saved {len(feat_df)} labeled feature rows to {out_path}")

if __name__ == "__main__":
    for sym in SYMBOLS:
        build_features(sym)
