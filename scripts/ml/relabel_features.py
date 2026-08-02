import os
import json
import logging
import pandas as pd
import numpy as np
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

def relabel_symbol(symbol):
    logger.info(f"Relabeling {symbol}...")
    file_path = os.path.join(DATA_DIR, f"training_{symbol}.parquet")
    
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found.")
        return
        
    df = pd.read_parquet(file_path)
    
    N = algo_config.get("forecast_horizon_bars")
    tp_mult = algo_config.get("tp_vol_mult")
    sl_mult = algo_config.get("sl_vol_mult")
    
    closes = df["close"].values.astype(np.float64)
    vols = df["volatility"].values.astype(np.float64)
    n_rows = len(closes)
    
    out_long = np.full(n_rows, -1, dtype=np.int8)
    out_short = np.full(n_rows, -1, dtype=np.int8)
    
    logger.info(f"Applying Triple Barrier (span=100, tp_mult={tp_mult}, sl_mult={sl_mult}, N={N})...")
    
    if HAS_CUDA:
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
        _triple_barrier_cpu(closes, vols, float(tp_mult), float(sl_mult), int(N), out_long, out_short)
        
    df["label_long"] = out_long
    df["label_short"] = out_short
    
    # Drop rows that couldn't be labeled (the last N rows)
    df = df[(df["label_long"] != -1) & (df["label_short"] != -1)].copy()
    df["label_long"] = df["label_long"].astype(int)
    df["label_short"] = df["label_short"].astype(int)
    
    df.to_parquet(file_path)
    logger.info(f"Saved {len(df)} relabeled rows for {symbol} to {file_path}")

if __name__ == "__main__":
    for sym in SYMBOLS:
        relabel_symbol(sym)
