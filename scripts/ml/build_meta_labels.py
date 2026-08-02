import os
import gc
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.meta_signal_builder import build_sequential_meta_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../workers/analyser/models"))

with open(os.path.join(os.path.dirname(__file__), "../../config.json"), "r") as f:
    _config = json.load(f)
_algo_config = _config.get("algorithm", {})
ENTRY_THRESHOLD = _algo_config.get("entry_prob_threshold", 0.65)
FORECAST_HORIZON_BARS = _algo_config.get("forecast_horizon_bars", 50)

def build_meta_dataset():
    path_long = os.path.join(MODEL_DIR, "xgb_long_v3.json")
    path_short = os.path.join(MODEL_DIR, "xgb_short_v3.json")
    
    if not os.path.exists(path_long) or not os.path.exists(path_short):
        logger.warning("Primary v3 models not found. Skipping meta-label build for now.")
        return
        
    model_long = xgb.Booster()
    model_long.load_model(path_long)
    model_short = xgb.Booster()
    model_short.load_model(path_short)
    
    signal_dfs = []
    
    split_time = 1704067200000
    
    for sym in tqdm(SYMBOLS, desc="Building Meta-Labels"):
        path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        logger.info(f"Processing {sym} for Meta-Labels...")
        df = pd.read_parquet(path)
        df = df.sort_values("close_time").reset_index(drop=True)
        
        # Load 4H klines for regime and trend features
        sym_base = sym.replace("USDT", "")
        path_4h = os.path.join(DATA_DIR, sym_base, f"{sym}_4h_klines.parquet")
        if not os.path.exists(path_4h):
            logger.error(f"4H data not found: {path_4h}. Skipping {sym}.")
            continue
            
        df_4h = pd.read_parquet(path_4h)
        df_4h['open_time'] = df_4h['open_time'].astype(np.int64)
        df_4h['close'] = df_4h['close'].astype(float)
        df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
        
        # Calculate 4H EMA and Regime
        df_4h['ema_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
        df_4h['regime'] = np.sign(df_4h['close'] - df_4h['ema_4h'])
        # Count consecutive bars in the same regime
        df_4h['regime_duration'] = df_4h.groupby((df_4h['regime'] != df_4h['regime'].shift()).cumsum()).cumcount() + 1
        
        # Shift join timestamp to the CLOSE of the 4H candle to prevent lookahead leakage
        df_4h['join_time'] = df_4h['open_time'] + (4 * 3600 * 1000)
        
        # Warmup cutoff: we need 200 bars of 4H to warm up EMA
        warmup_cutoff = df_4h['open_time'].min() + (200 * 4 * 3600 * 1000)
        
        train_df = df[df["close_time"] < split_time].reset_index(drop=True)
        train_df = train_df[train_df['close_time'] >= warmup_cutoff].reset_index(drop=True)
        
        del df
        gc.collect()
        
        # Merge 4H features onto the dollar bars without lookahead
        # direction='backward' means the dollar bar matches with the latest 4H candle that was finalized <= the dollar bar's close_time
        train_df = pd.merge_asof(
            train_df, 
            df_4h[['join_time', 'ema_4h', 'regime', 'regime_duration']], 
            left_on='close_time', 
            right_on='join_time', 
            direction='backward'
        )
        
        # If any rows failed to match, drop them
        train_df = train_df.dropna(subset=['ema_4h', 'regime']).reset_index(drop=True)
        
        logger.info(f"  Generating Long predictions for {sym}...")
        dmatrix_long = xgb.DMatrix(train_df[model_long.feature_names])
        train_df['long_prob'] = model_long.predict(dmatrix_long)
        
        del dmatrix_long
        gc.collect()
        
        logger.info(f"  Generating Short predictions for {sym}...")
        dmatrix_short = xgb.DMatrix(train_df[model_short.feature_names])
        train_df['short_prob'] = model_short.predict(dmatrix_short)
        
        del dmatrix_short
        gc.collect()
        
        # Sequential, single-position signal selection: a bar only counts as
        # a signal if it isn't within FORECAST_HORIZON_BARS of a prior
        # selected signal - matching the live bot, which never evaluates new
        # entries while a position is open. This also fixes a leakage bug:
        # letting every qualifying bar count separately made recent_signal_pnl_5
        # roll over heavily overlapping triple-barrier outcome windows (near
        # copies of the current row's own future), instead of genuine
        # distinct past trades.
        signals_df = build_sequential_meta_signals(
            train_df, entry_threshold=ENTRY_THRESHOLD, horizon=FORECAST_HORIZON_BARS
        )

        del train_df
        gc.collect()

        # Drop temporary cols
        signals_df.drop(columns=['join_time', 'is_win'], inplace=True, errors='ignore')

        signal_dfs.append(signals_df)
        logger.info(f"  Found {len(signals_df)} signals for {sym}.")

    if not signal_dfs:
        logger.error("No signals generated.")
        return
        
    combined_signals = pd.concat(signal_dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    combined_signals.to_parquet(out_path)
    logger.info(f"Generated {len(combined_signals)} total meta-training examples.")
    logger.info(f"Meta-label positive class rate: {combined_signals['meta_label'].mean():.4f}")

if __name__ == "__main__":
    build_meta_dataset()
