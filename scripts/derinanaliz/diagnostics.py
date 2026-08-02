"""
Purpose: Evaluates the precision, recall, and lift of the trained models on unseen out-of-sample data. It generates the binned probability table to see how model confidence scales with accuracy.
"""
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES, META_FEATURES
from shared.meta_signal_builder import build_sequential_meta_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

with open(os.path.join(os.path.dirname(__file__), "../../config.json"), "r") as f:
    _config = json.load(f)
_algo_config = _config.get("algorithm", {})
ENTRY_THRESHOLD = _algo_config.get("entry_prob_threshold", 0.65)
FORECAST_HORIZON_BARS = _algo_config.get("forecast_horizon_bars", 50)

def run_diagnostics():
    logger.info("Loading dataset for diagnostics...")
    dfs = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['symbol'] = sym  # Inject symbol for later grouping
            
            # Load 4H klines for regime and trend features
            sym_base = sym.replace("USDT", "")
            path_4h = os.path.join(DATA_DIR, sym_base, f"{sym}_4h_klines.parquet")
            if os.path.exists(path_4h):
                df_4h = pd.read_parquet(path_4h)
                df_4h['open_time'] = df_4h['open_time'].astype(np.int64)
                df_4h['close'] = df_4h['close'].astype(float)
                df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
                
                df_4h['ema_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
                df_4h['regime'] = np.sign(df_4h['close'] - df_4h['ema_4h'])
                df_4h['regime_duration'] = df_4h.groupby((df_4h['regime'] != df_4h['regime'].shift()).cumsum()).cumcount() + 1
                
                df_4h['join_time'] = df_4h['open_time'] + (4 * 3600 * 1000)
                
                df = df.sort_values("close_time").reset_index(drop=True)
                df = pd.merge_asof(
                    df, 
                    df_4h[['join_time', 'ema_4h', 'regime', 'regime_duration']], 
                    left_on='close_time', 
                    right_on='join_time', 
                    direction='backward'
                )
                df = df.dropna(subset=['ema_4h', 'regime']).reset_index(drop=True)
                dfs.append(df)
            else:
                logger.warning(f"4H data not found for {sym}")
                
    if not dfs:
        logger.error("No training data found.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True).sort_values("close_time").reset_index(drop=True)
            
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../workers/analyser/models"))
    path_long = os.path.join(base_dir, "xgb_long_v3.json")
    path_short = os.path.join(base_dir, "xgb_short_v3.json")
    
    split_time = 1704067200000
    test_df_raw = full_df[full_df["close_time"] >= split_time]
    
    # Enforce 1440 bar Purge/Embargo Gap between Train end (Dec 31 2023) and Test start
    test_df = test_df_raw.iloc[1440:].reset_index(drop=True)
    
    # Probability bins — defined here so both Long, Short and Meta sections can use it
    bins = [(0.0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
            (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),
            (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.00)]
    
    if len(test_df) == 0:
        logger.warning("No test data found (all data before 2024).")
        return
        
    # Evaluate Long Model
    if os.path.exists(path_long):
        logger.info(f"\n--- Loading LONG model from {path_long} ---")
        model_long = xgb.Booster()
        model_long.load_model(path_long)
        
        dtest = xgb.DMatrix(test_df[model_long.feature_names])
        probs = model_long.predict(dtest)
        
        test_pos_rate = test_df['label_long'].mean()
        logger.info(f"Test set positive base rate (LONG): {test_pos_rate:.4f} ({test_pos_rate*100:.1f}%)")
        logger.info(f"{'Threshold':>10} {'Signals':>10} {'Precision':>10} {'Recall':>10} {'Lift':>15}")
        for t in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            preds = (probs > t).astype(int)
            tp = ((preds == 1) & (test_df['label_long'].values == 1)).sum()
            fp = ((preds == 1) & (test_df['label_long'].values == 0)).sum()
            fn = ((preds == 0) & (test_df['label_long'].values == 1)).sum()
            total = preds.sum()
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            lift = prec / test_pos_rate if test_pos_rate > 0 else 0
            logger.info(f"{t:>10.2f} {total:>10} {prec:>10.4f} {rec:>10.4f} {lift:>15.2f}x")
            
        logger.info(f"\n{'Probability Bin':>20} {'Signals':>10} {'Precision':>10} {'Lift':>10}")
        for b_low, b_high in bins:
            mask = (probs >= b_low) & (probs < b_high)
            total = mask.sum()
            tp = (mask & (test_df['label_long'].values == 1)).sum()
            prec = tp / total if total > 0 else 0
            lift = prec / test_pos_rate if test_pos_rate > 0 else 0
            logger.info(f"{b_low:.2f} - {b_high:.2f} {total:>10} {prec:>10.4f} {lift:>9.2f}x")
            
    # Evaluate Short Model
    if os.path.exists(path_short):
        logger.info(f"\n--- Loading SHORT model from {path_short} ---")
        model_short = xgb.Booster()
        model_short.load_model(path_short)
        
        dtest = xgb.DMatrix(test_df[model_short.feature_names])
        probs = model_short.predict(dtest)
        
        test_pos_rate = test_df['label_short'].mean()
        logger.info(f"Test set positive base rate (SHORT): {test_pos_rate:.4f} ({test_pos_rate*100:.1f}%)")
        logger.info(f"{'Threshold':>10} {'Signals':>10} {'Precision':>10} {'Recall':>10} {'Lift':>15}")
        for t in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            preds = (probs > t).astype(int)
            tp = ((preds == 1) & (test_df['label_short'].values == 1)).sum()
            fp = ((preds == 1) & (test_df['label_short'].values == 0)).sum()
            fn = ((preds == 0) & (test_df['label_short'].values == 1)).sum()
            total = preds.sum()
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            lift = prec / test_pos_rate if test_pos_rate > 0 else 0
            logger.info(f"{t:>10.2f} {total:>10} {prec:>10.4f} {rec:>10.4f} {lift:>15.2f}x")
            
        logger.info(f"\n{'Probability Bin':>20} {'Signals':>10} {'Precision':>10} {'Lift':>10}")
        for b_low, b_high in bins:
            mask = (probs >= b_low) & (probs < b_high)
            total = mask.sum()
            tp = (mask & (test_df['label_short'].values == 1)).sum()
            prec = tp / total if total > 0 else 0
            lift = prec / test_pos_rate if test_pos_rate > 0 else 0
            logger.info(f"{b_low:.2f} - {b_high:.2f} {total:>10} {prec:>10.4f} {lift:>9.2f}x")
            
    # Evaluate Meta Model
    path_meta = os.path.join(base_dir, "xgb_meta_v1.json")
    if os.path.exists(path_meta) and os.path.exists(path_long) and os.path.exists(path_short):
        logger.info(f"\n--- Loading META model from {path_meta} ---")
        model_meta = xgb.Booster()
        model_meta.load_model(path_meta)
        
        # Get primary probabilities on test set
        dtest_l = xgb.DMatrix(test_df[model_long.feature_names])
        test_df['long_prob'] = model_long.predict(dtest_l)
        
        dtest_s = xgb.DMatrix(test_df[model_short.feature_names])
        test_df['short_prob'] = model_short.predict(dtest_s)
        
        # Sequential, single-position signal selection - same logic used to
        # build the meta training set (shared.meta_signal_builder), so the
        # offline evaluation population matches what the live/backtest engine
        # can actually act on (one trade at a time, locked out for
        # FORECAST_HORIZON_BARS after entry), instead of counting every bar
        # that clears the threshold as its own independent "signal".
        per_symbol_sigs = []
        for sym in SYMBOLS:
            sym_df = test_df[test_df['symbol'] == sym].sort_values('close_time').reset_index(drop=True)
            if len(sym_df) == 0:
                continue
            sym_sig_df = build_sequential_meta_signals(
                sym_df, entry_threshold=ENTRY_THRESHOLD, horizon=FORECAST_HORIZON_BARS
            )
            if len(sym_sig_df) > 0:
                per_symbol_sigs.append(sym_sig_df)

        sig_df = pd.concat(per_symbol_sigs, ignore_index=True) if per_symbol_sigs else pd.DataFrame()
        if len(sig_df) > 0:
            existing_meta_feats = [f for f in META_FEATURES if f in sig_df.columns]
            missing_meta = set(META_FEATURES) - set(existing_meta_feats)
            if missing_meta:
                logger.warning(f"META_FEATURES missing from test data: {missing_meta}. Diagnostics may be partial.")
            
            dmeta = xgb.DMatrix(sig_df[existing_meta_feats])
            sig_df['meta_prob'] = model_meta.predict(dmeta)
            
            raw_win_rate = sig_df['is_win'].mean()
            logger.info(f"Raw Primary Signals (sequential, single-position, >{ENTRY_THRESHOLD}): {len(sig_df)} | Raw Precision: {raw_win_rate*100:.2f}%")
            
            logger.info(f"\n{'Meta Threshold':>15} {'Surviving Signals':>20} {'Filtered Precision':>20} {'Lift vs Raw':>15}")
            for mt in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                filtered = sig_df[sig_df['meta_prob'] >= mt]
                f_total = len(filtered)
                f_prec = filtered['is_win'].mean() if f_total > 0 else 0
                lift = f_prec / raw_win_rate if raw_win_rate > 0 else 0
                logger.info(f"{mt:>15.2f} {f_total:>20} {f_prec*100:>19.2f}% {lift:>14.2f}x")
                
            logger.info(f"\n{'Meta Probability Bin':>22} {'Signals':>10} {'Filtered Precision':>20} {'Lift vs Raw':>15}")
            for b_low, b_high in bins:
                mask = (sig_df['meta_prob'] >= b_low) & (sig_df['meta_prob'] < b_high)
                total = mask.sum()
                tp = (mask & (sig_df['is_win'] == 1)).sum()
                f_prec = tp / total if total > 0 else 0
                lift = f_prec / raw_win_rate if raw_win_rate > 0 else 0
                logger.info(f"{b_low:.2f} - {b_high:.2f} {total:>10} {f_prec*100:>19.2f}% {lift:>14.2f}x")
        else:
            logger.info("No raw primary signals found in test set to evaluate Meta Model.")

if __name__ == "__main__":
    run_diagnostics()
