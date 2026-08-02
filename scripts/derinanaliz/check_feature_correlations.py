"""
Purpose: Evaluates feature-to-label correlations and feature-to-feature collinearity across Primary (Long/Short) and Meta models.
Helper script for Phase 2 feature pruning and pipeline diagnostics.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

# Resolve path to import shared features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES, META_FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def print_section_header(title: str):
    logger.info("\n" + "=" * 70)
    logger.info(f" {title} ".center(70, "="))
    logger.info("=" * 70)

def analyze_feature_to_label_correlations(df: pd.DataFrame, features: list, targets: list, model_name: str):
    print_section_header(f"{model_name}: FEATURE-TO-LABEL CORRELATIONS")
    
    existing_feats = [f for f in features if f in df.columns]
    existing_targets = [t for t in targets if t in df.columns]
    
    if not existing_feats or not existing_targets:
        logger.warning(f"Missing features or targets for {model_name}.")
        return

    results = []
    for f in existing_feats:
        row = {"Feature": f}
        max_abs = 0.0
        for t in existing_targets:
            corr = df[f].corr(df[t])
            if np.isnan(corr):
                corr = 0.0
            row[t] = corr
            max_abs = max(max_abs, abs(corr))
        row["max_abs"] = max_abs
        results.append(row)
        
    # Sort by maximum absolute correlation across targets
    results.sort(key=lambda x: x["max_abs"], reverse=True)
    
    # Format and print table
    if len(existing_targets) == 2:
        logger.info(f"{'Feature':<25} | {'Long Target':>15} | {'Short Target':>15}")
        logger.info("-" * 61)
        for r in results:
            logger.info(f"{r['Feature']:<25} | {r[existing_targets[0]]:>+15.4f} | {r[existing_targets[1]]:>+15.4f}")
    else:
        target_name = existing_targets[0]
        logger.info(f"{'Feature':<25} | {target_name:>15}")
        logger.info("-" * 43)
        for r in results:
            logger.info(f"{r['Feature']:<25} | {r[target_name]:>+15.4f}")

def analyze_feature_collinearity(df: pd.DataFrame, features: list, model_name: str, threshold: float = 0.50):
    print_section_header(f"{model_name}: FEATURE-TO-FEATURE COLLINEARITY (|corr| >= {threshold})")
    
    existing_feats = [f for f in features if f in df.columns]
    if len(existing_feats) < 2:
        logger.warning(f"Not enough features to compute collinearity for {model_name}.")
        return

    corr_matrix = df[existing_feats].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(existing_feats)):
        for j in range(i + 1, len(existing_feats)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and val >= threshold:
                raw_val = df[existing_feats[i]].corr(df[existing_feats[j]])
                high_corr_pairs.append((existing_feats[i], existing_feats[j], raw_val, abs(raw_val)))
                
    high_corr_pairs.sort(key=lambda x: x[3], reverse=True)
    
    if not high_corr_pairs:
        logger.info(f"No feature pairs found with absolute correlation >= {threshold:.2f}!")
    else:
        logger.info(f"{'Feature A':<25} | {'Feature B':<25} | {'Correlation':>12}")
        logger.info("-" * 68)
        for feat_a, feat_b, raw_corr, _ in high_corr_pairs:
            logger.info(f"{feat_a:<25} | {feat_b:<25} | {raw_corr:>+12.4f}")

def main():
    # 1. Primary Model Data Analysis
    primary_dfs = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
        if os.path.exists(path):
            primary_dfs.append(pd.read_parquet(path))
            
    if primary_dfs:
        primary_df = pd.concat(primary_dfs, ignore_index=True)
        logger.info(f"Loaded {len(primary_df)} rows for Primary Model analysis.")
        analyze_feature_to_label_correlations(primary_df, PRIMARY_FEATURES, ["label_long", "label_short"], "PRIMARY MODEL")
        analyze_feature_collinearity(primary_df, PRIMARY_FEATURES, "PRIMARY MODEL", threshold=0.50)
    else:
        logger.warning("No primary training data found in backtest/data/. Build features first.")

    # 2. Meta Model Data Analysis
    meta_path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    if os.path.exists(meta_path):
        meta_df = pd.read_parquet(meta_path)
        logger.info(f"\nLoaded {len(meta_df)} rows for Meta Model analysis.")
        analyze_feature_to_label_correlations(meta_df, META_FEATURES, ["meta_label"], "META MODEL")
        analyze_feature_collinearity(meta_df, META_FEATURES, "META MODEL", threshold=0.50)
    else:
        logger.warning("No meta training data found in backtest/data/. Build meta labels first.")

    print_section_header("CORRELATION ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
