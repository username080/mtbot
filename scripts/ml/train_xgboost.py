"""
Purpose: Trains the final Primary XGBoost models (Long and Short) using the best 
hyperparameters AND best_n_rounds found by the tuning script.

Key improvement: trains on 100% of the training data for exactly best_n_rounds
(no eval set, no early stopping). This is correct because:
- Optimal n_rounds was already determined robustly by CV in tune_xgboost.py
- Using an eval set during final training wastes ~20% of your precious data
- Early stopping on the final run would overfit to the held-out eval noise
"""
import os
import json
import logging
import argparse
import pandas as pd
import xgboost as xgb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))


def load_data(target_label):
    dfs = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            dfs.append(df)

    if not dfs:
        raise ValueError("No training data found.")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values("close_time").reset_index(drop=True)

    # Strictly isolate Training Data (Pre-2024)
    split_time = 1704067200000
    train_df = full_df[full_df["close_time"] < split_time].reset_index(drop=True)

    X = train_df[PRIMARY_FEATURES]
    y = train_df[target_label]
    return X, y


def train(target_label):
    logger.info(f"Loading data for final training ({target_label})...")
    X, y = load_data(target_label)

    param_name = "best_xgb_params_long.json" if target_label == "label_long" else "best_xgb_params_short.json"
    best_params_path = os.path.join(DATA_DIR, param_name)

    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Missing {best_params_path}. Run tune_xgboost.py first.")

    with open(best_params_path, "r") as f:
        loaded = json.load(f)

    # Pop the internal key saved by the tuner — it's not an XGBoost param
    best_n_rounds = int(loaded.pop('_best_n_rounds', 300))

    params = loaded
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'aucpr'
    params['tree_method'] = 'hist'

    try:
        if xgb.core.build_info().get('USE_CUDA'):
            params['device'] = 'cuda'
        else:
            params['device'] = 'cpu'
    except Exception:
        params['device'] = 'cpu'

    # Scale pos weight on the FULL training set
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    params['scale_pos_weight'] = (
        neg_count / pos_count if (pos_count > 0 and pos_count < neg_count) else 1.0
    )

    logger.info(f"Loaded params: {params}")
    logger.info(
        f"Training on 100% of training data for exactly {best_n_rounds} rounds "
        f"(no eval set, no early stopping — n_rounds fixed by CV tuning)..."
    )

    # Train on ALL training data — no eval set, no early stopping
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_n_rounds,
        verbose_eval=50
    )

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../workers/analyser/models"))
    os.makedirs(models_dir, exist_ok=True)

    model_name = "xgb_long_v3.json" if target_label == "label_long" else "xgb_short_v3.json"
    out_path = os.path.join(models_dir, model_name)
    model.save_model(out_path)
    logger.info(f"Saved final model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["long", "short"], required=True)
    args = parser.parse_args()
    label_col = "label_long" if args.target == "long" else "label_short"
    train(label_col)
