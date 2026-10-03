"""
Purpose: Trains the final Meta-Label XGBoost model using best hyperparameters
and best_n_rounds from tune_meta_model.py.

Trains on 100% of the meta training data for exactly best_n_rounds rounds.
No eval set, no early stopping (n_rounds already optimized by CV tuning).
"""
import os
import json
import logging
import pandas as pd
import xgboost as xgb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import META_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../workers/analyser/models"))


def train_meta():
    path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    if not os.path.exists(path):
        logger.error("Meta-training data not found. Run build_meta_labels.py first.")
        return

    df = pd.read_parquet(path)

    # Only use features that actually exist in the dataset (safe guard)
    existing_feats = [f for f in META_FEATURES if f in df.columns]
    missing = set(META_FEATURES) - set(existing_feats)
    if missing:
        logger.warning(f"Missing META_FEATURES in dataset: {missing}")

    X = df[existing_feats]
    y = df["meta_label"]

    logger.info(f"Training Meta-Model on {len(X)} samples with {len(existing_feats)} features...")
    logger.info(f"Meta-label positive rate: {y.mean():.4f}")

    params_path = os.path.join(DATA_DIR, "best_xgb_params_meta.json")
    if not os.path.exists(params_path):
        logger.error(f"Missing {params_path}. Run tune_meta_model.py first.")
        return

    with open(params_path, "r") as f:
        loaded = json.load(f)

    # Pop the internal key saved by the tuner
    best_n_rounds = int(loaded.pop('_best_n_rounds', 150))
    params = loaded

    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'aucpr'
    params['tree_method'] = 'hist'
    # Fixed constant, not persisted in best_xgb_params_meta.json (it isn't an
    # Optuna trial.suggest_* param) - must be set independently here, matching
    # tune_meta_model.py's objective() exactly.
    params['interaction_constraints'] = [["is_long_signal", "regime", "regime_duration", "trend_alignment", "trend_strength"]]

    try:
        if xgb.core.build_info().get('USE_CUDA'):
            params['device'] = 'cuda'
        else:
            params['device'] = 'cpu'
    except Exception:
        params['device'] = 'cpu'

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1.0

    logger.info(f"Loaded params: {params}")
    logger.info(
        f"Training on 100% of meta training data for exactly {best_n_rounds} rounds "
        f"(no eval set, no early stopping — n_rounds fixed by CV tuning)..."
    )

    # Train on ALL meta data — no eval set, no early stopping
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_n_rounds,
        verbose_eval=50
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "xgb_meta_v1.json")
    model.save_model(out_path)
    logger.info(f"Saved meta-model to {out_path}")


if __name__ == "__main__":
    train_meta()
