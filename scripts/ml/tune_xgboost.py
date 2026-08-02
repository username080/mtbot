"""
Purpose: Uses Optuna to find the optimal XGBoost hyperparameters for the Primary models.
Uses Purged Walk-Forward Cross-Validation with aucpr (Precision-Recall AUC) to prevent 
autocorrelation leakage and optimize ranking power on imbalanced financial labels.

Key improvements over vanilla TimeSeriesSplit:
- Purged WFO: embargo window on BOTH sides of every fold boundary
- aucpr: more sensitive than roc-auc for imbalanced Triple Barrier labels
- Saves best_n_rounds from CV so train script doesn't need eval set
"""
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
EMBARGO = 1440  # bars to remove on both sides of each fold boundary (~1 trading day)

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

    # Strictly isolate Training Data (Pre-2024) to prevent leakage
    split_time = 1704067200000
    train_df = full_df[full_df["close_time"] < split_time].reset_index(drop=True)

    X = train_df[PRIMARY_FEATURES]
    y = train_df[target_label]
    return X, y


def purged_wfo_splits(n_samples, n_splits=5, embargo=EMBARGO):
    """
    Purged Walk-Forward Optimization splits.
    Each fold uses an expanding training window. An embargo gap is enforced on
    BOTH sides of the fold boundary to prevent autocorrelation leakage.

    Returns list of (train_indices, test_indices) tuples.
    """
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        test_start = fold_size * (i + 1)
        test_end = min(test_start + fold_size, n_samples)

        # Purge: remove embargo window before test set starts
        train_end = test_start - embargo
        if train_end < 100:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) > 1000 and len(test_idx) > 100:
            splits.append((train_idx, test_idx))

    return splits


def objective(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        # Tighter depth: shallower trees generalize better on financial data
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
        # Stronger regularization floor (was 0-5, now 1-10)
        'gamma': trial.suggest_float('gamma', 1.0, 10.0),
        # Much higher min_child_weight to prevent splitting on noise (was 10-50)
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 500),
        # L2 and L1 regularization
        'lambda': trial.suggest_float('lambda', 1.0, 5.0),
        'alpha': trial.suggest_float('alpha', 0.0, 2.0),
    }

    try:
        if xgb.core.build_info().get('USE_CUDA'):
            params['device'] = 'cuda'
        else:
            params['device'] = 'cpu'
    except Exception:
        params['device'] = 'cpu'

    splits = purged_wfo_splits(len(X))
    if not splits:
        raise ValueError("No valid splits generated. Dataset may be too small.")

    fold_scores = []
    fold_rounds = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Calculate Scale Pos Weight dynamically per fold
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        fold_params = params.copy()
        fold_params['scale_pos_weight'] = (
            neg_count / pos_count if (pos_count > 0 and pos_count < neg_count) else 1.0
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        fold_scores.append(model.best_score)
        fold_rounds.append(model.best_iteration + 1)

    # Save average optimal rounds across folds as a user attribute
    trial.set_user_attr('best_n_rounds', int(np.mean(fold_rounds)))
    return np.mean(fold_scores)


def tune(target_label):
    logger.info(f"Loading data for Optuna tuning ({target_label})...")
    X, y = load_data(target_label)

    logger.info("Starting Purged WFO Optuna study (Maximizing aucpr, 100 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, n_jobs=1)

    logger.info("Number of finished trials: %d", len(study.trials))
    best_trial = study.best_trial

    logger.info("Best trial:")
    logger.info("  Value (aucpr): %s", best_trial.value)
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info("    %s: %s", key, value)

    best_n_rounds = best_trial.user_attrs.get('best_n_rounds', 300)
    logger.info(f"  Best N Rounds (avg across CV folds): {best_n_rounds}")

    out_name = "best_xgb_params_long.json" if target_label == "label_long" else "best_xgb_params_short.json"
    best_params_path = os.path.join(DATA_DIR, out_name)

    save_data = best_trial.params.copy()
    # Prefix with _ so train script knows to pop it before passing to xgb.train
    save_data['_best_n_rounds'] = best_n_rounds
    with open(best_params_path, "w") as f:
        json.dump(save_data, f, indent=4)
    logger.info(f"Saved best params + n_rounds to {best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["long", "short"], required=True)
    args = parser.parse_args()
    label_col = "label_long" if args.target == "long" else "label_short"
    tune(label_col)
