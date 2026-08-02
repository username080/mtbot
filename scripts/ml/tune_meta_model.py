"""
Purpose: Uses Optuna to find the optimal XGBoost hyperparameters for the Meta-Label model.
Uses Purged Walk-Forward Cross-Validation with aucpr. Saves best_n_rounds for the train script.

The Meta model dataset is much smaller (only signal bars) so we use n_splits=3 and a
smaller embargo (500 bars) to preserve enough folds.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import META_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
EMBARGO = 500  # smaller embargo — meta dataset is much smaller than primary


def purged_wfo_splits(n_samples, n_splits=3, embargo=EMBARGO):
    """
    Purged Walk-Forward Optimization splits with two-sided embargo.
    """
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        test_start = fold_size * (i + 1)
        test_end = min(test_start + fold_size, n_samples)
        train_end = test_start - embargo
        if train_end < 100:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 200 and len(test_idx) > 50:
            splits.append((train_idx, test_idx))
    return splits


def objective(trial):
    path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    df = pd.read_parquet(path)

    existing_feats = [f for f in META_FEATURES if f in df.columns]
    X = df[existing_feats]
    y = df["meta_label"]

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 20, 200),
        'gamma': trial.suggest_float('gamma', 1.0, 10.0),
        'lambda': trial.suggest_float('lambda', 1.0, 5.0),
        'alpha': trial.suggest_float('alpha', 0.0, 2.0),
        # Fixed, not Optuna-tuned: force the model to consider direction and
        # regime jointly rather than relying on it to discover the interaction
        # on its own (it wasn't - regime/trend_alignment ranked near the
        # bottom of feature importance before this).
        'interaction_constraints': [["is_long_signal", "regime", "regime_duration", "trend_alignment", "trend_strength"]],
    }

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

    splits = purged_wfo_splits(len(X))
    if not splits:
        raise ValueError("No valid splits generated. Meta dataset may be too small.")

    scores = []
    fold_rounds = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        scores.append(model.best_score)
        fold_rounds.append(model.best_iteration + 1)

    trial.set_user_attr('best_n_rounds', int(np.mean(fold_rounds)))
    return np.mean(scores)


def tune():
    logger.info("Starting Purged WFO Optuna study for Meta-Model (Maximizing aucpr, 100 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=1)

    logger.info("Meta-Model Tuning finished!")
    logger.info(f"Best aucpr: {study.best_value:.4f}")

    best_trial = study.best_trial
    logger.info("Best Parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    best_n_rounds = best_trial.user_attrs.get('best_n_rounds', 150)
    logger.info(f"Best N Rounds (avg across CV folds): {best_n_rounds}")

    out_path = os.path.join(DATA_DIR, "best_xgb_params_meta.json")
    save_data = best_trial.params.copy()
    save_data['_best_n_rounds'] = best_n_rounds
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=4)
    logger.info(f"Saved optimized parameters + n_rounds to {out_path}")


if __name__ == "__main__":
    tune()
