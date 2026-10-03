"""
Purged K-Fold cross-validation (Lopez de Prado, Advances in Financial Machine
Learning, Ch. 7) and out-of-fold (OOF) prediction generation.

Different from scripts/ml/tune_xgboost.py's purged_wfo_splits: that one is
walk-forward (each fold's train window is always entirely before its test
window, so only one side needs an embargo). Here, a held-out fold can sit
anywhere in the timeline - both sides of its boundary need the embargo
purged, or a training row just outside the held-out block could still have a
triple-barrier label that overlaps into it (or vice versa).

Used by scripts/ml/build_meta_labels.py to generate long_prob/short_prob for
the meta-training set from models that never saw the row they're scoring -
the primary models' *production* artifacts (trained on 100% of the data) are
unaffected; this only changes how meta-training inputs are generated.
See documentation.md §9 for the in-sample-prediction bug this fixes.
"""
import numpy as np
import xgboost as xgb


def purged_kfold_splits(n_samples: int, n_folds: int = 5, embargo: int = 1440):
    """
    n_folds contiguous chronological blocks. For each fold, the held-out block
    is its own index range; the training set is every other row EXCEPT an
    embargo-sized buffer purged from both sides of the held-out block's
    boundaries. Returns a list of (train_idx, test_idx) covering the whole
    input - every row is held out by exactly one fold.
    """
    fold_bounds = np.linspace(0, n_samples, n_folds + 1, dtype=int)
    splits = []
    for i in range(n_folds):
        test_start, test_end = int(fold_bounds[i]), int(fold_bounds[i + 1])
        test_idx = np.arange(test_start, test_end)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_start:test_end] = False

        left_purge_start = max(0, test_start - embargo)
        train_mask[left_purge_start:test_start] = False

        right_purge_end = min(n_samples, test_end + embargo)
        train_mask[test_end:right_purge_end] = False

        train_idx = np.where(train_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def generate_oof_predictions(X, y, params: dict, n_folds: int = 5, embargo: int = 1440, num_boost_round: int = 300) -> np.ndarray:
    """
    Fits one XGBoost classifier per purged K-fold split and predicts only that
    fold's held-out rows, so every row's prediction comes from a model that
    never trained on it.

    X, y: pandas DataFrame / Series, same row order, RangeIndex (0..n-1) -
    caller must reset_index(drop=True) first.
    params: XGBoost params (objective/eval_metric/tree_method/device already
    set, e.g. loaded from best_xgb_params_{long,short}.json) - scale_pos_weight
    is recomputed per fold from that fold's own training labels.
    num_boost_round: typically the tuned _best_n_rounds from the same params file.
    """
    n_samples = len(X)
    oof_preds = np.full(n_samples, np.nan, dtype=np.float64)

    for train_idx, test_idx in purged_kfold_splits(n_samples, n_folds=n_folds, embargo=embargo):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        fold_params = dict(params)
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        fold_params["scale_pos_weight"] = (
            neg_count / pos_count if (pos_count > 0 and pos_count < neg_count) else 1.0
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(fold_params, dtrain, num_boost_round=num_boost_round)
        oof_preds[test_idx] = model.predict(dtest)

    return oof_preds
