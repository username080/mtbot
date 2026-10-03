import numpy as np
import pandas as pd

from shared.purged_kfold import purged_kfold_splits, generate_oof_predictions


def test_purged_kfold_splits_cover_all_rows_exactly_once():
    n_samples = 1000
    splits = purged_kfold_splits(n_samples, n_folds=5, embargo=20)

    assert len(splits) == 5

    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    # Every row is held out by exactly one fold - no gaps, no duplicates.
    assert sorted(all_test_idx.tolist()) == list(range(n_samples))


def test_purged_kfold_splits_train_test_disjoint():
    n_samples = 1000
    for train_idx, test_idx in purged_kfold_splits(n_samples, n_folds=5, embargo=20):
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0


def test_purged_kfold_splits_embargo_purges_both_sides():
    # A held-out block in the middle of the timeline needs embargo purged on
    # BOTH sides (unlike tune_xgboost.py's walk-forward splits, where train is
    # always entirely before test and only one side needs purging).
    n_samples = 1000
    embargo = 20
    splits = purged_kfold_splits(n_samples, n_folds=5, embargo=embargo)

    # The middle fold (index 2) has real training data on both sides of it.
    train_idx, test_idx = splits[2]
    test_start, test_end = test_idx.min(), test_idx.max() + 1

    train_set = set(train_idx.tolist())
    # No training row within `embargo` bars of either boundary.
    for offset in range(1, embargo + 1):
        assert (test_start - offset) not in train_set
        assert (test_end - 1 + offset) not in train_set
    # But rows further away than the embargo ARE eligible for training.
    assert (test_start - embargo - 5) in train_set
    assert (test_end - 1 + embargo + 5) in train_set


def test_generate_oof_predictions_covers_every_row_with_valid_probabilities():
    rng = np.random.default_rng(42)
    n_samples = 600
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    # Deliberately give x1 real signal so the fold models have something to learn.
    y = (x1 + 0.1 * x2 + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 3,
        "device": "cpu",
    }

    oof_preds = generate_oof_predictions(X, y, params=params, n_folds=5, embargo=10, num_boost_round=20)

    assert len(oof_preds) == n_samples
    assert not np.isnan(oof_preds).any()
    assert np.all((oof_preds >= 0.0) & (oof_preds <= 1.0))

    # Sanity: since x1 carries real signal, out-of-fold predictions should be
    # at least weakly informative (better than random ranking).
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, oof_preds)
    assert auc > 0.6
