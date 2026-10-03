"""
Purpose: Single Feature Importance (SFI) analysis - Lopez de Prado's Advances in
Financial Machine Learning, Chapter 8. Complements feature_importance.py (embedded
MDI-style gain importance from the full multivariate model) and
check_feature_correlations.py (in-sample correlation) with the one method neither
of them does: scoring every candidate feature completely on its own, out-of-sample,
via purged walk-forward CV - exactly the idea written down in cokonemli.txt.

For every candidate feature and every target, reports BOTH:
  - "sfi_auc": a minimal single-feature XGBoost classifier (trained using only that
    one feature, nothing else) scored out-of-sample on purged/embargoed folds.
  - "raw_auc" / "spearman_corr": no model at all - the feature's own raw values used
    directly as a ranking score against the label.
AUC (0.5 = random), not accuracy, since these are imbalanced triple-barrier/meta
labels and accuracy would just reward predicting the majority class.

The full ranked table is the point - every candidate feature is reported, not just
the top N. --top-n / --threshold only add a "selected" marker on top of that table.

Candidate features = today's production PRIMARY_FEATURES / META_FEATURES, plus (for
the two primary targets only - see note below) a set of "Tier A" technical/
statistical features computed purely from columns already present in the training
data (close/high/low/volume/volatility) - no raw-data merge needed.

Also reports feature-to-feature collinearity and hierarchical clustering on the same
candidate set, so two features that both score well but are really the same signal
(e.g. two VPIN-derivative features) are visible next to their scores instead of
silently occupying two "top 10" slots - the standard fix for SFI's blind spot to
feature substitution effects (AFML Ch. 8's clustered-importance idea).

NOTE: Tier A features are only added for the label_long/label_short targets, which
run on the dense per-bar training_{symbol}.parquet files. training_meta_combined.parquet
holds one row per *selected trade signal* (sparse, sequentially filtered - see
shared/meta_signal_builder.py), not one row per bar, so rolling-window TA indicators
computed directly on it would span irregular, non-contiguous time gaps between
signals rather than a true bar-by-bar window. Recomputing Tier A on the dense per-bar
series and merge_asof-ing it back onto the sparse meta signals would require a
per-symbol split that training_meta_combined.parquet doesn't retain (no "symbol"
column), so it's left out of scope here rather than done incorrectly.
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES, META_FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Pre-2024 train-only cutoff, matches tune_xgboost.py / train_xgboost.py / diagnostics.py -
# this analysis must never peek at the OOS test window the real models are evaluated on.
SPLIT_TIME = 1704067200000

DEFAULT_TARGETS = ["label_long", "label_short", "meta_label"]
PRIMARY_TARGET_NAMES = ("label_long", "label_short")
META_TARGET_NAME = "meta_label"

TIER_A_FEATURES = [
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "williams_r",
    "adx_14",
    "cci_20",
    "roll_spread",
    "return_skew_50", "return_kurt_50",
    "return_sign_entropy_50",
]


# ---------------------------------------------------------------------------
# Tier A feature engineering - computed only from close/high/low/volume columns
# already present in the training parquet, no raw-data merge needed.
# ---------------------------------------------------------------------------

def add_tier_a_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # MACD (12/26/9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic Oscillator (14/3)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    rng_14 = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - low_14) / rng_14
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R (14)
    df["williams_r"] = -100 * (high_14 - close) / rng_14

    # ADX (14) - simplified rolling-mean smoothing rather than Wilder's recursive
    # smoothing; good enough for a candidate-screening pass, not meant to replace
    # a production-grade ADX.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift()
    true_range = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.rolling(14).mean()

    # CCI (20)
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    # Roll (1984) implied effective spread: 2*sqrt(-cov(dP_t, dP_t-1)) when the serial
    # covariance of price changes is negative (bid-ask bounce), else no measurable
    # spread under this estimator.
    dprice = close.diff()
    roll_cov = dprice.rolling(20).cov(dprice.shift(1))
    df["roll_spread"] = np.where(roll_cov < 0, 2 * np.sqrt(-roll_cov), 0.0)

    # Rolling return skewness / kurtosis (window=50)
    ret = close.pct_change()
    df["return_skew_50"] = ret.rolling(50).skew()
    df["return_kurt_50"] = ret.rolling(50).kurt()

    # Shannon entropy of the rolling up/down return-sign sequence (window=50).
    # Only the up/down proportions are used (flat bars excluded from both), so this
    # is a simplified two-state signed-entropy proxy, not entropy over a full
    # distribution that sums to 1 - low entropy means returns have been
    # one-directional over the window, high entropy means roughly 50/50 chop.
    sign = np.sign(ret).fillna(0.0)

    def _sign_entropy(x):
        p_up = np.mean(x > 0)
        p_down = np.mean(x < 0)
        ent = 0.0
        for p in (p_up, p_down):
            if p > 0:
                ent -= p * np.log2(p)
        return ent

    df["return_sign_entropy_50"] = sign.rolling(50).apply(_sign_entropy, raw=True)

    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_primary_dataset() -> pd.DataFrame:
    dfs = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"training_{sym}.parquet")
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
        else:
            logger.warning(f"Missing primary training data: {path}")
    if not dfs:
        raise FileNotFoundError("No primary training data found in backtest/data/.")

    df = pd.concat(dfs, ignore_index=True).sort_values("close_time").reset_index(drop=True)
    df = df[df["close_time"] < SPLIT_TIME].reset_index(drop=True)
    df = add_tier_a_features(df)
    return df


def load_meta_dataset() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing meta training data: {path}")
    return pd.read_parquet(path).sort_values("close_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Purged walk-forward CV - same design as scripts/ml/tune_xgboost.py's
# purged_wfo_splits, duplicated locally since scripts/ml has no package structure
# to import from (the rest of the derinanaliz/ scripts follow the same precedent,
# e.g. diagnostics.py reimplements its own purge gap rather than cross-importing).
# ---------------------------------------------------------------------------

def purged_wfo_splits(n_samples: int, n_splits: int = 5, embargo: int = 1440):
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

        if len(train_idx) > 1000 and len(test_idx) > 100:
            splits.append((train_idx, test_idx))

    return splits


def _subsample(idx: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


# ---------------------------------------------------------------------------
# Per-feature, per-fold scoring
# ---------------------------------------------------------------------------

def score_feature(df: pd.DataFrame, feature: str, target: str, splits, max_rows_per_fold: int, skip_model_fit: bool):
    values = df[feature].to_numpy(dtype=np.float64)
    labels = df[target].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)

    fold_results = []
    for train_idx, test_idx in splits:
        train_idx = _subsample(train_idx[finite_mask[train_idx]], max_rows_per_fold)
        test_idx = _subsample(test_idx[finite_mask[test_idx]], max_rows_per_fold)

        y_train = labels[train_idx]
        y_test = labels[test_idx]
        x_train = values[train_idx]
        x_test = values[test_idx]

        fold_entry = {"n_train": int(len(train_idx)), "n_test": int(len(test_idx))}

        # No-model stats: the feature's own raw values used directly as a score.
        if len(np.unique(y_test)) > 1 and len(x_test) > 1:
            try:
                raw_auc = float(roc_auc_score(y_test, x_test))
                # Normalize direction: a feature that's a strong *inverse* predictor
                # (raw_auc near 0) is just as useful a signal as one near 1 - only a
                # raw_auc near 0.5 means no standalone signal either way.
                fold_entry["raw_auc"] = max(raw_auc, 1.0 - raw_auc)
            except ValueError:
                fold_entry["raw_auc"] = None
            corr, _ = spearmanr(x_test, y_test)
            fold_entry["spearman_corr"] = float(corr) if np.isfinite(corr) else None
        else:
            fold_entry["raw_auc"] = None
            fold_entry["spearman_corr"] = None

        # Model-based SFI: a classifier trained using ONLY this one feature, nothing
        # else - the literal AFML Single Feature Importance definition. Its predicted
        # probabilities already account for direction, so unlike raw_auc above this
        # needs no normalization.
        if not skip_model_fit and len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            model = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x_train.reshape(-1, 1), y_train)
            preds = model.predict_proba(x_test.reshape(-1, 1))[:, 1]
            try:
                fold_entry["sfi_auc"] = float(roc_auc_score(y_test, preds))
            except ValueError:
                fold_entry["sfi_auc"] = None
        else:
            fold_entry["sfi_auc"] = None

        fold_results.append(fold_entry)

    return fold_results


def _agg_stat(fold_results, key: str) -> dict:
    vals = [f[key] for f in fold_results if f.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "worst": None, "n_folds": 0}
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "worst": float(arr.min()),
        "n_folds": int(len(arr)),
    }


def is_selected(agg_sfi: dict, agg_raw: dict, threshold: float, fold_results) -> bool:
    """
    A feature clears --threshold only if its mean AUC (model-based if available,
    else raw) beats the threshold AND a majority of its individual folds do too -
    guards against a feature that just got lucky on one slice of a low-signal,
    autocorrelated financial series (the multiple-comparisons trap raised earlier).
    """
    primary_key = "sfi_auc" if agg_sfi["mean"] is not None else "raw_auc"
    agg = agg_sfi if primary_key == "sfi_auc" else agg_raw
    if agg["mean"] is None or agg["mean"] < threshold:
        return False

    fold_vals = [f[primary_key] for f in fold_results if f.get(primary_key) is not None]
    if not fold_vals:
        return False
    passing = sum(1 for v in fold_vals if v >= threshold)
    return passing > len(fold_vals) / 2.0


# ---------------------------------------------------------------------------
# Feature-to-feature correlation & clustering
# ---------------------------------------------------------------------------

def compute_correlation_and_clusters(df: pd.DataFrame, features, cluster_distance: float = 0.3, corr_threshold: float = 0.50):
    if len(features) < 2:
        return [], {f: 1 for f in features}

    sub = df[features].replace([np.inf, -np.inf], np.nan)
    corr_matrix = sub.corr().abs().fillna(0.0)

    high_corr_pairs = []
    feats = list(corr_matrix.columns)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            val = corr_matrix.iloc[i, j]
            if val >= corr_threshold:
                raw_val = sub[feats[i]].corr(sub[feats[j]])
                high_corr_pairs.append({
                    "feature_a": feats[i],
                    "feature_b": feats[j],
                    "corr": float(raw_val) if np.isfinite(raw_val) else 0.0,
                })
    high_corr_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

    dist = 1.0 - corr_matrix.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against float noise
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    cluster_ids = fcluster(z, t=cluster_distance, criterion="distance")
    cluster_map = {feat: int(cid) for feat, cid in zip(feats, cluster_ids)}

    return high_corr_pairs, cluster_map


# ---------------------------------------------------------------------------
# Per-target orchestration
# ---------------------------------------------------------------------------

def analyze_target(df: pd.DataFrame, target: str, candidate_features, args):
    available = [f for f in candidate_features if f in df.columns]
    missing = sorted(set(candidate_features) - set(available))
    if missing:
        logger.warning(f"[{target}] Candidate features missing from dataset, skipping: {missing}")

    work_df = df.dropna(subset=available + [target]).reset_index(drop=True)
    if len(work_df) < 2000:
        logger.warning(f"[{target}] Not enough clean rows ({len(work_df)}) to analyze.")
        return

    splits = purged_wfo_splits(len(work_df), n_splits=args.folds, embargo=args.embargo)
    if not splits:
        logger.warning(f"[{target}] No valid purged CV splits (dataset too small for folds={args.folds}, embargo={args.embargo}).")
        return

    logger.info(f"\n{'=' * 100}")
    logger.info(f" SINGLE FEATURE IMPORTANCE: {target} "
                f"({len(available)} candidate features, {len(work_df)} rows, {len(splits)} purged folds) ".center(100, "="))
    logger.info("=" * 100)

    high_corr_pairs, cluster_map = compute_correlation_and_clusters(work_df, available, args.cluster_distance)

    rows = []
    for feature in available:
        fold_results = score_feature(work_df, feature, target, splits, args.max_rows_per_fold, args.skip_model_fit)
        agg_sfi = _agg_stat(fold_results, "sfi_auc")
        agg_raw = _agg_stat(fold_results, "raw_auc")
        agg_corr = _agg_stat(fold_results, "spearman_corr")

        selected = False
        if args.threshold is not None:
            selected = is_selected(agg_sfi, agg_raw, args.threshold, fold_results)

        rows.append({
            "feature": feature,
            "cluster": cluster_map.get(feature),
            "sfi_auc_mean": agg_sfi["mean"],
            "sfi_auc_worst": agg_sfi["worst"],
            "sfi_auc_std": agg_sfi["std"],
            "raw_auc_mean": agg_raw["mean"],
            "raw_auc_worst": agg_raw["worst"],
            "spearman_corr_mean": agg_corr["mean"],
            "n_folds": agg_sfi["n_folds"] or agg_raw["n_folds"],
            "selected": selected,
            "folds": fold_results,
        })

    def _rank_key(r):
        primary = r["sfi_auc_mean"] if r["sfi_auc_mean"] is not None else r["raw_auc_mean"]
        return -(primary if primary is not None else 0.0)

    rows.sort(key=_rank_key)

    if args.threshold is None:
        top_n_features = {r["feature"] for r in rows[:args.top_n]}
        for r in rows:
            r["selected"] = r["feature"] in top_n_features

    _print_table(rows)
    _print_correlation_table(high_corr_pairs)
    _write_outputs(target, rows, high_corr_pairs, cluster_map)


# ---------------------------------------------------------------------------
# Printing / output
# ---------------------------------------------------------------------------

def _print_table(rows):
    header = (f"{'Feature':<28} | {'Cluster':>7} | {'SFI AUC (mean)':>15} | "
              f"{'SFI AUC (worst)':>16} | {'Raw AUC':>9} | {'Spearman':>9} | {'Sel':>4}")
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows:
        sfi_mean = f"{r['sfi_auc_mean']:.4f}" if r["sfi_auc_mean"] is not None else "n/a"
        sfi_worst = f"{r['sfi_auc_worst']:.4f}" if r["sfi_auc_worst"] is not None else "n/a"
        raw_mean = f"{r['raw_auc_mean']:.4f}" if r["raw_auc_mean"] is not None else "n/a"
        corr_mean = f"{r['spearman_corr_mean']:+.4f}" if r["spearman_corr_mean"] is not None else "n/a"
        sel = "Y" if r["selected"] else ""
        logger.info(f"{r['feature']:<28} | {r['cluster']:>7} | {sfi_mean:>15} | "
                    f"{sfi_worst:>16} | {raw_mean:>9} | {corr_mean:>9} | {sel:>4}")


def _print_correlation_table(high_corr_pairs):
    if not high_corr_pairs:
        logger.info("\nNo feature pairs with |corr| >= 0.50.")
        return
    logger.info(f"\n{'Feature A':<28} | {'Feature B':<28} | {'Correlation':>12}")
    logger.info("-" * 74)
    for pair in high_corr_pairs:
        logger.info(f"{pair['feature_a']:<28} | {pair['feature_b']:<28} | {pair['corr']:>+12.4f}")


def _write_outputs(target, rows, high_corr_pairs, cluster_map):
    report_path = os.path.join(DATA_DIR, f"sfi_report_{target}.json")
    with open(report_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"\nWrote full SFI report ({len(rows)} features) to {report_path}")

    corr_path = os.path.join(DATA_DIR, f"sfi_correlation_{target}.json")
    with open(corr_path, "w") as f:
        json.dump({"high_corr_pairs": high_corr_pairs, "clusters": cluster_map}, f, indent=2)
    logger.info(f"Wrote correlation/cluster report to {corr_path}")

    selected_features = [r["feature"] for r in rows if r["selected"]]
    selected_path = os.path.join(DATA_DIR, f"sfi_selected_features_{target}.json")
    with open(selected_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Wrote selected feature list ({len(selected_features)} features) to {selected_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single Feature Importance (SFI) analysis - AFML Ch. 8.")
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS),
                         help="Comma-separated targets to evaluate: label_long,label_short,meta_label")
    parser.add_argument("--skip-model-fit", action="store_true",
                         help="Skip the single-feature XGBoost fit; report only the raw (no-model) AUC/Spearman stat.")
    parser.add_argument("--top-n", type=int, default=10,
                         help="Number of top features to mark as 'selected' (used when --threshold is not given).")
    parser.add_argument("--threshold", type=float, default=None,
                         help="AUC threshold for selection (overrides --top-n); requires a majority of folds to also clear it.")
    parser.add_argument("--cluster-distance", type=float, default=0.3,
                         help="Hierarchical clustering cut distance on 1-|corr| (lower = tighter clusters).")
    parser.add_argument("--folds", type=int, default=5, help="Number of purged walk-forward CV folds.")
    parser.add_argument("--embargo", type=int, default=1440, help="Embargo bars purged on both sides of each fold boundary.")
    parser.add_argument("--max-rows-per-fold", type=int, default=150_000,
                         help="Random subsample cap per fold (fixed seed) to bound runtime.")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = set(targets) - {*PRIMARY_TARGET_NAMES, META_TARGET_NAME}
    if unknown:
        logger.warning(f"Ignoring unknown targets: {sorted(unknown)}")

    primary_targets = [t for t in targets if t in PRIMARY_TARGET_NAMES]
    meta_targets = [t for t in targets if t == META_TARGET_NAME]

    if primary_targets:
        primary_df = load_primary_dataset()
        candidate_features = PRIMARY_FEATURES + TIER_A_FEATURES
        for target in primary_targets:
            analyze_target(primary_df, target, candidate_features, args)

    if meta_targets:
        meta_df = load_meta_dataset()
        for target in meta_targets:
            analyze_target(meta_df, target, META_FEATURES, args)


if __name__ == "__main__":
    main()
