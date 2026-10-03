import os
import gc
import json
import pandas as pd
import numpy as np
import logging
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES
from shared.meta_signal_builder import build_sequential_meta_signals
from shared.purged_kfold import generate_oof_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

# Out-of-fold generation for long_prob/short_prob - matches tune_xgboost.py's
# purged-CV conventions (EMBARGO=1440). See shared/purged_kfold.py and
# documentation.md §9 for why these can't come from the final, fully-fit primary
# models (that would be in-sample prediction on the meta-model's own training set).
OOF_FOLDS = 5
OOF_EMBARGO = 1440

with open(os.path.join(os.path.dirname(__file__), "../../config.json"), "r") as f:
    _config = json.load(f)
_algo_config = _config.get("algorithm", {})
ENTRY_THRESHOLD = _algo_config.get("entry_prob_threshold", 0.65)
FORECAST_HORIZON_BARS = _algo_config.get("forecast_horizon_bars", 50)


def _load_oof_params(target_label: str):
    """Loads tune_xgboost.py's tuned hyperparameters for `target_label` (same
    params the production model is trained with in train_xgboost.py), split
    into the xgb.train params dict and the tuned round count."""
    param_name = "best_xgb_params_long.json" if target_label == "label_long" else "best_xgb_params_short.json"
    path = os.path.join(DATA_DIR, param_name)
    with open(path, "r") as f:
        loaded = json.load(f)

    best_n_rounds = int(loaded.pop("_best_n_rounds", 300))
    params = loaded
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "aucpr"
    params["tree_method"] = "hist"
    try:
        import xgboost as xgb
        params["device"] = "cuda" if xgb.core.build_info().get("USE_CUDA") else "cpu"
    except Exception:
        params["device"] = "cpu"

    return params, best_n_rounds


def build_meta_dataset():
    path_long_params = os.path.join(DATA_DIR, "best_xgb_params_long.json")
    path_short_params = os.path.join(DATA_DIR, "best_xgb_params_short.json")

    if not os.path.exists(path_long_params) or not os.path.exists(path_short_params):
        logger.warning("Tuned primary hyperparameters not found. Run tune_xgboost.py first. Skipping meta-label build for now.")
        return

    long_params, long_n_rounds = _load_oof_params("label_long")
    short_params, short_n_rounds = _load_oof_params("label_short")

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

        available_features = [f for f in PRIMARY_FEATURES if f in train_df.columns]

        # Out-of-fold long_prob/short_prob: each row's prediction comes from a
        # purged-K-fold model that never trained on it, so the probability
        # distribution the meta-model trains on matches what the primary
        # models actually produce on unseen data - not the artificially
        # confident in-sample distribution the final, fully-fit production
        # models would give on their own training rows (documentation.md §9).
        logger.info(f"  Generating out-of-fold Long predictions for {sym} ({OOF_FOLDS} purged folds)...")
        train_df['long_prob'] = generate_oof_predictions(
            train_df[available_features], train_df['label_long'],
            params=long_params, n_folds=OOF_FOLDS, embargo=OOF_EMBARGO, num_boost_round=long_n_rounds
        )
        gc.collect()

        logger.info(f"  Generating out-of-fold Short predictions for {sym} ({OOF_FOLDS} purged folds)...")
        train_df['short_prob'] = generate_oof_predictions(
            train_df[available_features], train_df['label_short'],
            params=short_params, n_folds=OOF_FOLDS, embargo=OOF_EMBARGO, num_boost_round=short_n_rounds
        )
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
