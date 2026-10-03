# MTBot — Repository Documentation

This document exists so anyone (or any agent) opening this repo cold can understand what it does and how the pieces fit together without re-reading every file. It reflects the state of the code as of this writing; if something here looks stale, trust the code over this doc and update this file.

## 1. System overview

MTBot is a dollar-bar-driven crypto futures trading bot (BTCUSDT, ETHUSDT) built around Marcos López de Prado's *Advances in Financial Machine Learning* (AFML) methodology: dollar bars instead of time bars, fractional differencing for stationarity, the triple-barrier method for labeling, meta-labeling for confidence/sizing, and purged/embargoed cross-validation to avoid the leakage that overlapping financial labels otherwise cause.

There are two parallel consumers of the same feature code:
- **Live/backtest execution**: `workers/analyser/algorithm/signal_engine.py` (`SignalEngine`), which consumes dollar bars one at a time and emits ENTRY/EXIT orders.
- **Offline training**: `scripts/ml/` and `scripts/derinanaliz/`, which process historical dollar bars in bulk to build labeled datasets, tune/train the XGBoost models, and diagnose their performance.

Both paths share `workers/analyser/features/assembler.py` (`FeatureAssembler`) so live inference and offline training compute features identically.

## 2. Data pipeline

- **`workers/data_pipeline/dollar_bar_builder.py`** (`DollarBarBuilder`): builds dollar bars from a stream of Binance aggTrades. A bar closes once accumulated dollar volume (`price * qty` summed) crosses a threshold; `update_threshold()` retargets that threshold dynamically off recent daily volume to aim for roughly 1 bar/minute on average. Tracks OHLC, buy/sell volume split (via `isBuyerMaker`), trade count, and tick imbalance per bar.
- **Funding rate** is merged onto bars by timestamp (`merge_asof`, backward direction) in `build_training_features.py` for offline training; live trading gets it via `SignalEngine.update_funding_rate(rate)`, which stores the rate and feeds it into `assembler.update()` on the next live bar (see Section 9 — this used to be a no-op, a real train/live feature-parity bug).
- **Where data lives**: `backtest/data/` — raw dollar bars (`dollar_bars_<SYMBOL>_all.parquet`), funding rate parquet files, labeled training sets (`training_<SYMBOL>.parquet`, `training_meta_combined.parquet`), tuned hyperparameter JSON files, and 4H kline data per symbol subfolder (used for the regime/trend features).

## 3. Feature engineering

`FeatureAssembler` (`workers/analyser/features/assembler.py`) orchestrates per-bar feature submodules, each in `workers/analyser/features/`:

| Module | What it computes |
|---|---|
| `vpin.py` | VPIN (Volume-Synchronized Probability of Informed Trading) — rolling mean of `\|buy_volume - sell_volume\| / volume`. |
| `cusum.py` | CUSUM filter over tick imbalance. In this codebase it's used as a **leverage-scaling signal** (`risk_manager.py`), not as AFML's usual event-sampling filter — see Section 8. |
| `bocpd.py` | Bayesian Online Changepoint Detection (Adams & MacKay) on the buy-volume ratio — not an AFML technique, a separate addition. |
| `ffd.py` | Fixed-Width Window Fractional Differencing (AFML Ch. 5) applied to log price and log volume, to achieve stationarity while preserving more memory than integer differencing. |
| `regime.py` | Long-window (200-bar) price/volume z-scores and a volatility-regime ratio. |
| `bar_speed.py` | Z-score of bar formation time — detects unusually fast bar formation. |
| `flow_book.py` | Agreement streak between execution flow (tick imbalance) and resting book (quote imbalance, falls back to tick imbalance in backtest where quote data isn't available). |

`assembler.get_features()` also derives candlestick structure (wick/body ratios), trade intensity z-scores, NVI (net volume imbalance) and its rolling/derivative variants, funding-rate z-score/momentum, and several interaction features (e.g. `nvi_x_cusum_direction`, `volatility_adjusted_nvi`).

Classic TA indicators (EMA cross, ATR, Bollinger width, RSI, Amihud illiquidity, volatility ratio) are computed separately: `compute_live_ta` in `workers/analyser/algorithm/indicators.py` for live/backtest, and a vectorized equivalent (`compute_ta_features`) in `scripts/ml/build_training_features.py` for bulk offline processing — the two must stay numerically consistent since one model reads from both paths.

`shared/features.py` is the single source of truth for which features each model actually uses:
- `PRIMARY_FEATURES`: ~28 features fed to the long/short primary models (microstructure + TA + regime).
- `META_FEATURES`: ~21 features fed to the meta-label model (primary model outputs like `long_prob`/`is_long_signal`, plus a subset of microstructure/regime/trend features).

## 4. Labeling

**Triple-barrier method** — `scripts/ml/build_training_features.py`: for each bar, a GPU (Numba CUDA) or CPU (Numba `prange`) kernel walks forward up to `forecast_horizon_bars` bars, checking whether a volatility-scaled take-profit or stop-loss barrier is touched first (a vertical/time barrier if neither is touched in time). Unlike the canonical AFML recipe (one primary "side" call + one triple-barrier), this produces **two independent binary labels**, `label_long` and `label_short`, each with its own symmetric TP/SL distance — effectively two separate one-sided labeling passes rather than one side-then-magnitude pass.

**Meta-labeling** — `scripts/ml/build_meta_labels.py` + `shared/meta_signal_builder.py` (`build_sequential_meta_signals`): takes the primary models' `long_prob`/`short_prob` predictions, selects bars that cross `entry_prob_threshold`, and — critically — only counts a bar as a "signal" if it isn't within `forecast_horizon_bars` of a previously selected signal. This mirrors the live bot's single-position lockout (`SignalEngine` never evaluates new entries while `active_position` is True) and fixes a real leakage bug documented in `meta_signal_builder.py`'s docstring: densely counting every qualifying bar as its own "signal" let `recent_signal_pnl_5` roll over heavily overlapping triple-barrier windows — near-copies of a trade's own future — which produced a meta-model that looked great offline and then blocked ~99.97% of everything live. `build_sequential_meta_signals` is called identically by both `build_meta_labels.py` (training) and `diagnostics.py` (evaluation) so they can't silently drift apart again.

## 5. The 3-model system

Two "primary" directional models plus one "meta" confidence model — conceptually AFML's primary+meta-labeling architecture, with the primary stage split into long/short rather than a single side-caller:

1. **Primary Long / Short** (`workers/analyser/algorithm/primary_models.py`, `PrimaryModelRunner`): two separate XGBoost boosters (`workers/analyser/models/xgb_long_v3.json`, `xgb_short_v3.json`), both consuming `PRIMARY_FEATURES`, each trained on its own triple-barrier label. Output: `prob_long`, `prob_short`.
2. **Meta model** (`workers/analyser/algorithm/meta_model.py`, `MetaModelRunner`): a third XGBoost (`xgb_meta_v1.json`) consuming `META_FEATURES` plus the primary outputs, predicting whether a primary signal is actually going to win. Output `meta_prob` is the real score used downstream for filtering/sizing. Can be disabled via config (`meta_model.enabled`), in which case the primary probability is used directly.

**`SignalEngine.on_bar`** (`workers/analyser/algorithm/signal_engine.py`) wires everything together per bar:
1. Update rolling bar queue + 4H EMA/regime tracker (trend state used both for entry gating and as a meta-model feature).
2. Run feature assembly + primary model inference (or read precomputed probabilities when the backtest engine has GPU-vectorized them).
3. If in a position: ask `RiskManager.check_exit_conditions` (TP/SL/time-stop/opposite-signal panic exit) — while a position is open, no new entries are evaluated at all.
4. If not in a position and a pending signal is queued: wait for the **patience filter** (`FeatureAssembler.get_entry_patience_signals` — order-flow/book direction agreeing with the trade direction, or bar-speed spike) before actually entering, up to `max_patience_bars`.
5. Otherwise: check directional threshold → optionally run the meta model → pass through **`Gate.check`** (`workers/analyser/algorithm/gate.py`: confidence gap, spread, flash-crash volatility z-score, 3-loss cooldown, funding rate extremes, minimum TP floor) → if it passes, queue as a pending signal.

**`RiskManager`** (`workers/analyser/algorithm/risk_manager.py`): position sizing via `(2p - 1) * half_kelly_fraction` (structurally AFML Ch. 10's probability-based bet sizing, not literal Kelly-with-odds despite being named "Kelly" in this codebase), leverage scaled by the CUSUM ratio, and dynamic TP/SL as a multiple of current volatility, widened on strong-trend confirmation.

## 6. Training / tuning pipeline

Run in this order to go from raw dollar bars to deployed models:

1. **`scripts/ml/build_training_features.py`** — assembles `PRIMARY_FEATURES`/`META_FEATURES` for every bar + applies triple-barrier labeling → `backtest/data/training_<SYMBOL>.parquet`.
2. **`scripts/ml/tune_xgboost.py`** — Optuna hyperparameter search for the primary long/short models using **purged walk-forward CV** (`purged_wfo_splits`: expanding train window, `embargo=1440` bars purged on both sides of each fold boundary, `aucpr` as the objective since triple-barrier labels are imbalanced) → `backtest/data/best_xgb_params_{long,short}.json`.
3. **`scripts/ml/train_xgboost.py`** — trains the final primary models on 100% of pre-2024 data for exactly the `best_n_rounds` found by tuning (no eval set, no early stopping — early stopping on the final run would just overfit to eval noise) → `workers/analyser/models/xgb_{long,short}_v3.json`.
4. **`scripts/ml/build_meta_labels.py`** — generates `long_prob`/`short_prob` via **purged K-Fold out-of-fold prediction** (`shared/purged_kfold.py`, using the tuned hyperparameters from step 2 — not the final step-3 models, which would be in-sample here), applies `build_sequential_meta_signals`, merges in 4H regime/trend features → `backtest/data/training_meta_combined.parquet`.
5. **`scripts/ml/tune_meta_model.py`** / **`train_meta_model.py`** — same tune-then-train pattern for the meta model → `workers/analyser/models/xgb_meta_v1.json`.

**Diagnostics** (`scripts/derinanaliz/`):
- `feature_importance.py` — embedded (MDI-style) XGBoost gain importance, full multivariate model, per target.
- `check_feature_correlations.py` — in-sample feature-to-label correlation and feature-to-feature collinearity (`|corr| >= 0.50`).
- `single_feature_importance.py` — **Single Feature Importance (AFML Ch. 8)**: every candidate feature scored completely on its own, out-of-sample, via the same purged-CV design as `tune_xgboost.py` — see Section 7.
- `diagnostics.py` — precision/recall/lift of the trained models on the held-out 2024+ OOS window, binned by probability.
- `analyze_meta_blocking.py`, `analyze_underperformance.py` — investigate why signals get filtered out / underperform.

**Optimization** (`scripts/optimization/`): `sweep_meta_threshold.py` (grid-searches the meta threshold against full backtest P&L via `backtest/dollar_bar_engine.py`), `optimize_all_parameters.py`.

## 7. `single_feature_importance.py`

New script (`scripts/derinanaliz/single_feature_importance.py`) implementing AFML's **Single Feature Importance** — the one gap among the three feature-importance methods AFML Ch. 8 describes (MDI, MDA, SFI); this repo had MDI (`feature_importance.py`) but not SFI.

For every candidate feature and target (`label_long`, `label_short`, `meta_label`), across purged/embargoed CV folds, it reports:
- `sfi_auc`: out-of-sample AUC of a classifier trained using **only that one feature**.
- `raw_auc` / `spearman_corr`: a no-model stat — the feature's raw values used directly as a ranking score.
- Feature-to-feature correlation and hierarchical clustering on the same candidate set, so near-duplicate top scorers are visible rather than quietly occupying multiple "top N" slots.

The candidate set is today's `PRIMARY_FEATURES`/`META_FEATURES` plus a set of "Tier A" technical/statistical features (MACD, Stochastic, Williams %R, ADX, CCI, Roll's implied spread, rolling return skew/kurtosis, return-sign entropy) computed purely from columns already in the training data — added only for the two primary targets, not the meta target (see the module's own docstring for why: the meta training set is sparse/non-contiguous signal rows, not dense bars, so rolling-window indicators can't be computed directly on it without a per-symbol dense recomputation the saved parquet doesn't support).

This is a **selection + reporting tool only** — it does not modify `shared/features.py`, retrain any production model, or call `tune_xgboost.py`/`train_xgboost.py`. It writes `sfi_report_<target>.json` (full per-feature stats), `sfi_correlation_<target>.json` (collinearity + clusters), and `sfi_selected_features_<target>.json` (the `--top-n`/`--threshold` selection marker) to `backtest/data/`.

## 8. Where this follows AFML closely vs. where it deviates

**Faithful to the book:**
- Dollar bars with dynamic threshold retargeting (Ch. 2).
- Fractional differencing via `FastFFD` (Ch. 5).
- Triple-barrier labeling with volatility-scaled barriers + vertical time barrier (Ch. 3).
- Meta-labeling, including catching and fixing two real leakage bugs: the overlapping-label signal-selection issue (Ch. 3.6, see Section 4) and in-sample primary-model predictions feeding the meta-training set (fixed via `shared/purged_kfold.py`'s out-of-fold generation — see Section 9).
- Purged CV with an embargo on both sides of each split boundary (Ch. 7): walk-forward in `tune_xgboost.py` and `single_feature_importance.py`, purged K-Fold (embargo on both sides of a held-out block, not just one) in `shared/purged_kfold.py`.

**Notable gaps (prescribed by AFML, not implemented here):**
- **Sample uniqueness / overlap weighting (Ch. 4)** — triple-barrier labels for adjacent bars overlap heavily (their forward-looking windows share most of their span), which breaks the IID assumption most ML training relies on. AFML's fix is to weight samples by average uniqueness. This repo already hit the exact overlap problem once (the meta-label leakage bug above) and patched it operationally for the *meta* model via sequential non-overlapping signal selection — but the *primary* models still train on every bar with plain, unweighted `xgb.DMatrix(X, label=y)`.
- **CUSUM used as a leverage multiplier, not as AFML's event-sampling filter.** AFML's CUSUM filter exists to *subsample* bars down to structural-break events before labeling/training, reducing redundancy. Here `CUSUMFilter` scales live leverage instead (`risk_manager.py`); training still uses every bar, so none of the sample-reduction benefit AFML uses CUSUM for is realized.
- **No MDA (permutation importance)** — only MDI (`feature_importance.py`) and now SFI (`single_feature_importance.py`) are implemented; MDA (shuffle-and-remeasure on the full model) is the third AFML Ch. 8 method and would catch interaction effects that SFI misses.
- **Fractional-differencing `d` is a fixed config constant**, not fit per series via AFML's ADF-stat-vs-`d` minimization procedure (Ch. 5).

**Notable deviations (not wrong, just different from the book):**
- **Two independent one-sided primary models** (long XGBoost + short XGBoost, each trained on its own triple-barrier touch label) rather than AFML's canonical single side-caller + one meta-label.
- **BOCPD** (`bocpd.py`) is not an AFML technique at all — a separate addition (Adams & MacKay).
- A `signed_kyle_lambda` feature is referenced in a comment (`signal_engine.py`) and two test mocks (`tests/unit_tests/algorithm/test_a2.py`, `test_a3.py`) but is never actually computed anywhere in `features/assembler.py` or `indicators.py` — a stale/planned-but-unfinished feature reference worth cleaning up or implementing.

## 9. Recently fixed issues

Three bugs found during an AFML-alignment audit of this codebase, fixed in the same pass as the dead-code cleanup below (code only — not yet run/retrained against real data):

1. **Live funding rate was always 0.0.** `SignalEngine.update_funding_rate()` was a no-op, and the live `on_bar()` path never passed a rate into the assembler — so `funding_rate`/`funding_rate_z`/`funding_rate_momentum` (3 of `PRIMARY_FEATURES`, also in `META_FEATURES`) were permanently 0 in live inference despite being trained on real, time-varying data. Backtest didn't catch this because it precomputes primary-model probabilities straight from the real `funding_rate` parquet column, bypassing the assembler entirely — a real backtest/live parity gap. **Fixed**: `update_funding_rate()` now stores the rate (`self.live_funding_rate`) and the live path passes it into `assembler.update(..., funding_rate=...)`. Scope note: there's no live coordinator in this repo to verify end-to-end against (see Section 10) — this fix makes the mechanism correct for whatever does call it in production.
2. **`Gate`'s funding-rate safety filter was dead code.** It read a `"fundingRate"` (camelCase) key that no packet — live or backtest — ever set; the real column is snake_case `funding_rate` everywhere. The extreme-funding-rate trade filter silently never fired, in either mode. **Fixed**: `gate.py` now reads `metrics.get("funding_rate", 0.0)`; the dead camelCase assignment in `signal_engine.py` was removed rather than patched.
3. **Meta-model trained on in-sample primary-model predictions.** `build_meta_labels.py` used to generate `long_prob`/`short_prob` for the meta-training set by predicting with the final, fully-fit primary models on the exact same data they were trained on — artificially confident probabilities, unrepresentative of what those models produce on genuinely unseen data (violates the same principle stacked generalization relies on). **Fixed**: new `shared/purged_kfold.py` implements AFML's actual Purged K-Fold CV (Ch. 7) — embargo on *both* sides of a held-out block, since unlike `tune_xgboost.py`'s walk-forward splits, a held-out fold here can sit anywhere in the timeline. `build_meta_labels.py` now generates out-of-fold predictions per symbol (5 folds, embargo=1440, using the already-tuned hyperparameters) instead. `train_xgboost.py` and the deployed model artifacts are untouched — only how the meta-training inputs are generated changed.

**Related dead-code cleanup** in the same pass: 12 computed-but-unused features removed from `assembler.py`'s `get_features()` (including an exact duplicate — `vpin_acceleration` computed the identical formula to `vpin_delta_5` under a different name, one of which was actually used), plus `compute_ffd_series` (`ffd.py`), 7 unused methods on `backtest/portfolio.py` (and the trailing-stop state they existed to expose, also unused), `FlowBookDivergence.agreement_streak` / `VPIN.flow_direction`, and ~18 unused imports repo-wide.

**Test coverage added** for all of the above: `tests/unit_tests/shared/test_purged_kfold.py` (new — fold coverage/disjointness/embargo correctness, OOF prediction sanity), `tests/unit_tests/feature/assembler_test.py` (new — post-cleanup `FeatureAssembler` still produces valid features, removed features stay removed, funding rate propagates), and regression cases added to `tests/unit_tests/algorithm/test_a5.py` (Gate funding filter actually blocks) and `test_a6.py` (`SignalEngine` funding-rate wiring). Full suite: 21/21 passing as of this writing.

## 10. Known gotchas

- **`config.json` does not currently exist on disk in this checkout** (not tracked by git either), but several scripts expect it at the repo root and will throw `FileNotFoundError` if run as-is: `scripts/ml/build_training_features.py`, `scripts/ml/build_meta_labels.py`, `scripts/derinanaliz/diagnostics.py`, `scripts/optimization/sweep_meta_threshold.py`. The `derinanaliz` diagnostic scripts that don't need live-algorithm config (`feature_importance.py`, `check_feature_correlations.py`, `single_feature_importance.py`) deliberately avoid depending on it and read training parquet files directly.
- **There is no live coordinator/executor in this checkout** — `SignalEngine` is only ever instantiated by `backtest/dollar_bar_engine.py`, `backtest/multi_engine.py`, and test mocks. Whatever process is meant to run it live (polling Binance, calling `update_funding_rate()` periodically, etc.) isn't part of this repo.
- Training/backtest data in `backtest/data/` is large (dollar bar / training parquet files are 250-470MB each) and not something to casually reload multiple times in one process.
- `workers/analyser/features/__init__.py` re-exports its submodule classes (`VPIN`, `BOCPD`, etc.) as a package API, but nothing in this repo currently imports from the package root (`from workers.analyser.features import X`) — every consumer imports the submodule directly. Harmless, just worth knowing if you're tracing what's "live" code.
