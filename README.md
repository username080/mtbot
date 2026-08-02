# MTBot: Binance Futures Quant Trading Bot

A quantitative trading system for Binance Futures, built around volume-based ("dollar") bars, a dual XGBoost primary/meta-labeling model pipeline, and a multiprocess, shared-memory live execution engine. This repo contains the real strategy code, the backtesting engine, and the research/diagnostic tooling used to develop and evaluate it — not a stripped-down demo.

**Not included** (see `.gitignore`): trained model weights (`workers/analyser/models/*.json`), historical tick/bar datasets (`*.parquet`, `backtest/data/`), and anything credential-related (`.env`). Everything else — the live trading engine, the backtest engine, and the model-training/diagnostic scripts — is here.

## Architecture

- **`core/`** — Runtime infrastructure: a local Redis process manager (`redis_c.py`), a POSIX shared-memory service for cross-process portfolio/position state (`shm.py`), the internal packet schema/dispatcher, and a thin Binance client wrapper.
- **`manager/`** — The top-level orchestrator (`manager.py`) that wires up Redis, shared memory, logging, and heartbeat tracking, plus a pub/sub-style packet router (`redis_manager.py`).
- **`workers/data_pipeline/`** — Connects to Binance (REST + WebSocket), aggregates raw trades into **dollar bars** (volume-threshold bars instead of fixed-time candles) via `dollar_bar_builder.py`.
- **`workers/analyser/algorithm/`** — The actual trading logic:
  - `signal_engine.py` — orchestrates feature generation, model inference, entry/exit logic, and state per symbol.
  - `primary_models.py` — dual XGBoost models (long/short) trained with triple-barrier labeling.
  - `meta_model.py` — a secondary XGBoost "meta-labeling" gate that vets primary signals before sizing.
  - `gate.py` — pre-trade rule filter (spread, flash-crash volatility, funding rate, cooldowns, minimum TP floor).
  - `risk_manager.py` — Kelly-criterion position sizing and dynamic TP/SL/time-stop exit logic.
  - `workers/analyser/features/` — microstructure feature modules (VPIN, CUSUM, BOCPD, order-flow imbalance, fractional differencing, etc.) feeding the models.
- **`workers/executer/`** — Receives trade orders and manages real positions via the Binance Futures REST API.
- **`backtest/`** — A vectorized (GPU-batched XGBoost inference) event-driven backtest engine (`dollar_bar_engine.py`) that runs the exact same `SignalEngine` used live against historical data, plus a portfolio/trade-log simulator (`portfolio.py`).
- **`scripts/ml/`** — Feature/label construction (triple-barrier labeling), primary and meta-model training and hyperparameter tuning.
- **`scripts/derinanaliz/`** — Diagnostic tooling used to evaluate the strategy: out-of-sample precision/recall/lift tables, feature importance and correlation analysis, meta-gate impact analysis (with/without the meta-labeling filter), and loss/underperformance breakdowns.

## Why this design

1. **Multiprocessing over multithreading** — Python's GIL makes threaded bots stall during ML inference. Data ingestion, analysis, and execution run as separate OS processes.
2. **Zero-latency shared memory** — portfolio/position state is synchronized via POSIX shared memory rather than sockets or a database.
3. **Dollar bars over time bars** — bars are built from cumulative traded volume, not clock time, which keeps bar statistics more stationary across quiet vs. volatile periods.
4. **Triple-barrier labeling + meta-labeling** — primary models predict direction; a second-stage model is trained to filter which of those signals are actually worth taking, following the López de Prado meta-labeling approach.

## Weaknesses & caveats (honest, not marketing)

- **`main.py`'s web dashboard reference is currently stale.** It launches `web.app:app` via uvicorn, but the `web/` FastAPI dashboard was removed during the algorithm rewrite and hasn't been rebuilt yet. The core trading engine (`Manager`, workers, signal engine) does not depend on it, but `main.py` as written will fail on that subprocess step until the dashboard is either restored or that line is removed.
- **OS dependency**: POSIX shared memory and Unix domain sockets mean this runs on Linux/macOS; Windows needs WSL.
- **Complexity**: this isn't a plug-and-play script — extending it requires comfort with `multiprocessing`, Redis pub/sub, and the feature/model pipeline.
- **The current model edge is weak.** Backtesting and diagnostics (see `scripts/derinanaliz/`) show the primary models have only thin, low-confidence predictive power, and the meta-labeling model in particular is close to non-discriminative on out-of-sample data. This is the open problem — see Roadmap.
- **No crash-recovery supervisor.** A single bad bar can no longer take down the Analyser process (it's caught and logged, see `strategy_coordinator.py`), but if a worker process dies for any other reason (unhandled exception elsewhere, Redis drop, OOM, etc.), nothing currently restarts it — `Manager` starts each worker once and doesn't watch for a dead child. This is a known, intentional gap, not an oversight: `tests/system_tests/test_crash_supervision.py` has a real (`xfail(strict=True)`) test already describing the desired behavior. It's deliberately not implemented naively, because a restarted `Analyser` would start with fresh in-memory state (`active_position = False`, etc.) with no awareness of a real position still open on the exchange — auto-restart needs to reconcile actual exchange/position state on startup first, or it can make a mid-crash situation worse instead of better.

## Setup

1. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure credentials**
   ```bash
   cp .env.example .env
   ```
   Fill in `BINANCE_API_KEY` / `BINANCE_API_SECRET` (leave blank for dry-run/mock mode) and set `WEB_USER`/`WEB_PASS` before any real deployment.

3. **Strategy configuration**: `config.json` at the repo root holds all live strategy parameters (entry thresholds, leverage, gate rules, TP/SL multipliers, meta-model threshold, etc.) and is committed as-is — it documents the actual tuned parameters this strategy runs with, not just an example.

4. **Run the bot**
   ```bash
   python main.py
   ```
   Or via Docker: `docker compose up -d --build` (see `docker-compose.yml` / `Dockerfile`).

## Research & backtesting

```bash
# Build historical dollar bars from raw tick data, then feature/label them
python scripts/data/build_dollar_bars_historical.py
python scripts/ml/build_training_features.py

# Train primary + meta models
python scripts/ml/train_xgboost.py
python scripts/ml/train_meta_model.py

# Run diagnostics against the trained models
python scripts/derinanaliz/diagnostics.py
python scripts/derinanaliz/feature_importance.py
python scripts/derinanaliz/analyze_meta_blocking.py
```

`scripts/derinanaliz/` and `scripts/optimization/` expect `backtest/data/` to contain the parquet datasets produced by the build scripts above (not included in this repo — see `.gitignore`).

## Roadmap

The current strategy is a supervised-learning pipeline (XGBoost primary models + XGBoost meta-labeling gate, both trained on triple-barrier labels). Diagnostics show this approach has a real but thin edge, and a large share of the model's decisions collapse to a live/backtest execution-timing gap that needs closing before the supervised approach can be judged fairly. The next planned direction is moving toward a **reinforcement-learning-based** approach — framing entry/exit/sizing as a sequential decision problem directly, rather than a two-stage classification pipeline — using this codebase's dollar-bar pipeline, feature modules, and backtest engine as the environment/infrastructure layer.

## License
MIT License. See [LICENSE](LICENSE) for more details.
