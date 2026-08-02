"""
Analyser Strategy Coordinator — Imbalance Bar Architecture.

Subscribes to completed Macro Bars from the UDS pipeline, calculates rolling ATR,
queries the split-mutex Shared Memory state, and runs the 4-phase AnalyserBlackbox.
Orders are dispatched to the Executer process via the UDS ExecutorService.
"""
import logging
import time
import json
import os
from collections import deque
from typing import Deque, Dict, List, Optional
import pandas as pd
from core.services.executer_service import ExecutorService
from core.shm import shm_service
from core.packets.m_packet import MPacket
from workers.analyser.algorithm.signal_engine import SignalEngine
from shared.config_helper import fetch

logger = logging.getLogger(__name__)

# Lookback queue size for feature engineering (matches LOOKBACK + WARMUP_BARS)
CANDLE_QUEUE_SIZE = 160

class AnalyserStrategyCoordinator:
    """
    Coordinates raw aggTrade processing.
    """
    def __init__(
        self,
        worker_instance,
        historical_warmup_limit: int = CANDLE_QUEUE_SIZE,
    ):
        self.worker = worker_instance

        # ── Configuration ──────────────────────────────────────────────
        self.config = {}
        config_path = os.path.join(os.path.dirname(__file__), "../../../config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
        except Exception as exc:
            self.worker.send_log(f"[Coordinator] Error loading config: {exc}", errnum=1)
            
        self.time_stops_config = fetch(self.config, "time_stops_minutes")

        # ── State Trackers ─────────────────────────────────────────────
        self.cached_prices: Dict[str, float] = {}
        self._warmed_up: Dict[str, bool] = {}
        self.last_macro_times: Dict[str, float] = {}

        # Rolling ATR states
        self.prev_closes: Dict[str, float] = {}
        self.rolling_tr: Dict[str, Deque[float]] = {}

        algo_config = fetch(self.config, "algorithm")
        self.engines: Dict[str, SignalEngine] = {}
        symbols = fetch(algo_config, "symbols")
        for sym in symbols:
            self.engines[sym] = SignalEngine(sym, algo_config)

        # ── Execution Service Client ──────────────────────────────────
        self.executor_service = ExecutorService()

    # ── Time Stop (Microbar Supervisor) ────────────────────────────────
    def check_time_stops(self):
        """
        Microbar Supervisor logic:
        If a position is held and no macro bar has formed within the Time Stop limit,
        trigger a Best Price Exit (SMART_EXIT).
        """
        portfolio = shm_service.read_portfolio()
        positions = portfolio.get("positions", {})
        
        now = time.time()
        for symbol, qty in positions.items():
            if abs(qty) < 1e-8:
                continue
                
            last_time = self.last_macro_times.get(symbol, 0.0)
            if last_time == 0.0:
                continue # Unknown state
                
            # Get configured minutes
            try:
                limit_minutes = self.time_stops_config[symbol]
            except KeyError:
                limit_minutes = self.time_stops_config["DEFAULT"]
            limit_seconds = limit_minutes * 60
            
            if now - last_time > limit_seconds:
                self.worker.send_log(f"[Microbar Supervisor] TIME STOP TRIGGERED for {symbol}. No bar for {limit_minutes} minutes. Firing SMART_EXIT.")
                
                # Prevent multiple triggers
                self.last_macro_times[symbol] = now + 999999.0
                
                exit_side = "SELL" if qty > 0 else "BUY"
                self._dispatch_order({
                    "action": "SMART_EXIT",
                    "symbol": symbol,
                    "quantity": abs(qty),
                    "side": exit_side,
                    "reason": "Time Stop"
                })

    # ── Dollar Bar Processing ──────────────────────────────────────────

    def process_dollar_bar(self, packet: dict):
        """
        Called when a complete dollar bar packet is received from the pipeline.
        """
        symbol = packet["symbol"]
        if symbol not in self.engines:
            return
            
        engine = self.engines[symbol]

        # Log bar completion
        self.worker.send_log(f"[{symbol}] Dollar Bar Closed: Close={packet['close']}, Vol={packet['volume']}, Imbalance={packet['tick_imbalance']}")

        # A single malformed/unexpected bar must not be able to kill the whole
        # Analyser process (mirrors BaseWorker.dispatch()'s per-handler
        # try/except for Executer): if it did, any open position would go
        # unmanaged (no more TP/SL/time-stop checks) until someone notices
        # and restarts the bot by hand - there's no auto-restart supervisor
        # in Manager yet (see tests/system_tests/test_crash_supervision.py).
        try:
            order = engine.on_bar(packet)
        except Exception as exc:
            self.worker.send_log(f"[{symbol}] Error processing dollar bar in SignalEngine: {exc}", errnum=1)
            return

        if order:
            self._dispatch_order(order)

    def _dispatch_order(self, order: dict):
        self.worker.send_log(f"[Coordinator] Sending order command to Executer: {order}")
        action = order.get("action")
        payload = {k: v for k, v in order.items() if k != "action"}
        packet = MPacket.create(self.worker.worker_name, "Executer", action, payload=payload)
        success = self.executor_service.send_order(packet.to_dict())
        if not success:
            self.worker.send_log(f"[Coordinator] Failed to send order command over IPC UDS.", errnum=1)
