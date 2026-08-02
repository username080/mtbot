import logging
from collections import deque
from typing import Optional, Dict
import numpy as np

from ..features.assembler import FeatureAssembler
from .gate import Gate
from .indicators import compute_live_ta
from .primary_models import PrimaryModelRunner
from .meta_model import MetaModelRunner
from .risk_manager import RiskManager
from shared.config_helper import fetch

logger = logging.getLogger(__name__)

class SignalEngine:
    """
    The SignalEngine is the core brain of the trading algorithm.
    It receives raw dollar bars, coordinates feature generation, 
    runs the XGBoost machine learning models, applies risk management,
    and returns Trade Orders (ENTRY/EXIT) back to the Coordinator.
    """
    def __init__(self, symbol: str, config: dict):
        self.symbol = symbol
        self.config = config
        
        # We maintain a rolling queue of recent bars to compute technical indicators (like EMA/RSI)
        self.bar_queue = deque(maxlen=fetch(config, "bar_queue_size"))
        
        # The Assembler handles all complex microstructure features (VPIN, Kyle's Lambda, Order Flow)
        self.assembler = FeatureAssembler(config)
        
        # Initialize the 4 main sub-modules of the algorithm
        self.primary_models = PrimaryModelRunner(symbol)     # XGBoost directional models (Long/Short)
        self.meta_model = MetaModelRunner(symbol, config)    # XGBoost meta-label model (Trade Sizing/Confidence)
        self.risk_manager = RiskManager(symbol, config)      # Position sizing, dynamic TP/SL, and Trailing Stops
        self.gate = Gate(config)                             # Pre-trade filter (Spread, Volatility, Cooldowns)
        
        # Internal State Tracking
        self.pending_signal = None      # Holds a trade that passed the Gate, waiting for the Patience filter
        self.active_position = False    # True if we currently hold a trade
        self.position_direction = 0     # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        self.bars_held = 0              # Tracks time-in-market for the Vertical Barrier (Time Stop)
        
        # Dynamic barrier limits (updated at trade entry based on current volatility)
        self.trade_tp_pct = 0.0098
        self.trade_sl_pct = 0.0082
        
        # Historical trackers for the Meta Model to analyze past performance
        self.last_signal_pnls = deque(maxlen=5) 
        self.last_trade_time = 0
        self.current_equity = 10000.0
        
        # 4H Tumbling Bucket State
        self.current_4h_bucket = 0
        self.ema_4h = 0.0
        self.regime = 0
        self.regime_duration = 0

    def update_funding_rate(self, rate: float):
        pass # Disabling funding rate to avoid API block issues
        
    def on_bar(self, packet: dict) -> Optional[Dict]:
        """
        Called every time a new Dollar Bar closes.
        Returns a dict representing an Order, or None if no action is taken.
        """
        self.bar_queue.append(packet)
        
        # Warmup period: We need at least 50 bars to calculate a 50-EMA before we can trade
        if len(self.bar_queue) < min(fetch(self.config, "bar_queue_size"), 50):
            return None
            
        close = float(packet["close"])
        close_time = int(packet["close_time"])
        
        # 4H Tumbling Bucket Tracker
        bucket_index = close_time // 14400000
        if "ema_4h" in packet and packet["ema_4h"] is not None and not np.isnan(packet["ema_4h"]):
            self.ema_4h = float(packet["ema_4h"])
            self.regime = int(packet["regime"])
            self.regime_duration = int(packet["regime_duration"])
            self.current_4h_bucket = bucket_index
        else:
            if self.current_4h_bucket == 0:
                self.current_4h_bucket = bucket_index
                self.ema_4h = close
            elif bucket_index > self.current_4h_bucket:
                last_close = float(self.bar_queue[-2]["close"]) if len(self.bar_queue) > 1 else close
                alpha = 2.0 / 201.0
                self.ema_4h = (last_close * alpha) + (self.ema_4h * (1 - alpha))
                
                new_regime = 1 if last_close > self.ema_4h else -1
                if new_regime == self.regime:
                    self.regime_duration += 1
                else:
                    self.regime = new_regime
                    self.regime_duration = 1
                    
                self.current_4h_bucket = bucket_index
        
        # =====================================================================
        # PHASE 1: FEATURE GENERATION & MODEL INFERENCE
        # =====================================================================
        
        # BACKTEST GPU OPTIMIZATION: 
        # If the backtest engine already fed the entire dataset through the XGBoost model 
        # using GPU vectorization, the probabilities are attached directly to the packet.
        # We grab them instantly and skip the slow Python-based feature loops.
        if "precomputed_long_prob" in packet and "precomputed_short_prob" in packet:
            prob_long = float(packet["precomputed_long_prob"])
            prob_short = float(packet["precomputed_short_prob"])
            features = dict(packet)
            
            # We still need to update these specific modules because the Gate and Patience 
            # filters rely on their internal state (like CUSUM streaks and Flow Divergence)
            self.assembler.flow_book.update(packet)
            self.assembler.bar_speed.update(packet)
            self.assembler.cusum.update(packet)
            
        # LIVE TRADING PATH:
        # We must manually calculate all microstructure and TA features for this single bar,
        # and then ask the XGBoost models to predict the probabilities in real-time.
        else:
            self.assembler.update(packet, self.bar_queue)
            if len(self.bar_queue) < fetch(self.config, "bar_queue_size"):
                return None
                
            features = self.assembler.get_features()
            features["symbol_id"] = 0 if self.symbol == "BTCUSDT" else 1
            
            ta_feats = compute_live_ta(self.bar_queue)
            features.update(ta_feats)
            
            # Run the heavy machine learning inference
            prob_long, prob_short = self.primary_models.predict(features)
            
        # Add 4H features to the feature dict for the meta model
        features["ema_4h"] = self.ema_4h
        features["regime"] = self.regime
        features["regime_duration"] = self.regime_duration
        
        # We need trend alignment. is_long_signal - 0.5 > 0 for LONG.
        is_long = 1 if prob_long > prob_short else 0
        features["trend_alignment"] = 1 if np.sign(is_long - 0.5) == np.sign(close - self.ema_4h) else 0

        # Continuous trend-extension measure (ATR-normalized distance from the 4H EMA) -
        # mirrors shared.meta_signal_builder.build_sequential_meta_signals exactly.
        atr_val = features.get("atr_14", 0.0)
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            features["trend_strength"] = 0.0
        else:
            features["trend_strength"] = (close - self.ema_4h) / atr_val
            
        # =====================================================================
        # PHASE 2: RISK MANAGEMENT (EXIT LOGIC)
        # =====================================================================
        if self.active_position:
            self.bars_held += 1
            
            # Calculate current unrealized return
            ret = (close - self.entry_price) / self.entry_price
            if self.position_direction == -1:
                ret = -ret
                
            # Ask the Risk Manager if we should close the trade (hits TP, SL, or Time Limit)
            exit_order = self.risk_manager.check_exit_conditions(
                ret=ret,
                bars_held=self.bars_held,
                position_direction=self.position_direction,
                prob_long=prob_long,
                prob_short=prob_short,
                features=features,
                trade_tp_pct=self.trade_tp_pct,
                trade_sl_pct=self.trade_sl_pct
            )
            
            if exit_order:
                logger.info(f"[{self.symbol}] EXITING: {exit_order['reason']}")
                self.last_signal_pnls.append(1 if ret > 0 else 0)
                
                # Reset state
                self.active_position = False
                self.position_direction = 0
                self.bars_held = 0
                return exit_order
                
            # If we hold a position, we skip evaluating new entries
            return None
            
        # =====================================================================
        # PHASE 3: ENTRY EXECUTION (PATIENCE FILTER)
        # =====================================================================
        if self.pending_signal:
            direction = self.pending_signal["direction"]
            self.pending_signal["bars_waited"] += 1
            
            # Patience Filter: Wait for microstructure flow (order book pressure) to align 
            # with our desired trade direction before pulling the trigger, ensuring better fills.
            if self.assembler.get_entry_patience_signals(direction):
                logger.info(f"[{self.symbol}] ENTERING: Patience OK.")
                
                # Ask Risk Manager for optimal position size (Kelly Criterion)
                order = self.risk_manager.calculate_position_size(
                    cusum_ratio=self.assembler.get_cusum_ratio(),
                    prob=self.pending_signal["prob"],
                    price=close,
                    current_equity=self.current_equity,
                    volatility=self.pending_signal["volatility"],
                    score=self.pending_signal["score"],
                    regime=self.pending_signal["regime"],
                    direction=direction
                )
                
                # Extract the dynamic TP/SL limits generated by the Triple Barrier method
                self.trade_tp_pct = order.pop("tp_pct")
                self.trade_sl_pct = order.pop("sl_pct")
                
                # Lock the state
                self.active_position = True
                self.position_direction = direction
                self.entry_price = close
                self.bars_held = 0
                
                self.pending_signal = None
                self.last_trade_time = close_time
                return order
                
            # If we wait too long without good flow, abandon the trade idea
            if self.pending_signal["bars_waited"] >= fetch(self.config, "max_patience_bars"):
                logger.info(f"[{self.symbol}] Pending signal expired due to patience limit.")
                self.pending_signal = None
            return None
            
        # =====================================================================
        # PHASE 4: SIGNAL DISCOVERY & GATE FILTERING
        # =====================================================================
        if self.primary_models.is_loaded() or ("precomputed_long_prob" in packet):
            direction = 0
            entry_thresh = fetch(self.config, "entry_prob_threshold")
            meta_config = fetch(self.config, "meta_model")
            meta_thresh = fetch(meta_config, "meta_threshold")
            
            # Regime Filter: If the market is choppy (short-term vol < long-term vol), require higher confidence
            vol_ratio = features["volatility_ratio"]
            
            if vol_ratio < 0.95:
                prob_boost = self.config.get("choppy_prob_boost", 0.1)
                entry_thresh += prob_boost
                meta_thresh += prob_boost
                logger.debug(f"[{self.symbol}] CHOPPY REGIME (Vol Ratio: {vol_ratio:.2f}): Boosting entry_thresh to {entry_thresh:.2f}")
                
            # Base directional threshold check
            if prob_long > entry_thresh and prob_long > prob_short:
                direction = 1
            elif prob_short > entry_thresh and prob_short > prob_long:
                direction = -1
                
            if direction != 0:
                meta_prob = prob_long if direction == 1 else prob_short
                
                # Optional: Pass signal through the secondary Meta-Label model for sizing/vetting
                if self.meta_model.enabled:
                    meta_prob = self.meta_model.evaluate(
                        features=features,
                        long_prob=prob_long,
                        short_prob=prob_short,
                        direction=direction,
                        current_time=close_time,
                        last_trade_time=self.last_trade_time,
                        recent_pnls=list(self.last_signal_pnls)
                    )
                    
                if meta_prob >= meta_thresh:
                    features["fundingRate"] = packet.get("fundingRate", 0.0)
                    target_prob = prob_long if direction == 1 else prob_short
                    opposing_prob = prob_short if direction == 1 else prob_long
                    signal_type = "LONG" if direction == 1 else "SHORT"
                    
                    active_positions = {self.symbol: 1} if self.active_position else {}
                    bars_since = (close_time - self.last_trade_time) / 60000 if self.last_trade_time > 0 else 999999
                    
                    # THE GATE: Final safety checks (spread, flash crashes, cooldowns, minimum TP floor)
                    passed, reason = self.gate.check(
                        signal_type=signal_type,
                        prob=target_prob,
                        opposing_prob=opposing_prob,
                        symbol=self.symbol,
                        active_positions=active_positions,
                        metrics=features,
                        recent_pnls=list(self.last_signal_pnls),
                        bars_since_last_trade=bars_since
                    )
                    
                    if passed:
                        vol_val = features["volatility"]
                        if vol_val is None or np.isnan(vol_val) or vol_val <= 0:
                            vol_val = 0.005
                            
                        # Queue the signal for the Patience Filter
                        self.pending_signal = {
                            "direction": direction,
                            "bars_waited": 0,
                            "prob": target_prob,
                            "score": meta_prob,
                            "regime": "BULL_TREND" if self.regime == 1 else "BEAR_TREND",
                            "volatility": vol_val
                        }
                        logger.info(f"[{self.symbol}] Gate passed for {signal_type} (Prob: {target_prob:.2f}). Waiting for patience...")
                    else:
                        logger.debug(f"[{self.symbol}] Gate BLOCKED {signal_type}: {reason}")
                        
        return None
