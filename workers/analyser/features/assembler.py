from typing import Dict, Deque
from collections import deque
import numpy as np
from shared.config_helper import fetch
from .vpin import VPIN
from .bocpd import BOCPD
from .flow_book import FlowBookDivergence
from .cusum import CUSUMFilter
from .bar_speed import BarSpeedDetector
from .regime import RegimeFeatures
from .ffd import FastFFD

class FeatureAssembler:
    """
    Orchestrates feature modules and produces feature vector for XGBoost.
    """
    def __init__(self, config: dict):
        self.vpin = VPIN(window=fetch(config, "vpin_window"))
        self.bocpd = BOCPD(hazard_rate=fetch(config, "bocpd_hazard_rate"))
        self.flow_book = FlowBookDivergence()
        self.cusum = CUSUMFilter(threshold_percentile=fetch(config, "cusum_threshold_percentile"))
        self.bar_speed = BarSpeedDetector()
        self.regime = RegimeFeatures()
        self.ffd_price = FastFFD(d=fetch(config, "ffd_d_price"), threshold=fetch(config, "ffd_threshold"))
        self.ffd_volume = FastFFD(d=fetch(config, "ffd_d_vol"), threshold=fetch(config, "ffd_threshold"))

        # History for advanced features
        self.history = deque(maxlen=20)
        self.trade_size_history = deque(maxlen=200)
        self.funding_history = deque(maxlen=200)
        self.last_timestamp = 0
        self.last_funding = 0.0

    def update(self, bar: dict, bar_queue: Deque[dict], funding_rate: float = 0.0):
        self.vpin.update(bar)
        self.bocpd.update(bar)
        self.flow_book.update(bar)
        self.cusum.update(bar)
        self.bar_speed.update(bar)
        self.regime.update(bar_queue)
        self.ffd_price.update(np.log(max(1e-9, bar["close"])))
        self.ffd_volume.update(np.log(1.0 + max(0.0, bar["volume"])))

        self.last_timestamp = bar["close_time"]
        self.last_funding = funding_rate

        # Calculate buy_sell_ratio and net_volume_imbalance
        buy_vol = bar["buy_volume"]
        sell_vol = bar["sell_volume"]
        vol = bar["volume"]
        nvi = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)

        # Candlestick rejection structure
        high = bar["high"]
        low = bar["low"]
        open_p = bar["open"]
        close_p = bar["close"]
        rng = max(1e-9, high - low)
        upper_wick_ratio = (high - max(open_p, close_p)) / rng
        lower_wick_ratio = (min(open_p, close_p) - low) / rng
        body_ratio = abs(close_p - open_p) / rng

        # Trade size / intensity
        if "trade_count" in bar:
            trade_count = bar["trade_count"]
        elif "num_trades" in bar:
            trade_count = bar["num_trades"]
        else:
            trade_count = bar["trades"]
        if trade_count <= 0:
            trade_count = max(1.0, vol)
        avg_trade_size = vol / float(trade_count)
        self.trade_size_history.append(avg_trade_size)
        self.funding_history.append(funding_rate)

        # Save base features for lags/rolling
        self.history.append({
            "vpin": self.vpin.value,
            "price_z": self.regime.price_z,
            "volume_z": self.regime.volume_z,
            "nvi": nvi,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "body_ratio": body_ratio,
            "avg_trade_size": avg_trade_size,
        })

    def get_features(self) -> Dict[str, float]:
        """Returns flat dict of features for XGBoost."""
        import pandas as pd

        # Base features
        feats = {
            "vpin": self.vpin.value,
            "bocpd_cp": self.bocpd.changepoint_probability,
            "vol_regime": self.regime.vol_regime,
            "volume_z": self.regime.volume_z,
            "bar_speed_z": self.bar_speed.speed_z,
            "ffd_log_price_04": self.ffd_price.value,
            "ffd_log_vol_04": self.ffd_volume.value
        }

        dt = pd.to_datetime(self.last_timestamp, unit="ms")
        feats["hour_of_day"] = dt.hour

        # NOTE: If funding rate cannot be reliably retrieved in live trading, remove funding_rate and funding_rate_z from features
        feats["funding_rate"] = self.last_funding
        if len(self.funding_history) > 10:
            f_hist = np.array(self.funding_history)
            f_std = np.std(f_hist)
            feats["funding_rate_z"] = float((self.last_funding - np.mean(f_hist)) / f_std) if f_std > 1e-9 else 0.0
        else:
            feats["funding_rate_z"] = 0.0

        if len(self.funding_history) >= 6:
            feats["funding_rate_momentum"] = self.last_funding - self.funding_history[-6]
        else:
            feats["funding_rate_momentum"] = 0.0

        # New Directional & Candlestick Features
        if len(self.history) > 0:
            curr = self.history[-1]
            feats["net_volume_imbalance"] = curr["nvi"]
            feats["upper_wick_ratio"] = curr["upper_wick_ratio"]
            feats["lower_wick_ratio"] = curr["lower_wick_ratio"]
            feats["body_ratio"] = curr["body_ratio"]

            # Cumulative NVI over last 10 bars
            recent_nvi = [h["nvi"] for h in list(self.history)[-10:]]
            feats["cumulative_nvi_10"] = sum(recent_nvi)

            if len(self.history) >= 13:
                recent_nvi_3_ago = [h["nvi"] for h in list(self.history)[-13:-3]]
                feats["nvi_acceleration"] = feats["cumulative_nvi_10"] - sum(recent_nvi_3_ago)
            else:
                feats["nvi_acceleration"] = 0.0

            if len(self.history) >= 6:
                feats["vpin_delta_5"] = feats["vpin"] - self.history[-6]["vpin"]
            else:
                feats["vpin_delta_5"] = 0.0
        else:
            feats["net_volume_imbalance"] = 0.0
            feats["upper_wick_ratio"] = 0.0
            feats["lower_wick_ratio"] = 0.0
            feats["body_ratio"] = 0.0
            feats["cumulative_nvi_10"] = 0.0
            feats["nvi_acceleration"] = 0.0
            feats["vpin_delta_5"] = 0.0

        # Trade Intensity Z-Score (Whale Footprint)
        if len(self.trade_size_history) > 10:
            ts_curr = self.trade_size_history[-1]
            ts_hist = np.array(self.trade_size_history)
            ts_std = np.std(ts_hist)
            feats["trade_intensity_z"] = float((ts_curr - np.mean(ts_hist)) / ts_std) if ts_std > 1e-9 else 0.0
        else:
            feats["trade_intensity_z"] = 0.0

        # Interaction Features (False-Positive Filters)
        cusum_val = getattr(self.cusum, "cusum", 0.0)
        cusum_dir = 1.0 if cusum_val > 0 else (-1.0 if cusum_val < 0 else 0.0)
        feats["nvi_x_cusum_direction"] = feats["net_volume_imbalance"] * cusum_dir
        feats["volatility_adjusted_nvi"] = feats["net_volume_imbalance"] / max(0.1, feats["vol_regime"])

        return feats

    def get_entry_patience_signals(self, direction: int) -> bool:
        """
        Returns True if patience filters (Flow-Book and Bar Speed) align with direction.
        """
        flow_ok = self.flow_book.agrees(direction)
        speed_ok = self.bar_speed.is_fast(-1.5)
        # Note: during backtest, quote_imbalance is 0 and time_elapsed is 1.0, so flow_ok and speed_ok are False.
        if not flow_ok and not speed_ok:
            if self.flow_book.streak == 0 and abs(self.bar_speed.speed_z) < 1e-6:
                return True
        return flow_ok or speed_ok

    def get_cusum_ratio(self) -> float:
        return self.cusum.ratio
