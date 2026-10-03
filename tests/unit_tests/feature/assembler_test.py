import math
from collections import deque

from workers.analyser.features.assembler import FeatureAssembler

# Features assembler.py used to compute but nothing ever consumed (removed in
# the dead-code cleanup pass - see documentation.md §9. Asserting
# their absence protects against silently re-introducing them.
REMOVED_DEAD_FEATURES = [
    "vpin_lag1", "volume_z_lag1", "vpin_mean_10", "day_of_week",
    "vpin_x_funding", "buy_sell_ratio", "bs_ratio_mean_5", "bs_ratio_mean_20",
    "vpin_acceleration", "vpin_bullish_conf", "bocpd_x_bar_speed_z",
    "vpin_percentile_24h",
]

# A representative subset of what's still actually used (PRIMARY_FEATURES /
# META_FEATURES) that assembler.py itself is responsible for producing.
EXPECTED_LIVE_FEATURES = [
    "vpin", "bocpd_cp", "vol_regime", "volume_z", "bar_speed_z",
    "ffd_log_price_04", "ffd_log_vol_04", "hour_of_day",
    "funding_rate", "funding_rate_z", "funding_rate_momentum",
    "net_volume_imbalance", "upper_wick_ratio", "lower_wick_ratio",
    "body_ratio", "cumulative_nvi_10", "nvi_acceleration", "vpin_delta_5",
    "trade_intensity_z", "nvi_x_cusum_direction", "volatility_adjusted_nvi",
]


def _make_config():
    return {
        "vpin_window": 5,
        "bocpd_hazard_rate": 0.01,
        "cusum_threshold_percentile": 95,
        "ffd_d_price": 0.40,
        "ffd_d_vol": 0.40,
        "ffd_threshold": 1e-4,
    }


def _make_bar(i: int, close: float) -> dict:
    return {
        "close": close,
        "high": close + 5.0,
        "low": close - 5.0,
        "open": close - 1.0,
        "volume": 10.0 + i,
        "buy_volume": 6.0 + i,
        "sell_volume": 4.0,
        "close_time": 1_700_000_000_000 + i * 60_000,
        "trade_count": 50 + i,
    }


def test_feature_assembler_produces_finite_features_after_warmup():
    assembler = FeatureAssembler(_make_config())
    bar_queue = deque(maxlen=200)

    close = 50_000.0
    for i in range(30):
        close += (1 if i % 2 == 0 else -0.5)
        bar = _make_bar(i, close)
        bar_queue.append(bar)
        assembler.update(bar, bar_queue, funding_rate=0.0002)

    feats = assembler.get_features()

    assert isinstance(feats, dict)
    for key in EXPECTED_LIVE_FEATURES:
        assert key in feats, f"Expected feature '{key}' missing from get_features() output"
        val = feats[key]
        assert isinstance(val, (int, float)), f"'{key}' is not numeric: {val!r}"
        assert math.isfinite(val), f"'{key}' is not finite: {val!r}"

    for key in REMOVED_DEAD_FEATURES:
        assert key not in feats, f"Dead feature '{key}' should have been removed but is still produced"


def test_feature_assembler_propagates_funding_rate():
    # Regression test for documentation.md §9 item 1's assembler-side half: the live bug
    # was that funding_rate never reached the assembler at all - this checks
    # the assembler itself correctly reflects whatever rate it's given.
    assembler = FeatureAssembler(_make_config())
    bar_queue = deque(maxlen=200)

    bar = _make_bar(0, 50_000.0)
    bar_queue.append(bar)
    assembler.update(bar, bar_queue, funding_rate=0.00123)

    feats = assembler.get_features()
    assert feats["funding_rate"] == 0.00123
