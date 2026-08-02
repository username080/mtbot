"""
Shared Feature Definitions for MTBot ML Pipeline and Live Execution Engine.
Centralizing feature lists here ensures consistency across training, tuning, diagnostics, testing, and live inference.
"""

PRIMARY_FEATURES = [
    "bollinger_width", 
    "upper_wick_ratio", 
    "volatility_adjusted_nvi", 
    "net_volume_imbalance", 
    "vpin", 
    "atr_14", 
    "body_ratio", 
    "ffd_log_price_04", 
    "rsi_14", 
    "vol_regime", 
    "hour_of_day", 
    "ema_cross", 
    "lower_wick_ratio", 
    "cumulative_nvi_10", 
    "funding_rate", 
    "bocpd_cp", 
    "funding_rate_z", 
    "nvi_x_cusum_direction", 
    "ffd_log_vol_04", 
    "trade_intensity_z", 
    "atr_percentile", 
    "volume_z", 
    "bar_speed_z",
    
    # New Dynamic/Momentum Features
    "vpin_delta_5",
    "nvi_acceleration",
    "amihud_illiquidity",
    "volatility_ratio",
    "funding_rate_momentum"
]

META_FEATURES = [
    "is_long_signal",
    "long_prob",
    "recent_signal_pnl_5",
    "net_volume_imbalance",
    "rsi_14",
    "hours_since_last_trade",
    "confidence_gap",
    "ema_cross",
    "lower_wick_ratio",
    "vol_regime",
    "vpin",
    "bollinger_width",
    "volatility_adjusted_nvi",
    "atr_14",
    "funding_rate",
    "bocpd_cp",
    "regime",
    "regime_duration",
    "ema_4h",
    "trend_alignment",
    "trend_strength"
]

def calculate_recent_pnl(pnls: list) -> float:
    """
    Computes the mean win rate of the recent trades.
    If no trades are present, defaults to 0.5 (neutral).
    """
    if len(pnls) > 0:
        return sum(pnls) / len(pnls)
    return 0.5
