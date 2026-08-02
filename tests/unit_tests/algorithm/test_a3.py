import pytest
from unittest.mock import MagicMock, patch
from workers.analyser.algorithm.meta_model import MetaModelRunner

@patch('xgboost.Booster')
def test_meta_model_runner(mock_xgb_booster):
    mock_meta = MagicMock()
    # Mock return value [score]
    mock_meta.predict.return_value = [0.85]
    mock_xgb_booster.return_value = mock_meta
    
    config = {
        "meta_model": {
            "enabled": True,
            "meta_threshold": 0.55
        }
    }
    
    with patch('os.path.exists', return_value=True):
        runner = MetaModelRunner("BTCUSDT", config)
        
    assert runner.enabled == True
    
    dummy_features = {
        "atr_percentile": 0.6,
        "vol_regime": 1.0,
        "bar_speed_z": 0.2,
        "vpin_acceleration": 0.05,
        "hour_of_day": 14,
        "day_of_week": 2,
        "rsi_14": 55.0,
        "ema_cross": 0.01,
        "net_volume_imbalance": 0.1,
        "vpin": 0.2,
        "buy_sell_ratio": 0.6,
        "trade_intensity_z": 1.2,
        "signed_kyle_lambda": -0.5,
        "upper_wick_ratio": 0.1,
        "lower_wick_ratio": 0.1,
        "ffd_log_price_04": 0.001,
        # Remaining META_FEATURES not auto-populated by evaluate() (see shared/features.py)
        "bollinger_width": 200.0,
        "volatility_adjusted_nvi": 0.1,
        "atr_14": 150.0,
        "funding_rate": 0.0001,
        "bocpd_cp": 0.01,
        "regime": 1,
        "regime_duration": 5,
        "ema_4h": 50000.0,
        "trend_alignment": 1,
        "trend_strength": 0.5,
    }
    
    # Simulate a LONG signal evaluation
    # current_time is 1 hour (3600000 ms) after last trade
    current_time = 1700003600000
    last_trade_time = 1700000000000
    recent_pnls = [1, 0, 1]  # 2 wins, 1 loss -> avg 0.666
    
    meta_score = runner.evaluate(
        features=dummy_features,
        long_prob=0.60,
        short_prob=0.40,
        direction=1,
        current_time=current_time,
        last_trade_time=last_trade_time,
        recent_pnls=recent_pnls
    )
    
    assert meta_score == 0.85
    mock_meta.predict.assert_called_once()
    
    # Verify disabled fallback
    config["meta_model"]["enabled"] = False
    with patch('os.path.exists', return_value=True):
        disabled_runner = MetaModelRunner("BTCUSDT", config)
        
    fallback_score = disabled_runner.evaluate(
        features=dummy_features,
        long_prob=0.60,
        short_prob=0.40,
        direction=1,
        current_time=current_time,
        last_trade_time=last_trade_time,
        recent_pnls=recent_pnls
    )
    
    # Since it's a LONG signal (direction 1), it should fallback to long_prob (0.60)
    assert fallback_score == 0.60
