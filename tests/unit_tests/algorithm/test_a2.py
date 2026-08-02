import pytest
from unittest.mock import MagicMock, patch
from workers.analyser.algorithm.primary_models import PrimaryModelRunner

@patch('xgboost.Booster')
def test_primary_models_runner(mock_xgb_booster):
    # Setup mock to avoid loading real GPU models during tests
    mock_long = MagicMock()
    mock_short = MagicMock()
    
    # XGB predict returns an array of probabilities, e.g. [0.65]
    mock_long.predict.return_value = [0.65]
    mock_short.predict.return_value = [0.45]
    
    mock_xgb_booster.side_effect = [mock_long, mock_short]
    
    # Initialize runner with dummy symbol
    # We patch os.path.exists to return True so it thinks the models are there
    with patch('os.path.exists', return_value=True):
        runner = PrimaryModelRunner("BTCUSDT")
        
    assert runner.is_loaded() == True
    
    # Mock features dictionary matching expected PRIMARY_FEATURES
    dummy_features = {
        "vpin": 0.2,
        "signed_kyle_lambda": -0.5,
        "bocpd_cp": 0.01,
        "price_z": 0.5,
        "net_volume_imbalance": 0.1,
        "cumulative_nvi_10": 0.5,
        "vol_regime": 1.0,
        "volume_z": 0.2,
        "cusum_ratio": 0.0,
        "bar_speed_z": 0.2,
        "trade_intensity_z": 1.2,
        "upper_wick_ratio": 0.1,
        "lower_wick_ratio": 0.1,
        "body_ratio": 0.8,
        "funding_rate": 0.0001,
        "funding_rate_z": 0.5,
        "nvi_x_cusum_direction": 0.0,
        "volatility_adjusted_nvi": 0.1,
        "bocpd_x_bar_speed_z": -0.002,
        "ffd_log_price_04": 0.001,
        "ffd_log_vol_04": -0.001,
        "hour_of_day": 14,
        "symbol_id": 0,
        "ema_cross": 0.01,
        "price_vs_ema200": 0.05,
        "atr_14": 150.0,
        "atr_percentile": 0.6,
        "bollinger_width": 200.0,
        "rsi_14": 55.0,
        "high_low_range_z": 0.5,
        # New Dynamic/Momentum Features (see shared/features.py PRIMARY_FEATURES)
        "vpin_delta_5": 0.01,
        "nvi_acceleration": 0.02,
        "amihud_illiquidity": 0.0001,
        "volatility_ratio": 1.1,
        "funding_rate_momentum": 0.0002,
    }
    
    prob_long, prob_short = runner.predict(dummy_features)
    
    assert prob_long == 0.65
    assert prob_short == 0.45
    
    # Verify the DMatrix was called
    mock_long.predict.assert_called_once()
    mock_short.predict.assert_called_once()
