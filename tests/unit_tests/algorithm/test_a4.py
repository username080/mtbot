import pytest
from workers.analyser.algorithm.risk_manager import RiskManager

def test_risk_manager_kelly_sizing():
    config = {
        "base_leverage": 2.0,
        "max_leverage": 5.0,
        "half_kelly_fraction": 0.5,
        "tp_vol_mult": 2.0,
        "sl_vol_mult": 2.0
    }
    
    rm = RiskManager("BTCUSDT", config)
    
    # prob = 0.60 -> raw_kelly = (2 * 0.60 - 1) = 0.20
    # fraction = 0.20 * 0.5 = 0.10
    # So 10% of equity
    equity = 10000.0
    price = 50000.0
    
    order = rm.calculate_position_size(
        cusum_ratio=0.5,  # base_lev * (0.5 + 0.5*0.5) = 2.0 * 0.75 = 1.5x leverage
        prob=0.60,
        price=price,
        current_equity=equity,
        volatility=0.01,
        score=0.75,
        regime="BULL_TREND",
        direction=1
    )
    
    # 10,000 * 0.10 * 1.5 = 1500 notional
    # Qty = 1500 / 50000 = 0.03
    assert abs(order["quantity"] - 0.03) < 0.0001
    assert order["leverage"] == 1.5
    assert order["side"] == "BUY"
    assert order["tp_pct"] == 0.02  # 0.01 vol * 2.0
    assert order["sl_pct"] == 0.02
    
def test_risk_manager_trend_boost():
    config = {
        "forecast_horizon_bars": 50,
        "panic_exit_threshold": 0.60
    }
    
    rm = RiskManager("BTCUSDT", config)
    
    # A feature set matching a strong long trend
    features = {
        "price_vs_ema200": 0.01,  # > 0.005
        "ema_cross": 1.0,         # > 0
        "momentum_20": 0.02       # > 0
    }
    
    # Initially the TP was 2% (0.02). The boost is 1.8x, so TP should be 0.036
    # And we are manually feeding in a return of 0.037 which should trigger Take Profit
    exit_order = rm.check_exit_conditions(
        ret=0.037,
        bars_held=10,
        position_direction=1,
        prob_long=0.65,
        prob_short=0.35,
        features=features,
        trade_tp_pct=0.02,
        trade_sl_pct=0.02
    )
    
    assert exit_order is not None
    assert exit_order["reason"] == "Take Profit Hit"
