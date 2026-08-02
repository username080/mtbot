import pytest
from workers.analyser.algorithm.gate import Gate

def test_gate_circuit_breakers():
    config = {
        "gate": {
            "cooldown_bars_after_loss": 3,
            "min_atr_percentile": 0.20,
            "max_primary_prob": 0.70,
            "min_confidence_gap": 0.10,
            "max_funding_long": 0.0015,
            "min_funding_short": -0.0015,
            "max_spread": 0.0010,
            "flash_crash_z_threshold": 4.0,
            "min_tp_pct": 0.005
        },
        "tp_vol_mult": 2.0
    }
    
    gate = Gate(config)
    
    # Base passing case
    metrics = {
        "spread": 0.0005,
        "bar_speed_z": 1.0,
        "high_low_range_z": 1.0,
        "fundingRate": 0.0001,
        "volatility": 0.005
    }
    
    passed, reason = gate.check(
        signal_type="LONG",
        prob=0.65,
        opposing_prob=0.40,
        symbol="BTCUSDT",
        active_positions={},
        metrics=metrics,
        recent_pnls=[1.0, -1.0, 1.0]
    )
    assert passed == True
    
    # 1. Test Spread Failure
    metrics_spread = metrics.copy()
    metrics_spread["spread"] = 0.0020
    passed, reason = gate.check("LONG", 0.65, 0.40, "BTCUSDT", {}, metrics_spread)
    assert passed == False
    assert "Spread" in reason
    
    # 2. Test Flash Crash Failure
    metrics_flash = metrics.copy()
    metrics_flash["bar_speed_z"] = 5.0
    passed, reason = gate.check("LONG", 0.65, 0.40, "BTCUSDT", {}, metrics_flash)
    assert passed == False
    assert "Extreme volatility" in reason
    
    # 3. Test Cooldown Logic (3 consecutive losses)
    passed, reason = gate.check("LONG", 0.65, 0.40, "BTCUSDT", {}, metrics, recent_pnls=[-0.01, -0.02, -0.05], bars_since_last_trade=1)
    assert passed == False
    assert "Cooldown active" in reason
    
    # 4. Test Confidence Gap
    passed, reason = gate.check("LONG", 0.55, 0.50, "BTCUSDT", {}, metrics)
    assert passed == False
    assert "Confidence gap" in reason
