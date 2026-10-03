from unittest.mock import patch
from workers.analyser.algorithm.signal_engine import SignalEngine

@patch('workers.analyser.algorithm.signal_engine.compute_live_ta')
@patch('workers.analyser.algorithm.primary_models.PrimaryModelRunner')
@patch('workers.analyser.algorithm.meta_model.MetaModelRunner')
@patch('workers.analyser.algorithm.gate.Gate')
@patch('workers.analyser.algorithm.risk_manager.RiskManager')
def test_signal_engine_orchestration(mock_rm, mock_gate, mock_meta, mock_pm, mock_ta):
    config = {
        "bar_queue_size": 10,
        "entry_prob_threshold": 0.52,
        "max_patience_bars": 5,
        "tp_vol_mult": 2.0,
        "vpin_window": 50,
        "lambda_window": 20,
        "bocpd_hazard_rate": 0.01,
        "cusum_threshold_percentile": 95,
        "ffd_d_price": 0.40,
        "ffd_d_vol": 0.40,
        "ffd_threshold": 1e-4,
        "regime_thresholds": {
            "choppy_adx_atr_ratio": 0.35,
            "choppy_prob_boost": 0.05
        },
        "meta_model": {
            "enabled": False,
            "meta_threshold": 0.55
        },
        "gate": {
            # Dummy gate config to prevent fetch errors if Gate isn't fully mocked
            "cooldown_bars_after_loss": 3,
            "min_atr_percentile": 0.20,
            "max_primary_prob": 0.70,
            "min_confidence_gap": 0.10,
            "max_funding_long": 0.0015,
            "min_funding_short": -0.0015,
            "max_spread": 0.0010,
            "flash_crash_z_threshold": 4.0,
            "min_tp_pct": 0.005,
            "tp_vol_mult": 2.0
        }
    }

    # Setup Mocks
    mock_ta.return_value = {
        "adx_atr_ratio": 0.5,
        "ema_cross": 1.0,
        "price_vs_ema200": 0.01,
        "volatility": 0.005,
        "volatility_ratio": 1.0
    }

    mock_pm_instance = mock_pm.return_value
    mock_pm_instance.is_loaded.return_value = True
    # We will trigger a LONG signal
    mock_pm_instance.predict.return_value = (0.65, 0.45)

    mock_meta_instance = mock_meta.return_value
    mock_meta_instance.enabled = False
    mock_meta_instance.evaluate.return_value = 0.65

    mock_gate_instance = mock_gate.return_value
    mock_gate_instance.check.return_value = (True, "Pass")

    mock_rm_instance = mock_rm.return_value
    mock_rm_instance.calculate_position_size.return_value = {
        "action": "ENTRY",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.1,
        "leverage": 2.0,
        "tp_pct": 0.02,
        "sl_pct": 0.02
    }

    engine = SignalEngine("BTCUSDT", config)
    # Inject mocked instances
    engine.primary_models = mock_pm_instance
    engine.meta_model = mock_meta_instance
    engine.gate = mock_gate_instance
    engine.risk_manager = mock_rm_instance

    # Regression test for documentation.md §9 item 1: update_funding_rate() used to be a
    # no-op, so the live path never actually fed a real funding rate into the
    # assembler. Set a non-zero rate before any bars come in and verify it
    # reaches the assembler on the first bar that completes warmup.
    engine.update_funding_rate(0.00075)
    assert engine.live_funding_rate == 0.00075

    # 1. Fill the queue (size 10)
    for i in range(9):
        engine.on_bar({
            "close": 50000.0,
            "close_time": 1000 + i,
            "volume": 1.0,
            "buy_volume": 0.5,
            "sell_volume": 0.5,
            "high": 50100.0,
            "low": 49900.0,
            "open": 50000.0,
            "tick_imbalance": 0.0,
            "quote_imbalance": 0.0,
            "time_elapsed": 1.0,
            "trades": 100,
        })

    assert engine.pending_signal is None
    # Warmup isn't complete yet (bar_queue_size=10, only 9 bars fed so far),
    # so on_bar has returned before ever reaching the assembler - last_funding
    # is still its initial 0.0 at this point.
    assert engine.assembler.last_funding == 0.0

    # 2. 10th bar triggers the prediction and gate - the first bar that
    # actually reaches self.assembler.update(), so the stored funding rate
    # set above should now flow through instead of the old always-0.0 bug.
    res = engine.on_bar({
        "close": 50000.0,
        "close_time": 1010,
        "volume": 1.0,
        "buy_volume": 0.5,
        "sell_volume": 0.5,
        "high": 50100.0,
        "low": 49900.0,
        "open": 50000.0,
        "tick_imbalance": 0.0,
        "quote_imbalance": 0.0,
        "time_elapsed": 1.0,
        "trades": 100,
    })

    assert res is None
    assert engine.pending_signal is not None
    assert engine.pending_signal["direction"] == 1
    assert engine.assembler.last_funding == 0.00075

    # 3. Next bar triggers patience / entry
    res2 = engine.on_bar({
        "close": 50000.0,
        "close_time": 1011,
        "volume": 1.0,
        "buy_volume": 0.5,
        "sell_volume": 0.5,
        "high": 50100.0,
        "low": 49900.0,
        "open": 50000.0,
        "tick_imbalance": 0.0,
        "quote_imbalance": 0.0,
        "time_elapsed": 1.0,
        "trades": 100,
    })

    assert res2 is not None
    assert res2["action"] == "ENTRY"
    assert engine.active_position == True
    assert engine.pending_signal is None
