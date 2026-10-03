import pandas as pd
from collections import deque
from workers.analyser.algorithm.indicators import compute_live_ta

def test_compute_live_ta():
    # Create a dummy bar queue of 200 items (the default maxlen in SignalEngine)
    bar_queue = deque(maxlen=200)
    for i in range(200):
        bar_queue.append({
            "close": 50000.0 + (i * 10),
            "high": 50100.0 + (i * 10),
            "low": 49900.0 + (i * 10),
            "volume": 1.5,
            "close_time": 1700000000 + (i * 60000)
        })
    
    # Compute indicators
    features = compute_live_ta(bar_queue)
    
    # Verify outputs
    assert isinstance(features, dict)
    assert "ema_20" in features
    assert "atr_14" in features
    assert "bollinger_width" in features
    assert "rsi_14" in features
    assert "price_vs_ema200" in features
    
    # Since prices are steadily increasing by 10 per bar
    # The price should be higher than EMA200
    assert features["price_vs_ema200"] > 0
    assert not pd.isna(features["ema_20"])
