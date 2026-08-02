import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from workers.data_pipeline.dollar_bar_builder import DollarBarBuilder

def test_dollar_bar_builder():
    builder = DollarBarBuilder("BTCUSDT", initial_threshold=100_000.0)
    
    trade1 = {"p": 50000.0, "q": 1.0, "m": True, "T": 1000}
    trade2 = {"p": 50010.0, "q": 0.5, "m": False, "T": 1010}
    
    assert builder.feed(trade1) is None
    assert builder.feed(trade2) is None
    
    assert builder.dollar_volume_accumulated == 75005.0
    
    trade3 = {"p": 50005.0, "q": 1.0, "m": False, "T": 1020}
    packet = builder.feed(trade3)
    
    assert packet is not None
    assert packet["symbol"] == "BTCUSDT"
    assert packet["open"] == 50000.0
    assert packet["close"] == 50005.0
    assert packet["high"] == 50010.0
    assert packet["low"] == 50000.0
    assert packet["volume"] == 2.5
    assert packet["buy_volume"] == 1.5
    assert packet["sell_volume"] == 1.0
    assert packet["tick_imbalance"] == 0.5
    assert packet["time_elapsed"] == 20
    assert packet["trade_count"] == 3
