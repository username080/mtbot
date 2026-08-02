import numpy as np
from typing import Optional, Dict, Any

class DollarBarBuilder:
    """
    Constructs dollar bars from a stream of aggTrades.
    Accumulates tick-level statistics (trade count, intensity, avg trade size).
    """
    def __init__(self, symbol: str, initial_threshold: float = 1_000_000.0):
        self.symbol = symbol
        self.threshold = initial_threshold
        self.reset()
        
    def reset(self):
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0
        self.trade_sizes = []
        self.dollar_volume_accumulated = 0.0
        self.open_time = 0
        self.close_time = 0
        
    def update_threshold(self, new_daily_volume: float):
        """
        Update the dollar volume threshold.
        Targeting ~1440 bars per day (1 per minute on average).
        """
        self.threshold = max(new_daily_volume / 1440.0, 100_000.0)  # Minimum threshold safety
        
    def feed(self, agg_trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Feed an aggTrade into the builder. 
        Returns a packet dictionary if the bar completes, else None.
        aggTrade format: {'s': symbol, 'p': price, 'q': qty, 'm': isBuyerMaker, 'T': timestamp}
        """
        price = float(agg_trade.get("p", 0.0))
        qty = float(agg_trade.get("q", 0.0))
        is_buyer_maker = agg_trade.get("m", False)
        ts = int(agg_trade.get("T", 0))
        
        dollar_val = price * qty
        
        # Initialize OHLC on first trade
        if self.trade_count == 0:
            self.open = price
            self.high = price
            self.low = price
            self.open_time = ts
            
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.close_time = ts
        
        self.volume += qty
        self.dollar_volume_accumulated += dollar_val
        self.trade_count += 1
        self.trade_sizes.append(dollar_val)
        
        # In Binance, isBuyerMaker=True means the maker was the buyer, 
        # so the taker was the seller (sell volume).
        if is_buyer_maker:
            self.sell_volume += qty
        else:
            self.buy_volume += qty
            
        # Check if threshold is reached
        if self.dollar_volume_accumulated >= self.threshold:
            packet = self._build_packet()
            self.reset()
            return packet
            
        return None
        
    def _build_packet(self) -> Dict[str, Any]:
        time_elapsed = max(self.close_time - self.open_time, 1)  # avoid div by zero, in ms
        avg_trade_size = float(np.mean(self.trade_sizes)) if self.trade_sizes else 0.0
        trade_intensity = self.trade_count / time_elapsed
        tick_imbalance = self.buy_volume - self.sell_volume
        
        packet = {
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "tick_imbalance": tick_imbalance,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "time_elapsed": time_elapsed,
            "trade_count": self.trade_count,
            "avg_trade_size": avg_trade_size,
            "trade_intensity": trade_intensity,
            # quote_imbalance to be attached by fetcher
        }
        return packet
