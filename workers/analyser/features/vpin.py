from collections import deque
import numpy as np

class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.
    """
    def __init__(self, window=50):
        self.window = window
        self.imbalances = deque(maxlen=window)

    def update(self, bar):
        buy_vol = bar["buy_volume"]
        sell_vol = bar["sell_volume"]
        vol = bar["volume"]

        if vol < 1e-9:
            self.imbalances.append(0.0)
            return

        imb = abs(buy_vol - sell_vol) / vol
        self.imbalances.append(imb)

    @property
    def value(self):
        if len(self.imbalances) < self.window:
            return 0.0
        return float(np.mean(self.imbalances))
