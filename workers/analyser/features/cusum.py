from collections import deque
import numpy as np

class CUSUMFilter:
    """
    CUSUM filter used for leverage scaling.
    """
    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.history = deque(maxlen=200)
        self.cusum = 0.0
        self.direction = 0
        self.threshold = 0.0
        
    def update(self, bar):
        tick_imb = bar.get("tick_imbalance", 0.0)
        
        self.history.append(abs(tick_imb))
        if len(self.history) >= 50:
            self.threshold = float(np.percentile(self.history, self.threshold_percentile))
        
        # Accumulate tick imbalance
        self.cusum = max(0.0, self.cusum + tick_imb) if tick_imb > 0 else min(0.0, self.cusum + tick_imb)
        
        if self.cusum > 0:
            self.direction = 1
        elif self.cusum < 0:
            self.direction = -1
        else:
            self.direction = 0
            
    @property
    def ratio(self):
        if self.threshold <= 1e-9:
            return 0.0
        r = abs(self.cusum) / self.threshold
        return float(min(r, 1.5))
