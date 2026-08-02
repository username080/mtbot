import numpy as np

class RegimeFeatures:
    """
    Long-term regime features computed from the 200-bar queue.
    """
    def update(self, bar_queue):
        if not bar_queue:
            return
            
        self.closes = np.array([b["close"] for b in bar_queue])
        self.volumes = np.array([b["volume"] for b in bar_queue])
        self.highs = np.array([b["high"] for b in bar_queue])
        self.lows = np.array([b["low"] for b in bar_queue])
        
    @property
    def price_z(self):
        if len(self.closes) < 5:
            return 0.0
        mean = np.mean(self.closes)
        std = np.std(self.closes)
        if std < 1e-9:
            return 0.0
        return float((self.closes[-1] - mean) / std)
        
    @property
    def vol_regime(self):
        if len(self.closes) < 5:
            return 1.0
        # TR calculation roughly
        tr = np.maximum(self.highs - self.lows, 0)
        mean_tr = np.mean(tr)
        if mean_tr < 1e-9:
            return 1.0
        return float(tr[-1] / mean_tr)
        
    @property
    def volume_z(self):
        if len(self.volumes) < 5:
            return 0.0
        mean = np.mean(self.volumes)
        std = np.std(self.volumes)
        if std < 1e-9:
            return 0.0
        return float((self.volumes[-1] - mean) / std)
