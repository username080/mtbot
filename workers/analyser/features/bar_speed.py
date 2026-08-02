from collections import deque
import numpy as np

class BarSpeedDetector:
    """
    Detects unusually fast bar formation.
    """
    def __init__(self, window=20):
        self.window = window
        self.times = deque(maxlen=window)
        
    def update(self, bar):
        time_elapsed = bar.get("time_elapsed", 1.0)
        self.times.append(time_elapsed)
        
    @property
    def speed_z(self):
        if len(self.times) < 5:
            return 0.0
        current = self.times[-1]
        hist = np.array(self.times)
        mean = np.mean(hist)
        std = np.std(hist)
        
        if std < 1e-9:
            return 0.0
        return float((current - mean) / std)
        
    def is_fast(self, threshold=-1.5):
        # Negative z-score means less time elapsed -> faster bar
        return self.speed_z < threshold
