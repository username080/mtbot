"""
Fixed-Width Window Fractional Differencing (FFD) Module.
Based on Marcos Lopez de Prado's Advances in Financial Machine Learning, Chapter 5.

Achieves statistical stationarity (I(0)) for non-stationary price/volume series while preserving 
80-90% of long-term memory and correlation, overcoming the memory loss of standard d=1 integer differencing.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Union

class FastFFD:
    """
    Real-time streaming Fixed-Width Window Fractional Differencer for live execution and bar-by-bar assembling.
    """
    def __init__(self, d: float = 0.40, threshold: float = 1e-4):
        self.d = float(d)
        self.threshold = float(threshold)
        self.weights = self._compute_weights(self.d, self.threshold)
        self.rev_weights = self.weights[::-1]  # Reversed for chronological dot product [X_{t-L+1}, ..., X_t]
        self.window_len = len(self.weights)
        self.history = deque(maxlen=self.window_len)
        self._current_val = 0.0
        
    @staticmethod
    def _compute_weights(d: float, threshold: float) -> np.ndarray:
        """
        Precomputes binomial coefficient weights w_k = -w_{k-1} * (d - k + 1) / k until |w_k| < threshold.
        """
        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1
            if k > 2000:  # Safety bound
                break
        return np.array(weights, dtype=np.float64)
        
    def update(self, val: Union[float, int]):
        """
        Feed a new value (e.g. log close or log volume) into the streaming FFD filter.
        """
        if val is None or np.isnan(val) or np.isinf(val):
            val = 0.0
        val = float(val)
        self.history.append(val)
        
        n = len(self.history)
        if n < 5:
            self._current_val = 0.0
            return
            
        if n == self.window_len:
            # Full window: exact dot product
            self._current_val = float(np.dot(self.history, self.rev_weights))
        else:
            # Warmup: dot product with truncated weights
            w_slice = self.weights[:n][::-1]
            self._current_val = float(np.dot(self.history, w_slice))
            
    @property
    def value(self) -> float:
        """Returns the current stationary fractionally differenced value."""
        return self._current_val
        
    def reset(self):
        self.history.clear()
        self._current_val = 0.0


def compute_ffd_series(series: Union[pd.Series, np.ndarray], d: float = 0.40, threshold: float = 1e-4) -> np.ndarray:
    """
    Fast vectorized Fixed-Width Window Fractional Differencing for entire series (useful in data preparation).
    """
    if isinstance(series, pd.Series):
        vals = series.values.astype(np.float64)
    else:
        vals = np.asarray(series, dtype=np.float64)
        
    weights = FastFFD._compute_weights(d, threshold)
    window_len = len(weights)
    
    if len(vals) == 0:
        return np.array([], dtype=np.float64)
        
    # convolve with weights (which are ordered w0, w1, ... for X_t, X_{t-1}, ...)
    # using mode='full' and taking the first len(vals) outputs matches the causally filtered series
    out = np.convolve(vals, weights, mode='full')[:len(vals)]
    
    # Set warmup period to 0.0 to prevent artifact distortion before filter stabilizes
    warmup = min(window_len - 1, len(out))
    out[:warmup] = 0.0
    
    return out
