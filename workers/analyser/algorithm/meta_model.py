import logging
import os
import xgboost as xgb
from shared.features import META_FEATURES, calculate_recent_pnl

logger = logging.getLogger(__name__)

from shared.config_helper import fetch

class MetaModelRunner:
    def __init__(self, symbol: str, config: dict):
        self.symbol = symbol
        self.config = config
        self.model_meta = None
        meta_config = fetch(config, "meta_model")
        self.enabled = fetch(meta_config, "enabled")
        
        base_dir = os.path.dirname(__file__)
        path_meta = os.path.abspath(os.path.join(base_dir, "../models/xgb_meta_v1.json"))
        
        if os.path.exists(path_meta):
            self.model_meta = xgb.Booster()
            self.model_meta.load_model(path_meta)
            self.model_meta.set_param({"device": "cuda"})
            logger.info(f"[{symbol}] Loaded Meta-Label XGBoost model v1 (on GPU)")
        else:
            self.enabled = False
            
    def evaluate(self, features: dict, long_prob: float, short_prob: float, direction: int, 
                 current_time: int, last_trade_time: int, recent_pnls: list) -> float:
        if not self.enabled or not self.model_meta:
            return long_prob if direction == 1 else short_prob
            
        hours_since = 100.0
        if last_trade_time > 0:
            hours_since = (current_time - last_trade_time) / (1000 * 3600.0)
            
        recent_pnl = calculate_recent_pnl(recent_pnls)
            
        meta_input = features.copy()
        meta_input.update({
            "long_prob": long_prob,
            "short_prob": short_prob,
            "is_long_signal": 1 if direction == 1 else 0,
            "confidence_gap": abs(long_prob - short_prob),
            "recent_signal_pnl_5": recent_pnl,
            "hours_since_last_trade": hours_since
        })
        
        row = [[meta_input[c] for c in META_FEATURES]]
        dmatrix = xgb.DMatrix(row, feature_names=META_FEATURES)
        return float(self.model_meta.predict(dmatrix)[0])
