import logging
import os
import xgboost as xgb
from shared.features import PRIMARY_FEATURES

logger = logging.getLogger(__name__)

class PrimaryModelRunner:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_long = None
        self.model_short = None
        
        base_dir = os.path.dirname(__file__)
        path_long = os.path.abspath(os.path.join(base_dir, "../models/xgb_long_v3.json"))
        path_short = os.path.abspath(os.path.join(base_dir, "../models/xgb_short_v3.json"))
        
        if os.path.exists(path_long) and os.path.exists(path_short):
            self.model_long = xgb.Booster()
            self.model_long.load_model(path_long)
            self.model_long.set_param({"device": "cuda"})
            
            self.model_short = xgb.Booster()
            self.model_short.load_model(path_short)
            self.model_short.set_param({"device": "cuda"})
            logger.info(f"[{symbol}] Loaded Primary Dual XGBoost models v3 (on GPU)")
        else:
            logger.warning(f"[{symbol}] Primary models v3 not found. Signals will not trigger unless precomputed.")
            
    def is_loaded(self):
        return self.model_long is not None and self.model_short is not None
        
    def predict(self, features: dict) -> tuple[float, float]:
        if not self.is_loaded():
            return 0.0, 0.0
            
        row = [[features[c] for c in PRIMARY_FEATURES]]
        dmatrix = xgb.DMatrix(row, feature_names=PRIMARY_FEATURES)
        
        prob_long = float(self.model_long.predict(dmatrix)[0])
        prob_short = float(self.model_short.predict(dmatrix)[0])
        return prob_long, prob_short
