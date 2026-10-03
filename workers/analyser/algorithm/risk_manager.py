import logging

logger = logging.getLogger(__name__)

from shared.config_helper import fetch

class RiskManager:
    def __init__(self, symbol: str, config: dict):
        self.symbol = symbol
        self.config = config
        
    def calculate_position_size(self, cusum_ratio: float, prob: float, price: float, current_equity: float, volatility: float, score: float, regime: str, direction: int) -> dict:
        # 1. Leverage Scaling (CUSUM)
        base_lev = fetch(self.config, "base_leverage")
        lev = base_lev * (0.5 + 0.5 * min(cusum_ratio, 1.5))
        lev = min(lev, fetch(self.config, "max_leverage"))
        
        # 2. Kelly Criterion Sizing (Fractional Kelly)
        half_kelly_fraction = fetch(self.config, "half_kelly_fraction")
        raw_kelly = (2 * prob - 1) * half_kelly_fraction
        
        # Cap position fraction between 10% and 100% of cash
        kelly_fraction = max(0.1, min(raw_kelly, 1.0))
        
        target_notional = current_equity * kelly_fraction * lev
        qty = target_notional / price
        
        tp_mult = fetch(self.config, "tp_vol_mult")
        sl_mult = fetch(self.config, "sl_vol_mult")
        
        trade_tp_pct = max(0.0010, volatility * tp_mult)
        trade_sl_pct = max(0.0010, volatility * sl_mult)
        
        side = "BUY" if direction == 1 else "SELL"
        
        return {
            "action": "ENTRY",
            "symbol": self.symbol,
            "side": side,
            "quantity": qty,
            "leverage": lev,
            "reason": "Patience OK",
            "score": score,
            "regime": regime,
            "tp_pct": trade_tp_pct,
            "sl_pct": trade_sl_pct
        }
        
    def check_exit_conditions(self, ret: float, bars_held: int, position_direction: int, 
                              prob_long: float, prob_short: float, features: dict, 
                              trade_tp_pct: float, trade_sl_pct: float) -> dict:
        # Dynamic TP / Time limit widening
        max_bars = fetch(self.config, "forecast_horizon_bars")
        
        p_vs_ema200 = features.get("price_vs_ema200", 0.0)
        e_cross = features.get("ema_cross", 0.0)
        mom_20 = features.get("momentum_20", 0.0)
        
        # Trend-Following Boost
        if (position_direction == 1 and p_vs_ema200 > 0.005 and e_cross > 0 and mom_20 > 0) or \
           (position_direction == -1 and p_vs_ema200 < -0.005 and e_cross < 0 and mom_20 < 0):
            trade_tp_pct *= 1.8
            max_bars = int(max_bars * 1.6)
            
        if ret >= trade_tp_pct:
            return {"action": "EXIT", "symbol": self.symbol, "side": "SELL" if position_direction == 1 else "BUY", "reason": "Take Profit Hit"}
            
        current_sl = -trade_sl_pct
        if ret <= current_sl:
            return {"action": "EXIT", "symbol": self.symbol, "side": "SELL" if position_direction == 1 else "BUY", "reason": "Stop Loss Hit"}
            
        if bars_held >= max_bars:
            panic_thresh = fetch(self.config, "panic_exit_threshold")
            if (position_direction == 1 and prob_long < panic_thresh) or \
               (position_direction == -1 and prob_short < panic_thresh):
                return {"action": "EXIT", "symbol": self.symbol, "side": "SELL" if position_direction == 1 else "BUY", "reason": "Time Limit Hit"}
                
        # Opposite Signal Exit
        panic_thresh = fetch(self.config, "panic_exit_threshold")
        if position_direction == 1 and prob_short > panic_thresh:
            return {"action": "EXIT", "symbol": self.symbol, "side": "SELL" if position_direction == 1 else "BUY", "reason": "Opposite Signal"}
        if position_direction == -1 and prob_long > panic_thresh:
            return {"action": "EXIT", "symbol": self.symbol, "side": "SELL" if position_direction == 1 else "BUY", "reason": "Opposite Signal"}
            
        return None
