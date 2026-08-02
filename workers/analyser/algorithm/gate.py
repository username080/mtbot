import logging

logger = logging.getLogger(__name__)

from shared.config_helper import fetch

class Gate:
    """
    Static rules with dynamic coefficients to filter raw signals.
    """
    def __init__(self, config):
        gate_config = fetch(config, "gate")
            
        self.cooldown_bars = fetch(gate_config, "cooldown_bars_after_loss")
        self.min_atr_percentile = fetch(gate_config, "min_atr_percentile")
        self.min_confidence_gap = fetch(gate_config, "min_confidence_gap")
        
        # Default funding rate extremes (0.15% per 8hr)
        self.max_funding_long = fetch(gate_config, "max_funding_long")
        self.min_funding_short = fetch(gate_config, "min_funding_short")
        
        # Spread limit (e.g. 10 bps)
        self.max_spread = fetch(gate_config, "max_spread")
        
        # Flash crash threshold
        self.flash_crash_z_threshold = fetch(gate_config, "flash_crash_z_threshold")
        
        # Minimum TP Floor
        self.tp_vol_mult = fetch(config, "tp_vol_mult")
        self.min_tp_pct = fetch(gate_config, "min_tp_pct")

        
    def check(self, signal_type, prob, opposing_prob, symbol, active_positions, metrics, recent_pnls=None, bars_since_last_trade=999999) -> tuple[bool, str]:
        if symbol in active_positions:
            return False, "Already in position"
                
        # 1. Dual-Model Gap
        gap = prob - opposing_prob
        if gap < self.min_confidence_gap:
            return False, f"Confidence gap {gap:.2f} too low"
            
        # 2. Spread / Slippage Check (Will pass in backtest if spread is not provided)
        spread = metrics.get("spread", 0.0)
        if spread > self.max_spread:
            return False, f"Spread {spread:.5f} exceeds maximum allowed {self.max_spread:.5f}"
            
        # 3. Flash Crash / Extreme Volatility Filter
        bar_speed_z = metrics.get("bar_speed_z", 0.0)
        high_low_range_z = metrics.get("high_low_range_z", 0.0)
        if bar_speed_z > self.flash_crash_z_threshold or high_low_range_z > self.flash_crash_z_threshold:
            return False, f"Extreme volatility detected (bar_speed_z={bar_speed_z:.2f}, high_low_range_z={high_low_range_z:.2f})"
            
        # 4. Cooldown Logic (Consecutive Losses)
        if recent_pnls and len(recent_pnls) >= 3:
            # If the last 3 trades were losses (PnL <= 0)
            if all(pnl <= 0 for pnl in recent_pnls[-3:]):
                if bars_since_last_trade < self.cooldown_bars:
                    return False, f"Cooldown active: {int(bars_since_last_trade)} bars passed, need {self.cooldown_bars}"
            
        # 5. Funding Rate
        funding = metrics["fundingRate"]
        if signal_type == "LONG" and funding > self.max_funding_long:
            return False, f"Funding rate {funding:.5f} too high for long"
        if signal_type == "SHORT" and funding < self.min_funding_short:
            return False, f"Funding rate {funding:.5f} too low for short"
            
        # 6. Minimum Take Profit Floor
        vol = metrics["volatility"]
        if vol is None or vol <= 0:
            vol = 0.005
            
        expected_tp = vol * self.tp_vol_mult
        if expected_tp < self.min_tp_pct:
            return False, f"Expected TP ({expected_tp*100:.2f}%) is below floor ({self.min_tp_pct*100:.2f}%)"

        return True, "Pass"
