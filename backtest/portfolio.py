"""
Backtest Portfolio — State management for the backtest engine.

Tracks cash, open positions, equity curve, and trade log.
Designed to be a drop-in replacement for the live state_manager
interface, so the blackbox sees no difference between backtesting
and live trading.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from collections import deque
import numpy as np
import pandas as pd
import random

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Trade record
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """Single round-trip trade record."""
    symbol:      str
    entry_time:  pd.Timestamp
    entry_price: float
    qty:         float
    leverage:    float
    sl_price:    float                  = 0.0
    tp_price:    float                  = 0.0
    is_short:    bool                   = False
    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    pnl:         Optional[float]        = None
    pnl_pct:     Optional[float]        = None
    reason:      str                    = ""
    regime:      str                    = ""
    score:       float                  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio
# ─────────────────────────────────────────────────────────────────────────────
class BacktestPortfolio:
    """
    Portfolio state manager for the backtest engine.

    Manages a single position per symbol (no pyramiding).
    Records all trades and the equity curve at each bar.

    Parameters
    ----------
    initial_cash : float   starting capital in quote currency (e.g. USDT)
    """

    def __init__(self, initial_cash: float = 100_000.0, commission_rate: float = 0.0004):
        self.initial_cash = initial_cash
        self.cash: float  = initial_cash
        self.commission_rate: float = commission_rate

        # symbol → open Trade
        self._open_positions: Dict[str, Trade] = {}

        # Completed trade log
        self.trade_log: List[Trade] = []

        # Equity snapshot at each bar  [(timestamp, equity)]
        self.equity_curve: List[tuple] = []

        # Dynamic win rate queue (Burn-in 50/50 prior over 50 trades, interleaved)
        self.win_queue = deque([x for _ in range(25) for x in (True, False)], maxlen=50)

    # ── Equity ────────────────────────────────────────────────────────
    def current_equity(self, prices: Dict[str, float]) -> float:
        """Total portfolio value = cash + mark-to-market open positions."""
        equity = self.cash
        for sym, trade in self._open_positions.items():
            mkt_price = prices.get(sym, trade.entry_price)
            if trade.is_short:
                pnl = (trade.entry_price - mkt_price) * trade.qty * trade.leverage
            else:
                pnl = (mkt_price - trade.entry_price) * trade.qty * trade.leverage
            # We return the initial margin/cost to equity, plus the unrealised PnL
            equity += (trade.qty * trade.entry_price) + pnl
        return equity

    def snapshot_equity(self, timestamp: pd.Timestamp,
                        prices: Dict[str, float]):
        """Record equity at the current bar."""
        eq = self.current_equity(prices)
        self.equity_curve.append((timestamp, eq))

    # ── Position management ───────────────────────────────────────────
    def open_position(
        self,
        symbol:      str,
        qty:         float,
        price:       float,
        leverage:    float,
        timestamp:   pd.Timestamp,
        sl_price:    float = 0.0,
        tp_price:    float = 0.0,
        is_short:    bool = False,
        regime:      str  = "",
        score:       float = 0.0,
    ) -> bool:
        """
        Open a new long position.

        Returns False if already in a position for this symbol.
        """
        if symbol in self._open_positions:
            logger.debug(f"[Portfolio] Already in position for {symbol}.")
            return False

        # Apply random slippage (2 to 7 bps)
        slippage = random.uniform(0.0002, 0.0007)
        if is_short:
            price = price * (1 - slippage)  # Worse fill for short (sell lower)
        else:
            price = price * (1 + slippage)  # Worse fill for long (buy higher)

        cost = qty * price            # no leverage deduction here; PnL uses it
        entry_commission = qty * price * leverage * self.commission_rate
        
        if cost + entry_commission > self.cash:
            logger.warning(
                f"[Portfolio] Insufficient cash ({self.cash:.2f}) "
                f"to open {qty} × {price:.2f} for {symbol} (with comm)."
            )
            return False

        self.cash -= (cost + entry_commission)
        trade = Trade(
            symbol=symbol, entry_time=timestamp, entry_price=price,
            qty=qty, leverage=leverage, sl_price=sl_price, tp_price=tp_price,
            is_short=is_short, regime=regime, score=score,
        )
        self._open_positions[symbol] = trade

        logger.info(
            f"[Portfolio] OPEN {symbol}: qty={qty}, price={price}, "
            f"lev={leverage}x, SL={sl_price}, TP={tp_price}"
        )
        return True

    def close_position(
        self,
        symbol:    str,
        price:     float,
        timestamp: pd.Timestamp,
        reason:    str = "",
    ) -> Optional[Trade]:
        """
        Close an open position and realise PnL.

        Returns the completed Trade record, or None if no position exists.
        """
        trade = self._open_positions.pop(symbol, None)
        if trade is None:
            return None

        # Apply random slippage (2 to 7 bps)
        slippage = random.uniform(0.0002, 0.0007)
        if trade.is_short:
            price = price * (1 + slippage)  # Worse fill for closing short (buy higher)
        else:
            price = price * (1 - slippage)  # Worse fill for closing long (sell lower)

        proceeds = trade.qty * price
        exit_commission = trade.qty * price * trade.leverage * self.commission_rate
        
        if trade.is_short:
            pnl     = (trade.entry_price - price) * trade.qty * trade.leverage - exit_commission
            pnl_pct = ((trade.entry_price - price) / trade.entry_price * trade.leverage) - self.commission_rate
        else:
            pnl     = (price - trade.entry_price) * trade.qty * trade.leverage - exit_commission
            pnl_pct = ((price - trade.entry_price) / trade.entry_price * trade.leverage) - self.commission_rate

        self.cash += trade.qty * trade.entry_price + pnl   # return cost + PnL

        trade.exit_time  = timestamp
        trade.exit_price = price
        trade.pnl        = round(pnl, 4)
        trade.pnl_pct    = round(pnl_pct * 100, 4)
        trade.reason     = reason
        self.trade_log.append(trade)

        # Update dynamic win rate
        self.win_queue.append(pnl > 0)

        logger.info(
            f"[Portfolio] CLOSE {symbol}: price={price}, pnl={pnl:.2f}, "
            f"pnl_pct={pnl_pct*100:.2f}%, reason={reason}"
        )
        return trade

    def has_position(self, symbol: str) -> bool:
        return symbol in self._open_positions

    # ── Metrics summary ───────────────────────────────────────────────
    def summary(self) -> dict:
        """Compute aggregate performance metrics over all completed trades."""
        equity_vals = [e for _, e in self.equity_curve]
        final_eq = round(equity_vals[-1] if equity_vals else self.cash, 2)
        if not self.trade_log:
            return {
                "initial_cash":   self.initial_cash,
                "final_equity":   final_eq,
                "total_return_%": 0.0,
                "sharpe_ratio":   0.0,
                "max_drawdown_%": 0.0,
                "total_trades":   0,
                "win_rate_%":     0.0,
                "avg_pnl":        0.0,
                "avg_pnl_%":      0.0,
                "best_trade_%":   0.0,
                "worst_trade_%":  0.0,
                "total_wins":     0,
                "total_losses":   0,
            }

        pnls     = [t.pnl for t in self.trade_log]
        pnl_pcts = [t.pnl_pct for t in self.trade_log]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        
        tp_hits = sum(1 for t in self.trade_log if "Take Profit" in t.reason or "Trailing Stop" in t.reason)
        sl_hits = sum(1 for t in self.trade_log if "Stop Loss" in t.reason)
        time_stops = sum(1 for t in self.trade_log if "Time Limit" in t.reason or "Time Stop" in t.reason)
        opposite_hits = sum(1 for t in self.trade_log if "Opposite" in t.reason)
        
        total_t = len(self.trade_log)

        equity_vals = [e for _, e in self.equity_curve]
        if equity_vals:
            eq_series   = pd.Series(equity_vals)
            returns     = eq_series.pct_change().dropna()
            sharpe      = (returns.mean() / returns.std() * np.sqrt(8760)
                           if returns.std() > 0 else 0.0)
            peak        = eq_series.cummax()
            drawdown    = (eq_series - peak) / peak
            max_dd      = drawdown.min()
            total_ret   = (equity_vals[-1] - self.initial_cash) / self.initial_cash
        else:
            sharpe = max_dd = total_ret = 0.0

        return {
            "initial_cash":   self.initial_cash,
            "final_equity":   round(equity_vals[-1] if equity_vals else self.cash, 2),
            "total_return_%": round(total_ret * 100, 2),
            "sharpe_ratio":   round(sharpe, 4),
            "max_drawdown_%": round(max_dd * 100, 2),
            "total_trades":   total_t,
            "win_rate_%":     round(len(wins) / total_t * 100, 2),
            "precision_%":    round(len(wins) / total_t * 100, 2),
            "tp_hit_count":   tp_hits,
            "tp_hit_%":       round(tp_hits / total_t * 100, 2),
            "sl_hit_count":   sl_hits,
            "sl_hit_%":       round(sl_hits / total_t * 100, 2),
            "time_stop_count": time_stops,
            "time_stop_%":    round(time_stops / total_t * 100, 2),
            "opposite_signal_count": opposite_hits,
            "opposite_signal_%": round(opposite_hits / total_t * 100, 2),
            "avg_pnl":        round(np.mean(pnls), 4),
            "avg_pnl_%":      round(np.mean(pnl_pcts), 4),
            "best_trade_%":   round(max(pnl_pcts), 4),
            "worst_trade_%":  round(min(pnl_pcts), 4),
            "total_wins":     len(wins),
            "total_losses":   len(losses),
        }
