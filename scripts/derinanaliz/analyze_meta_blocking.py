import sys, os, json, copy, logging
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logging.getLogger("backtest.dollar_bar_engine").setLevel(logging.WARNING) # Quiet mode for fast runs
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from backtest.dollar_bar_engine import DollarBarBacktestEngine

with open(os.path.join(ROOT_DIR, "config.json")) as f:
    config = json.load(f)

symbol = "BTCUSDT"
data_path = os.path.join(ROOT_DIR, "backtest/data/training_BTCUSDT.parquet")
if not os.path.exists(data_path):
    data_path = os.path.join(ROOT_DIR, "backtest/data/dollar_bars_BTCUSDT_all.parquet")
funding_path = os.path.join(ROOT_DIR, f"backtest/data/funding_rate_{symbol}.parquet")

print("="*70)
print("  META-LABELING BLOCKING & TRADE IMPACT ANALYSIS")
print("="*70)

# ─────────────────────────────────────────────────────────────
# Run 1: NO META-LABELING BLOCKING (meta_threshold = 0.0)
# ─────────────────────────────────────────────────────────────
print("\n[Run 1/2] Simulating strategy WITHOUT Meta-Labeling blocking (meta_threshold = 0.0)...")
algo_no_gate = copy.deepcopy(config.get("algorithm", {}))
if "meta_model" not in algo_no_gate:
    algo_no_gate["meta_model"] = {}
algo_no_gate["meta_model"]["meta_threshold"] = 0.0  # 0.0 means 0% blocked, all primary signals allowed!

engine_no_gate = DollarBarBacktestEngine(symbol, data_path, funding_path, algo_no_gate, oos_start_date="2024-01-01")
engine_no_gate.run()
trades_no_gate = engine_no_gate.portfolio.trade_log

# ─────────────────────────────────────────────────────────────
# Run 2: BASELINE WITH META-LABELING BLOCKING (meta_threshold = 0.55)
# ─────────────────────────────────────────────────────────────
print("\n[Run 2/2] Simulating strategy WITH Meta-Labeling blocking (meta_threshold = 0.55)...")
algo_with_gate = copy.deepcopy(config.get("algorithm", {}))
if "meta_model" not in algo_with_gate:
    algo_with_gate["meta_model"] = {}
algo_with_gate["meta_model"]["meta_threshold"] = 0.55

engine_with_gate = DollarBarBacktestEngine(symbol, data_path, funding_path, algo_with_gate, oos_start_date="2024-01-01")
engine_with_gate.run()
trades_with_gate = engine_with_gate.portfolio.trade_log

# ─────────────────────────────────────────────────────────────
# Comparative Impact Analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  META-LABELING GATE IMPACT REPORT (2024-2025 OOS)")
print("="*70)

pnl_no = sum(t.pnl for t in trades_no_gate)
pnl_with = sum(t.pnl for t in trades_with_gate)
wr_no = sum(1 for t in trades_no_gate if t.pnl > 0) / len(trades_no_gate) * 100 if trades_no_gate else 0
wr_with = sum(1 for t in trades_with_gate if t.pnl > 0) / len(trades_with_gate) * 100 if trades_with_gate else 0

blocked_count = len(trades_no_gate) - len(trades_with_gate)
blocked_pnl_diff = pnl_no - pnl_with

print(f"1. TRADE COUNT & FREQUENCY:")
print(f"   - Without Meta-Gate: {len(trades_no_gate):>5} trades")
print(f"   - With Meta-Gate:    {len(trades_with_gate):>5} trades")
print(f"   - TOTAL BLOCKED:     {blocked_count:>5} trades ({blocked_count/len(trades_no_gate)*100:.1f}% of all primary signals vetoed!)")

print(f"\n2. PROFITABILITY & WIN RATE:")
print(f"   - Without Meta-Gate PnL: ${pnl_no:>+10.2f} (Win Rate: {wr_no:.2f}%)")
print(f"   - With Meta-Gate PnL:    ${pnl_with:>+10.2f} (Win Rate: {wr_with:.2f}%)")
print(f"   - NET DOLLAR IMPACT:     ${blocked_pnl_diff:>+10.2f}")
if blocked_pnl_diff < 0:
    print(f"     ✅ META-LABELING SAVED US ${abs(blocked_pnl_diff):.2f}! (The blocked trades were net losers)")
else:
    print(f"     ⚠️ META-LABELING BLOCKED +${blocked_pnl_diff:.2f} IN PROFIT! (The gate blocked high return trades)")

# 3. Analyze what type of trades were blocked
# Let's match trades by entry time to see which specific trades were blocked
with_gate_times = set(t.entry_time for t in trades_with_gate)
blocked_trades = [t for t in trades_no_gate if t.entry_time not in with_gate_times]

if blocked_trades:
    blocked_longs = [t for t in blocked_trades if not t.is_short]
    blocked_shorts = [t for t in blocked_trades if t.is_short]
    blocked_wins = [t for t in blocked_trades if t.pnl > 0]
    blocked_losses = [t for t in blocked_trades if t.pnl <= 0]
    
    print(f"\n3. ANATOMY OF THE {len(blocked_trades)} BLOCKED TRADES:")
    print(f"   - Blocked Longs:  {len(blocked_longs):>4} (PnL: ${sum(t.pnl for t in blocked_longs):>+9.2f})")
    print(f"   - Blocked Shorts: {len(blocked_shorts):>4} (PnL: ${sum(t.pnl for t in blocked_shorts):>+9.2f})")
    print(f"   - Blocked Winners: {len(blocked_wins):>4} (Total Won:  +${sum(t.pnl for t in blocked_wins):.2f} | Avg: +{np.mean([t.pnl_pct for t in blocked_wins]):.3f}%)")
    print(f"   - Blocked Losers:  {len(blocked_losses):>4} (Total Lost: -${abs(sum(t.pnl for t in blocked_losses)):.2f} | Avg: {np.mean([t.pnl_pct for t in blocked_losses]):.3f}%)")
    
    # Did it block any huge TP hits?
    blocked_tps = [t for t in blocked_trades if "Take Profit" in t.reason or "Trailing" in t.reason]
    print(f"   - Blocked Take-Profit Jackpots: {len(blocked_tps)} trades (+${sum(t.pnl for t in blocked_tps):.2f})")
    
    # Did it block stop losses?
    blocked_sls = [t for t in blocked_trades if "Stop Loss" in t.reason]
    print(f"   - Blocked Stop-Loss Disasters:  {len(blocked_sls)} trades (-${abs(sum(t.pnl for t in blocked_sls)):.2f})")

