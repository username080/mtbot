import sys, os, json, logging
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logging.getLogger("backtest.dollar_bar_engine").setLevel(logging.WARNING)
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

print("Running quiet backtest for deep underperformance analysis...")
engine = DollarBarBacktestEngine(symbol, data_path, funding_path, config.get("algorithm", {}), oos_start_date="2024-01-01", oos_end_date="2025-01-01")
engine.run()

trades = engine.portfolio.trade_log
print(f"\n{'='*70}")
print(f"  DEEP UNDERPERFORMANCE & LOSS ANALYSIS — {len(trades)} TRADES")
print(f"{'='*70}")

# 1. Analyze the 61 Stop Loss Trades
sl_trades = [t for t in trades if "Stop Loss" in t.reason]
print(f"\n1. ANATOMY OF THE {len(sl_trades)} STOP LOSS TRADES (-${abs(sum(t.pnl for t in sl_trades)):.2f}):")
sl_longs = [t for t in sl_trades if not t.is_short]
sl_shorts = [t for t in sl_trades if t.is_short]
print(f"   - Long SLs: {len(sl_longs)} (-${abs(sum(t.pnl for t in sl_longs)):.2f})")
print(f"   - Short SLs: {len(sl_shorts)} (-${abs(sum(t.pnl for t in sl_shorts)):.2f})")

# Breakdown by month of SLs
sl_months = {}
for t in sl_trades:
    m = t.exit_time.strftime("%Y-%m")
    sl_months[m] = sl_months.get(m, 0.0) + t.pnl
print("   - Stop Loss bleeding by month:")
for m, p in sorted(sl_months.items(), key=lambda x: x[1]):
    print(f"     * {m}: ${p:>.2f} ({sum(1 for t in sl_trades if t.exit_time.strftime('%Y-%m')==m)} trades)")

# 2. Analyze Time Limit Trades
time_trades = [t for t in trades if "Time" in t.reason]
time_wins = [t for t in time_trades if t.pnl > 0]
time_losses = [t for t in time_trades if t.pnl <= 0]
print(f"\n2. ANATOMY OF THE {len(time_trades)} TIME LIMIT TRADES (Bar 50 Exit):")
print(f"   - Winners at Bar 50: {len(time_wins)} (+${sum(t.pnl for t in time_wins):.2f}) | Avg gain: +{np.mean([t.pnl_pct for t in time_wins]):.4f}%")
print(f"   - Losers at Bar 50:  {len(time_losses)} (-${abs(sum(t.pnl for t in time_losses)):.2f}) | Avg loss: {np.mean([t.pnl_pct for t in time_losses]):.4f}%")
print(f"   - Total Net from Time Limit: ${sum(t.pnl for t in time_trades):.2f}")

# 3. Why did we miss the 2024 Bull Run? (Compare Buy & Hold vs Active Bot)
first_price = trades[0].entry_price if trades else 0
last_price = trades[-1].exit_price if trades else 0
bh_return = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0
print(f"\n3. BUY & HOLD vs ACTIVE QUANT BOT:")
print(f"   - BTC Price Start (Jan 2024): ${first_price:.2f}")
print(f"   - BTC Price End (Dec 2024):   ${last_price:.2f}")
print(f"   - BTC Buy & Hold Return:      +{bh_return:.2f}% (+$10,000 becomes ${10000*(1+bh_return/100):.2f})")
bot_pnl = sum(t.pnl for t in trades)
bot_ret = (bot_pnl / 10000.0) * 100
print(f"   - Bot Net Return:             +{bot_ret:.2f}% (+${bot_pnl:,.2f})")
print(f"   - Why the gap? Let's check our time in market:")

total_hours = (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 3600 if trades and trades[-1].exit_time else (2 * 365 * 24)
hours_in_position = sum((t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades if t.exit_time and t.entry_time)
print(f"     * Total dataset duration: {total_hours:,.1f} hours ({total_hours/24:,.1f} days)")
print(f"     * Time held in position:  {hours_in_position:,.1f} hours ({hours_in_position/total_hours*100:.1f}% time in market)")

# 4. What if we eliminated Shorts or filtered by Trade Score?
print(f"\n4. HYPOTHETICAL IMPROVEMENT SIMULATIONS:")
long_only_pnl = sum(t.pnl for t in trades if not t.is_short)
print(f"   - Simulation A (Long-Only): Net PnL = ${long_only_pnl:+.2f} ({(long_only_pnl/10000)*100:.2f}%)")

# Let's check score distribution
wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl <= 0]
print(f"   - Avg Trade Score (Meta-Probability) on Winners: {np.mean([t.score for t in wins]):.4f}")
print(f"   - Avg Trade Score (Meta-Probability) on Losers:  {np.mean([t.score for t in losses]):.4f}")
