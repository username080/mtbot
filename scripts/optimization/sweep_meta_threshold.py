import sys
import os
import json
import logging
import numpy as np

# Mute noisy engine logs for fast sweep
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logging.getLogger("backtest.dollar_bar_engine").setLevel(logging.WARNING)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from backtest.dollar_bar_engine import DollarBarBacktestEngine

def run_meta_threshold_sweep():
    print("======================================================================")
    print("        META-MODEL THRESHOLD OPTIMIZATION SWEEP")
    print("======================================================================")

    config_path = os.path.join(ROOT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    symbol = "BTCUSDT"
    data_path = os.path.join(ROOT_DIR, f"backtest/data/training_{symbol}.parquet")
    funding_path = os.path.join(ROOT_DIR, f"backtest/data/funding_rate_{symbol}.parquet")
    
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    thresholds_to_test = [round(x, 2) for x in np.arange(0.40, 0.61, 0.02)]
    results = []
    
    print(f"\nTesting {len(thresholds_to_test)} threshold values across 2024 Out-Of-Sample...\n")
    print(f"{'Thresh':<8} {'Trades':<8} {'Win Rate':<10} {'Net Return':<12} {'Net PnL ($)':<14} {'Max DD ($)':<12}")
    print("-" * 68)
    
    for thresh in thresholds_to_test:
        # Clone algorithm config and apply threshold
        algo_config = json.loads(json.dumps(config.get("algorithm", {})))
        if "meta_model" not in algo_config:
            algo_config["meta_model"] = {}
        algo_config["meta_model"]["meta_threshold"] = thresh
        
        # Instantiate and run engine
        engine = DollarBarBacktestEngine(symbol, data_path, funding_path, algo_config, oos_start_date="2024-01-01", oos_end_date="2025-01-01")
        summary = engine.run()
        
        trades = engine.portfolio.trade_log
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = summary.get("net_profit", 0.0)
        net_ret = summary.get("total_return_pct", 0.0)
        max_dd = summary.get("max_drawdown", 0.0)
        
        results.append({
            "threshold": thresh,
            "trades": total_trades,
            "win_rate": win_rate,
            "net_return": net_ret,
            "net_pnl": net_pnl,
            "max_dd": max_dd
        })
        
        print(f"{thresh:<8.2f} {total_trades:<8} {win_rate:>6.2f}%    {net_ret:>8.2f}%    ${net_pnl:>11.2f}   ${max_dd:>9.2f}")

    print("-" * 68)
    
    # Identify optimal threshold by Net Return
    best = max(results, key=lambda x: x["net_return"])
    print(f"\n🏆 OPTIMAL THRESHOLD: {best['threshold']:.2f}")
    print(f"   - Generates {best['trades']} trades with {best['win_rate']:.2f}% Win Rate")
    print(f"   - Net Return: +{best['net_return']:.2f}% (+${best['net_pnl']:.2f})")
    print("======================================================================")

if __name__ == "__main__":
    run_meta_threshold_sweep()
