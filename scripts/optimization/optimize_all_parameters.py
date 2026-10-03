import sys, os, json, logging, itertools, argparse
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logging.getLogger("backtest.dollar_bar_engine").setLevel(logging.WARNING)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from backtest.dollar_bar_engine import DollarBarBacktestEngine

def run_master_parameter_optimization(num_samples=100):
    print("======================================================================")
    print("      MASTER PARAMETER OPTIMIZATION SWEEP — 2024 OUT-OF-SAMPLE")
    print("======================================================================")

    config_path = os.path.join(ROOT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    symbol = "BTCUSDT"
    data_path = os.path.join(ROOT_DIR, f"backtest/data/training_{symbol}.parquet")
    if not os.path.exists(data_path):
        data_path = os.path.join(ROOT_DIR, f"backtest/data/dollar_bars_{symbol}_all.parquet")
    funding_path = os.path.join(ROOT_DIR, f"backtest/data/funding_rate_{symbol}.parquet")
    
    # 1. Define Optimization Parameter Grid with user-specified ranges and step sizes!
    param_grid = {
        "forecast_horizon_bars": [30, 40, 50, 60, 70],
        "entry_prob_threshold": [round(x, 2) for x in np.arange(0.54, 0.61, 0.02)],
        "min_confidence_gap": [round(x, 2) for x in np.arange(0.08, 0.15, 0.02)],
        "meta_threshold": [round(x, 2) for x in np.arange(0.40, 0.61, 0.02)],
        "meta_model_enabled": [True, False],
        "choppy_adx_atr_ratio": [round(x, 2) for x in np.arange(0.25, 0.46, 0.05)],
        "choppy_prob_boost": [round(x, 2) for x in np.arange(0.02, 0.09, 0.02)],
        
        # New Gate Optimizations
        "gate_cooldown_bars": [1, 2, 3, 4, 5],
        "gate_min_atr_pct": [0.10, 0.20, 0.30],
        "gate_flash_crash_z": [3.0, 4.0, 5.0, 6.0]
    }
    
    # Create parameter combinations
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total parameter combinations in grid: {len(all_combinations):,}")
    
    np.random.seed(42)
    sample_size = min(num_samples, len(all_combinations))
    sample_indices = np.random.choice(len(all_combinations), size=sample_size, replace=False)
    sampled_combos = [all_combinations[i] for i in sample_indices]
    
    results = []
    print(f"\nTesting {len(sampled_combos)} parameter combinations on 2024 dataset...\n")
    print(f"{'ID':<4} {'Horizon':<8} {'Entry':<6} {'Gap':<6} {'Meta':<6} {'MetaOn':<7} {'CoolDn':<7} {'AtrPct':<7} {'FlashZ':<7} {'Trades':<8} {'WinRate':<8} {'Return %':<10} {'Max DD':<8}")
    print("-" * 115)
    
    for idx, p in enumerate(sampled_combos, 1):
        algo_config = json.loads(json.dumps(config.get("algorithm", {})))
        
        # Apply parameters
        algo_config["forecast_horizon_bars"] = p["forecast_horizon_bars"]
        algo_config["entry_prob_threshold"] = p["entry_prob_threshold"]
        
        if "gate" not in algo_config: algo_config["gate"] = {}
        algo_config["gate"]["min_confidence_gap"] = p["min_confidence_gap"]
        algo_config["gate"]["cooldown_bars_after_loss"] = p["gate_cooldown_bars"]
        algo_config["gate"]["min_atr_percentile"] = p["gate_min_atr_pct"]
        algo_config["gate"]["flash_crash_z_threshold"] = p["gate_flash_crash_z"]
        
        if "meta_model" not in algo_config: algo_config["meta_model"] = {}
        algo_config["meta_model"]["meta_threshold"] = p["meta_threshold"]
        algo_config["meta_model"]["enabled"] = p["meta_model_enabled"]
        
        if "regime_thresholds" not in algo_config: algo_config["regime_thresholds"] = {}
        algo_config["regime_thresholds"]["choppy_adx_atr_ratio"] = p["choppy_adx_atr_ratio"]
        algo_config["regime_thresholds"]["choppy_prob_boost"] = p["choppy_prob_boost"]
        
        # 2024 Only Backtest
        engine = DollarBarBacktestEngine(
            symbol, data_path, funding_path, algo_config, 
            oos_start_date="2024-01-01", oos_end_date="2025-01-01"
        )
        summary = engine.run()
        
        trades = engine.portfolio.trade_log
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        net_ret = summary.get("total_return_pct", 0.0)
        max_dd = summary.get("max_drawdown", 0.0)
        
        results.append({
            "id": idx,
            "params": p,
            "trades": total_trades,
            "win_rate": win_rate,
            "net_return": net_ret,
            "max_dd": max_dd
        })
        
        print(f"{idx:<4} {p['forecast_horizon_bars']:<8} {p['entry_prob_threshold']:<6.2f} {p['min_confidence_gap']:<6.2f} {p['meta_threshold']:<6.2f} {'YES' if p['meta_model_enabled'] else 'NO':<7} {p['gate_cooldown_bars']:<7} {p['gate_min_atr_pct']:<7.2f} {p['gate_flash_crash_z']:<7.1f} {total_trades:<8} {win_rate:>5.1f}%   {net_ret:>7.2f}%   ${max_dd:>6.0f}")

    print("-" * 115)
    best = max(results, key=lambda x: x["net_return"])
    print(f"\n🏆 OPTIMAL PARAMETER SET (Combo #{best['id']}):")
    for k, v in best['params'].items():
        print(f"   * {k}: {v}")
    print(f"\n   -> Performance: +{best['net_return']:.2f}% Return | {best['trades']} Trades | {best['win_rate']:.1f}% Win Rate | Max DD: ${best['max_dd']:.0f}")
    print("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Parameter Optimization Sweep")
    parser.add_argument("--samples", type=int, default=100, help="Number of random parameter combinations to test (default: 100)")
    args = parser.parse_args()
    run_master_parameter_optimization(num_samples=args.samples)
