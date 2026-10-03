import logging
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.analyser.algorithm.signal_engine import SignalEngine
from backtest.portfolio import BacktestPortfolio

logger = logging.getLogger(__name__)

class MultiAssetBacktestEngine:
    def __init__(self, symbols: list, data_paths: list, config: dict, oos_start_date: str = "2024-01-01", oos_end_date: str = None):
        self.symbols = symbols
        self.config = config
        self.oos_start_timestamp = pd.to_datetime(oos_start_date) if oos_start_date else None
        
        self.engines = {}
        dataframes = []
        
        for symbol, data_path in zip(symbols, data_paths):
            logger.info(f"Loading backtest data for {symbol} from {data_path}...")
            df = pd.read_parquet(data_path)
            
            # Add symbol column so we know which bar this is after concatenation
            df["symbol"] = symbol
            
            # SLICE DATASET
            if self.oos_start_timestamp:
                oos_ms = self.oos_start_timestamp.value // 10**6
                idx_matches = df[df["close_time"] >= oos_ms].index
                if len(idx_matches) > 0:
                    idx_2024 = idx_matches[0]
                    start_idx = max(0, idx_2024 - 1000)
                    df = df.iloc[start_idx:].copy()
                    
            if oos_end_date:
                oos_end_ms = pd.to_datetime(oos_end_date).value // 10**6
                df = df[df["close_time"] < oos_end_ms].copy()
                
            # MERGE 4H FEATURES TO AVOID WARM-UP DISCREPANCY IN BACKTEST
            sym_base = symbol.replace("USDT", "")
            path_4h = os.path.join(os.path.dirname(data_path), sym_base, f"{symbol}_4h_klines.parquet")
            if os.path.exists(path_4h):
                df_4h = pd.read_parquet(path_4h)
                df_4h['open_time'] = df_4h['open_time'].astype(np.int64)
                df_4h['close'] = df_4h['close'].astype(float)
                df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
                
                df_4h['ema_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
                df_4h['regime'] = np.sign(df_4h['close'] - df_4h['ema_4h'])
                df_4h['regime_duration'] = df_4h.groupby((df_4h['regime'] != df_4h['regime'].shift()).cumsum()).cumcount() + 1
                
                df_4h['join_time'] = df_4h['open_time'] + (4 * 3600 * 1000)
                
                df = pd.merge_asof(
                    df, 
                    df_4h[['join_time', 'ema_4h', 'regime', 'regime_duration']], 
                    left_on='close_time', 
                    right_on='join_time', 
                    direction='backward'
                )
                df = df.dropna(subset=['ema_4h']).reset_index(drop=True)
                logger.info(f"Successfully joined fully-warmed 4H features into backtest data for {symbol}.")
            
            # Engine initialization
            engine = SignalEngine(symbol, config)
            self.engines[symbol] = engine
            
            # VECTORIZED BATCH GPU INFERENCE
            pm = engine.primary_models
            if hasattr(pm, "model_long") and pm.model_long:
                feats = [f for f in pm.model_long.feature_names if f in df.columns]
                if len(feats) == len(pm.model_long.feature_names):
                    logger.info(f"Vectorizing batch GPU inference for {symbol}...")
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(df[feats])
                    df["precomputed_long_prob"] = pm.model_long.predict(dmatrix)
                    df["precomputed_short_prob"] = pm.model_short.predict(dmatrix)
                else:
                    logger.warning(f"Missing features for {symbol}, falling back to slow loop.")
                    
            dataframes.append(df)
            
        logger.info("Concatenating and sorting multi-asset data by timestamp...")
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data = self.data.sort_values("close_time").reset_index(drop=True)
        
        self.portfolio = BacktestPortfolio(initial_cash=10000.0, commission_rate=config.get("commission_rate", 0.0004))
        self.current_prices = {}

    def run(self):
        logger.info(f"Starting multi-asset backtest with {len(self.data)} bars...")
        current_month = None
        
        for row in self.data.itertuples():
            bar = row._asdict()
            if "Index" in bar: del bar["Index"]
            
            symbol = bar["symbol"]
            close_time = int(bar["close_time"])
            while close_time > 3000000000000:
                close_time = close_time // 1000
            
            close = bar["close"]
            timestamp = pd.to_datetime(close_time, unit='ms')
            
            # Update running prices
            self.current_prices[symbol] = close
            
            if self.oos_start_timestamp and timestamp >= self.oos_start_timestamp:
                if timestamp.month != current_month:
                    current_month = timestamp.month
                    months_list = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
                    m_str = f"{months_list[timestamp.month - 1]} {timestamp.year}"
                    logger.info(f"\n==========================================")
                    logger.info(f"  📅 ENTERING {m_str} 📅")
                    logger.info(f"==========================================\n")
            
            engine = self.engines[symbol]
            engine.current_equity = self.portfolio.cash
            
            order = engine.on_bar(bar)
            
            if self.oos_start_timestamp and timestamp < self.oos_start_timestamp:
                continue
                
            self.portfolio.snapshot_equity(timestamp, self.current_prices)
            
            if order:
                action = order.get("action")
                side = order.get("side")
                reason = order.get("reason", "")
                
                if action == "ENTRY":
                    qty = order.get("quantity", 0.0)
                    lev = order.get("leverage", 1.0)
                    is_short = (side == "SELL")
                    
                    self.portfolio.open_position(
                        symbol=symbol,
                        qty=qty,
                        price=close,
                        leverage=lev,
                        timestamp=timestamp,
                        is_short=is_short,
                        score=order.get("score", 0.0),
                        regime=order.get("regime", "")
                    )
                elif action == "EXIT":
                    self.portfolio.close_position(
                        symbol=symbol,
                        price=close,
                        timestamp=timestamp,
                        reason=reason
                    )
                    
        for sym in self.symbols:
            if self.portfolio.has_position(sym):
                self.portfolio.close_position(
                    sym,
                    self.current_prices[sym],
                    pd.to_datetime(self.data.iloc[-1]["close_time"], unit='ms'),
                    "End of Backtest"
                )
            
        logger.info("Multi-asset Backtest complete.")
        return self.portfolio.summary()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r") as f:
        config = json.load(f)
        
    symbols = ["BTCUSDT", "ETHUSDT"]
    data_paths = [
        os.path.join(os.path.dirname(__file__), f"data/training_{sym}.parquet")
        for sym in symbols
    ]
    
    # Fallback to dollar_bars_... if training_... is missing
    for i, path in enumerate(data_paths):
        if not os.path.exists(path):
            data_paths[i] = os.path.join(os.path.dirname(__file__), f"data/dollar_bars_{symbols[i]}_all.parquet")
            
    if all(os.path.exists(p) for p in data_paths):
        engine = MultiAssetBacktestEngine(symbols, data_paths, config.get("algorithm", {}), oos_start_date="2024-01-01", oos_end_date=None)
        summary = engine.run()
        print("\n=== Multi-Asset Backtest Summary (Out-of-Sample: 2024 Onwards) ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
    else:
        logger.warning(f"Data not found for one or more symbols. Please run build scripts first.")
