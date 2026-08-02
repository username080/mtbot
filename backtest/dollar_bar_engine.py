import logging
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Optional
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.analyser.algorithm.signal_engine import SignalEngine
from backtest.portfolio import BacktestPortfolio

logger = logging.getLogger(__name__)

class DollarBarBacktestEngine:
    def __init__(self, symbol: str, data_path: str, funding_path: str, config: dict,
                  oos_start_date: str = "2025-01-01", oos_end_date: str = None):
        self.symbol = symbol
        self.config = config
        self.oos_start_timestamp = pd.to_datetime(oos_start_date) if oos_start_date else None
        
        logger.info(f"Loading backtest data from {data_path}...")
        self.data = pd.read_parquet(data_path)
        self.data = self.data.sort_values("close_time").reset_index(drop=True)
        
        # SLICE DATASET: Exactly 1000 bars before Jan 1, 2024 for instant 0.1s warm-up!
        if self.oos_start_timestamp:
            oos_ms = self.oos_start_timestamp.value // 10**6
            idx_matches = self.data[self.data["close_time"] >= oos_ms].index
            if len(idx_matches) > 0:
                idx_2024 = idx_matches[0]
                start_idx = max(0, idx_2024 - 1000)
                self.data = self.data.iloc[start_idx:].reset_index(drop=True)
                logger.info(f"Sliced dataset to {len(self.data)} bars starting exactly 1000 bars before Jan 1, 2024!")
                
        if oos_end_date:
            oos_end_ms = pd.to_datetime(oos_end_date).value // 10**6
            self.data = self.data[self.data["close_time"] < oos_end_ms].reset_index(drop=True)
            logger.info(f"Sliced dataset ending before {oos_end_date}!")
            
        # MERGE 4H FEATURES TO AVOID WARM-UP DISCREPANCY IN BACKTEST
        sym_base = self.symbol.replace("USDT", "")
        path_4h = os.path.join(os.path.dirname(data_path), sym_base, f"{self.symbol}_4h_klines.parquet")
        if os.path.exists(path_4h):
            df_4h = pd.read_parquet(path_4h)
            df_4h['open_time'] = df_4h['open_time'].astype(np.int64)
            df_4h['close'] = df_4h['close'].astype(float)
            df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
            
            df_4h['ema_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
            df_4h['regime'] = np.sign(df_4h['close'] - df_4h['ema_4h'])
            df_4h['regime_duration'] = df_4h.groupby((df_4h['regime'] != df_4h['regime'].shift()).cumsum()).cumcount() + 1
            
            df_4h['join_time'] = df_4h['open_time'] + (4 * 3600 * 1000)
            
            self.data = pd.merge_asof(
                self.data, 
                df_4h[['join_time', 'ema_4h', 'regime', 'regime_duration']], 
                left_on='close_time', 
                right_on='join_time', 
                direction='backward'
            )
            self.data = self.data.dropna(subset=['ema_4h']).reset_index(drop=True)
            logger.info("Successfully joined fully-warmed 4H features into backtest data.")
        
        # VECTORIZED VOLATILITY RATIO (Choppy Regime Detection)
        # Replaces the old ADX/ATR approach — volatility_ratio is now in PRIMARY_FEATURES
        if "close" in self.data.columns:
            ret_series = self.data["close"].pct_change()
            vol_100 = ret_series.ewm(span=100, adjust=False).std().fillna(0.005)
            vol_10 = ret_series.ewm(span=10, adjust=False).std().fillna(0.005)
            self.data["volatility_ratio"] = (vol_10 / vol_100.replace(0, 0.005)).fillna(1.0)
        
        self.funding_df = None
        if os.path.exists(funding_path):
            self.funding_df = pd.read_parquet(funding_path)
            self.funding_df = self.funding_df.sort_values("fundingTime")
            
        self.portfolio = BacktestPortfolio(initial_cash=10000.0, commission_rate=config.get("commission_rate", 0.0004))
        self.signal_engine = SignalEngine(symbol, config)
        
        # VECTORIZED BATCH GPU INFERENCE: Pre-predict all rows in one shot
        pm = self.signal_engine.primary_models
        if hasattr(pm, "model_long") and pm.model_long:
            long_feats = [f for f in pm.model_long.feature_names if f in self.data.columns]
            short_feats = [f for f in pm.model_short.feature_names if f in self.data.columns]
            if len(long_feats) == len(pm.model_long.feature_names) and len(short_feats) == len(pm.model_short.feature_names):
                logger.info("Vectorizing batch GPU XGBoost inference across all bars...")
                import xgboost as xgb
                dmatrix_long = xgb.DMatrix(self.data[long_feats])
                dmatrix_short = xgb.DMatrix(self.data[short_feats])
                self.data["precomputed_long_prob"] = pm.model_long.predict(dmatrix_long)
                self.data["precomputed_short_prob"] = pm.model_short.predict(dmatrix_short)
                logger.info("✓ Batch GPU inference complete! Running lightning simulation loop...")
            else:
                logger.warning("Missing features for vectorization! Falling back to slow live TA loop.")
        
    def run(self):
        logger.info(f"Starting backtest for {self.symbol} with {len(self.data)} bars...")
        
        current_funding_idx = 0
        funding_times = self.funding_df["fundingTime"].values if self.funding_df is not None else []
        funding_rates = self.funding_df["fundingRate"].values if self.funding_df is not None else []
        
        current_month = None
        
        for row in self.data.itertuples():
            bar = row._asdict()
            if "Index" in bar: del bar["Index"]
            
            close_time = int(bar["close_time"])
            # Normalize timestamp to milliseconds if it was stored in microseconds or nanoseconds
            while close_time > 3000000000000:
                close_time = close_time // 1000
            close = bar["close"]
            timestamp = pd.to_datetime(close_time, unit='ms')
            
            # Print whenever we enter a new month during the Out-of-Sample test!
            if self.oos_start_timestamp and timestamp >= self.oos_start_timestamp:
                if timestamp.month != current_month:
                    current_month = timestamp.month
                    months_list = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
                    m_str = f"{months_list[timestamp.month - 1]} {timestamp.year}"
                    logger.info(f"\n==========================================")
                    logger.info(f"  📅 ENTERING {m_str} 📅")
                    logger.info(f"==========================================\n")
            
            while current_funding_idx < len(funding_times) - 1 and funding_times[current_funding_idx + 1] <= close_time:
                current_funding_idx += 1
                
            if current_funding_idx < len(funding_rates):
                self.signal_engine.update_funding_rate(funding_rates[current_funding_idx])
                
            self.signal_engine.current_equity = self.portfolio.cash
            order = self.signal_engine.on_bar(bar)
            
            # If we are in the warm-up period (before 2024), silently warm up indicators but DO NOT trade!
            if self.oos_start_timestamp and timestamp < self.oos_start_timestamp:
                continue
                
            prices = {self.symbol: close}
            self.portfolio.snapshot_equity(timestamp, prices)
            
            if order:
                action = order.get("action")
                side = order.get("side")
                reason = order.get("reason", "")
                
                if action == "ENTRY":
                    qty = order.get("quantity", 0.0)
                    lev = order.get("leverage", 1.0)
                    is_short = (side == "SELL")
                    
                    self.portfolio.open_position(
                        symbol=self.symbol,
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
                        symbol=self.symbol,
                        price=close,
                        timestamp=timestamp,
                        reason=reason
                    )
                    
        if self.portfolio.has_position(self.symbol):
            self.portfolio.close_position(
                self.symbol,
                self.data.iloc[-1]["close"],
                pd.to_datetime(self.data.iloc[-1]["close_time"], unit='ms'),
                "End of Backtest"
            )
            
        logger.info("Backtest complete.")
        return self.portfolio.summary()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r") as f:
        config = json.load(f)
        
    symbol = "BTCUSDT"
    data_path = os.path.join(os.path.dirname(__file__), f"data/training_{symbol}.parquet")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), f"data/dollar_bars_{symbol}_all.parquet")
    funding_path = os.path.join(os.path.dirname(__file__), f"data/funding_rate_{symbol}.parquet")
    
    if os.path.exists(data_path):
        engine = DollarBarBacktestEngine(symbol, data_path, funding_path, config.get("algorithm", {}), oos_start_date="2024-01-01", oos_end_date=None)
        summary = engine.run()
        print("\n=== Backtest Summary (Out-of-Sample: 2024 Onwards) ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
    else:
        logger.warning(f"Data not found at {data_path}. Please run build scripts first.")
