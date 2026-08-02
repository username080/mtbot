"""
Purpose: Processes raw historical Binance AggTrades (ticks) and aggregates them into Dollar Bars (Macro Bars) based on a dynamic daily volume threshold.
"""
import os
import logging
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.data_pipeline.dollar_bar_builder import DollarBarBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtest/data/tick_db"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtest/data"))
os.makedirs(OUT_DIR, exist_ok=True)

def build_bars_for_symbol(symbol):
    logger.info(f"Building historical dollar bars for {symbol}")
    
    years = sorted(os.listdir(os.path.join(DB_DIR, symbol))) if os.path.exists(os.path.join(DB_DIR, symbol)) else []
    
    if not years:
        logger.warning(f"No tick data found for {symbol} in {DB_DIR}")
        return
        
    builder = DollarBarBuilder(symbol)
    all_bars = []
    
    for year in years:
        year_path = os.path.join(DB_DIR, symbol, year)
        months = sorted(os.listdir(year_path))
        for month_file in months:
            file_path = os.path.join(year_path, month_file)
            logger.info(f"Processing {file_path}")
            
            try:
                df = pd.read_parquet(file_path)
                
                if len(df) > 0:
                    total_vol_quote = (df['price'] * df['quantity']).sum()
                    days_in_month = 30
                    daily_vol = total_vol_quote / days_in_month
                    builder.update_threshold(daily_vol)
                
                for row in df.itertuples():
                    agg_trade = {
                        "p": row.price,
                        "q": row.quantity,
                        "m": row.is_buyer_maker,
                        "T": row.timestamp
                    }
                    bar = builder.feed(agg_trade)
                    if bar:
                        bar["quote_imbalance"] = 0.0
                        all_bars.append(bar)
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
    if all_bars:
        out_df = pd.DataFrame(all_bars)
        out_path = os.path.join(OUT_DIR, f"dollar_bars_{symbol}_all.parquet")
        out_df.to_parquet(out_path)
        logger.info(f"Saved {len(out_df)} bars to {out_path}")

if __name__ == "__main__":
    for sym in SYMBOLS:
        build_bars_for_symbol(sym)
