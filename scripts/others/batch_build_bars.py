"""
Purpose: Runs the dollar bar builder across multiple historical datasets in parallel or batch mode to speed up dataset generation.
"""
import os
import logging
import pandas as pd
import pyarrow.parquet as pq
import gc
from pathlib import Path
import sys

# Add root directory to python path so we can import our worker modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
from workers.data_pipeline.dollar_bar_builder import DollarBarBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_bars_for_path(input_dir, symbol):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist. Skipping {symbol}.")
        return
        
    final_output_dir = Path(ROOT_DIR) / "backtest" / "data"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_out_file = final_output_dir / f"dollar_bars_{symbol}_all.parquet"
    
    # We create a progressive cache directory
    cache_dir = input_path.parent / "bars_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure they are sorted so we process time sequentially
    parquet_files = sorted(list(input_path.glob("*.parquet")))
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_path}")
        return
        
    logger.info(f"Found {len(parquet_files)} parquet files. Building dollar bars for {symbol}...")
    
    builder = DollarBarBuilder(symbol)
    
    for file_path in parquet_files:
        # Extract the specific date string e.g. "2019-01" from "BTCUSDT-aggTrades-2019-01.parquet"
        date_str = file_path.stem.split("aggTrades-")[-1]
        cache_file = cache_dir / f"dollar_bars_{symbol}_{date_str}.parquet"
        
        # Crash-Proofing: Skip if already compiled!
        if cache_file.exists():
            logger.info(f"Skipping {file_path.name}, already compiled: {cache_file.name}")
            continue
            
        logger.info(f"Processing: {file_path.name}")
        try:
            pf = pq.ParquetFile(file_path)
            
            # PASS 1: Calculate total volume efficiently in chunks
            total_vol_quote = 0.0
            for batch in pf.iter_batches(batch_size=5_000_000, columns=['price', 'quantity']):
                df_chunk = batch.to_pandas()
                if 'price' in df_chunk['price'].values:
                    df_chunk.drop(df_chunk.index[df_chunk['price'] == 'price'], inplace=True)
                total_vol_quote += (df_chunk['price'].astype(float) * df_chunk['quantity'].astype(float)).sum()
                del df_chunk
            
            daily_vol = total_vol_quote / 30.0
            builder.update_threshold(daily_vol)
            
            month_bars = []
            
            # PASS 2: Stream rows to the builder in chunks to save RAM
            for batch in pf.iter_batches(batch_size=5_000_000):
                df_chunk = batch.to_pandas()
                
                # Sanitize Chunk
                if 'price' in df_chunk['price'].values:
                    df_chunk.drop(df_chunk.index[df_chunk['price'] == 'price'], inplace=True)
                    
                df_chunk['price'] = df_chunk['price'].astype(float)
                df_chunk['quantity'] = df_chunk['quantity'].astype(float)
                df_chunk['timestamp'] = df_chunk['timestamp'].astype(int)
                if df_chunk['is_buyer_maker'].dtype != bool:
                    df_chunk['is_buyer_maker'] = df_chunk['is_buyer_maker'].astype(str).str.lower() == 'true'
                    
                for row in df_chunk.itertuples():
                    agg_trade = {
                        "p": row.price,
                        "q": row.quantity,
                        "m": row.is_buyer_maker,
                        "T": row.timestamp
                    }
                    bar = builder.feed(agg_trade)
                    if bar:
                        bar["quote_imbalance"] = 0.0 
                        month_bars.append(bar)
                        
                del df_chunk
                gc.collect()
                
            if month_bars:
                out_df = pd.DataFrame(month_bars)
                out_df.to_parquet(cache_file, index=False)
                logger.info(f"Saved {len(out_df)} bars for {date_str} to progressive cache.")
            else:
                logger.warning(f"No bars generated for {date_str}")
                    
        except Exception as e:
            logger.error(f"Failed processing {file_path.name}: {e}")
            
    # Final Compilation
    logger.info(f"Merging all cached months for {symbol} into the final dataset...")
    cached_files = sorted(list(cache_dir.glob("*.parquet")))
    if cached_files:
        dfs = [pd.read_parquet(f) for f in cached_files]
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_parquet(final_out_file, index=False)
        logger.info(f"SUCCESS: Compiled and saved {len(full_df)} total {symbol} Dollar Bars to {final_out_file}")
    else:
        logger.warning(f"No dollar bars generated for {symbol}!")

def main():
    logger.info("Starting Batch Dollar Bar Builder (Crash-Proof Edition)...")
    
    # Process BTC
    build_bars_for_path(os.path.join(ROOT_DIR, "backtest/data/BTC/raw_agg_trade"), "BTCUSDT")

    # Process ETH
    build_bars_for_path(os.path.join(ROOT_DIR, "backtest/data/ETH/raw_agg_trades"), "ETHUSDT")
    
    logger.info("All building processes complete!")

if __name__ == "__main__":
    main()
