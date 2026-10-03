"""
Purpose: Downloads raw historical tick data (AggTrades) directly from the Binance public data archives for the specified symbols and date ranges.
"""
#!/usr/bin/env python3
import os
import sys
import time
import zipfile
import urllib.request
import logging

try:
    import polars as pl
except ImportError:
    print("Polars not found. Please run: pip install polars")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"
SYMBOLS = ["BTCUSDT"]  # User explicitly requested just BTC
YEARS = range(2019, 2024) # 5 years
MONTHS = range(1, 13)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backtest/data"))
ZIP_DIR = os.path.join(DATA_DIR, "raw_zips")
TEMP_DIR = os.path.join(DATA_DIR, "raw_csv_temp")
DB_DIR = os.path.join(DATA_DIR, "tick_db")                                                                                                                                                                                                                      

os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

def phase1_download_all():
    logger.info("=== PHASE 1: DOWNLOADING ALL ZIP FILES ===")
    for symbol in SYMBOLS:
        for year in YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                filename = f"{symbol}-aggTrades-{year}-{month_str}.zip"
                url = f"{BASE_URL}/{symbol}/{filename}"
                zip_path = os.path.join(ZIP_DIR, filename)
                
                if os.path.exists(zip_path):
                    logger.info(f"[{symbol}] {year}-{month_str} ZIP already exists. Skipping download.")
                    continue
                
                try:
                    logger.info(f"[{symbol}] Downloading {year}-{month_str} from Binance...")
                    urllib.request.urlretrieve(url, zip_path)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        logger.warning(f"[{symbol}] Data not found for {year}-{month_str} (404).")
                    else:
                        logger.error(f"HTTP error {e.code} for {url}.")
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}.")

def phase2_convert_to_binary():
    logger.info("=== PHASE 2: CONVERTING ZIPS TO POLARS BINARY ===")
    for symbol in SYMBOLS:
        for year in YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                filename = f"{symbol}-aggTrades-{year}-{month_str}.zip"
                csv_filename = filename.replace(".zip", ".csv")
                
                zip_path = os.path.join(ZIP_DIR, filename)
                csv_path = os.path.join(TEMP_DIR, csv_filename)
                
                out_dir = os.path.join(DB_DIR, symbol, str(year))
                os.makedirs(out_dir, exist_ok=True)
                parquet_path = os.path.join(out_dir, f"{month_str}.parquet")
                
                if os.path.exists(parquet_path):
                    logger.info(f"[{symbol}] {year}-{month_str} Parquet already exists. Skipping conversion.")
                    continue
                
                if not os.path.exists(zip_path):
                    logger.warning(f"[{symbol}] {year}-{month_str} ZIP missing. Skipping conversion.")
                    continue

                # 1. Extract CSV
                logger.info(f"[{symbol}] Extracting {filename}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(TEMP_DIR)
                except Exception as e:
                    logger.error(f"Failed to extract {filename}: {e}.")
                    continue

                # 2. Convert to Parquet with Polars
                logger.info(f"[{symbol}] Compressing {csv_filename} into Binary Parquet using Polars...")
                try:
                    t0 = time.time()
                    df = pl.read_csv(
                        csv_path,
                        has_header=False,
                        columns=[1, 2, 5, 6], # Only keep price, qty, timestamp, is_buyer_maker
                        new_columns=["price", "quantity", "timestamp", "is_buyer_maker"]
                    )
                    
                    # Cast data types safely after reading (pass as *args, not a list)
                    df = df.with_columns(
                        pl.col("price").cast(pl.Float32),
                        pl.col("quantity").cast(pl.Float32),
                        pl.col("timestamp").cast(pl.Int64),
                        pl.col("is_buyer_maker").cast(pl.Boolean)
                    )
                    
                    # Normalize timestamps to milliseconds if they are in microseconds/nanoseconds
                    if df.height > 0:
                        first_ts = df["timestamp"][0]
                        if first_ts > 1e17: # Nanoseconds
                            df = df.with_columns((pl.col("timestamp") // 1_000_000).cast(pl.Int64))
                        elif first_ts > 1e14: # Microseconds
                            df = df.with_columns((pl.col("timestamp") // 1_000).cast(pl.Int64))
                    
                    df.write_parquet(parquet_path, compression="zstd")
                    elapsed = time.time() - t0
                    
                    original_size = os.path.getsize(csv_path) / (1024 * 1024)
                    new_size = os.path.getsize(parquet_path) / (1024 * 1024)
                    logger.info(f"[{symbol}] Parquet Created! Time: {elapsed:.2f}s | Size: {original_size:.1f}MB -> {new_size:.1f}MB")
                    
                except Exception as e:
                    logger.error(f"Failed to process {csv_filename} with Polars: {e}")
                    if os.path.exists(parquet_path):
                        os.remove(parquet_path)

                # 3. Cleanup extracted CSV (we keep the ZIP to be safe, since we have 300GB)
                if os.path.exists(csv_path):
                    os.remove(csv_path)

def main():
    logger.info("Starting Separated Download & Binary Conversion Pipeline...")
    phase1_download_all()
    phase2_convert_to_binary()
    logger.info(f"Pipeline Complete! Check {DB_DIR} for the final binary database.")

if __name__ == "__main__":
    main()
