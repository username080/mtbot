"""
Purpose: Batch extracts and converts raw historical tick data from Binance ZIP files into Parquet format for faster processing.
"""
import os
import zipfile
import polars as pl
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_convert():
    input_dir = "/home/sysop/Lantransfer_cache/ETC/raw_zips"
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist.")
        return

    # Automatically derive the output directory. 
    # e.g. if input is /home/sysop/Lantransfer_cache/ETC/raw_zips 
    # output will be /home/sysop/Lantransfer_cache/ETC/parquets
    output_dir = input_path.parent / "parquets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard Binance aggTrades columns
    standard_columns_7 = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker']
    standard_columns_8 = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']
    
    zip_files = list(input_path.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No .zip files found in {input_path}")
        return
        
    for zip_path in sorted(zip_files):
        logger.info(f"Processing: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_filenames = [f for f in zf.namelist() if f.endswith('.csv')]
                
                if not csv_filenames:
                    logger.warning(f"No CSV found inside {zip_path.name}. Skipping.")
                    continue
                    
                csv_filename = csv_filenames[0]
                zf.extract(csv_filename, path=input_path)
                extracted_csv_path = input_path / csv_filename
                
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {zip_path}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            continue
            
        # Convert CSV to Parquet
        try:
            # Read CSV without headers
            df = pl.read_csv(extracted_csv_path, has_header=False)
            
            # Check column count to assign headers
            num_cols = len(df.columns)
            if num_cols == 7:
                df = df.rename(dict(zip(df.columns, standard_columns_7)))
            elif num_cols == 8:
                df = df.rename(dict(zip(df.columns, standard_columns_8)))
            else:
                logger.warning(f"Unexpected column count ({num_cols}) in {csv_filename}. Assigning generic headers.")
                df = df.rename(dict(zip(df.columns, [f"col_{i}" for i in range(num_cols)])))
            
            # Use the same filename but change extension to .parquet
            parquet_filename = zip_path.stem + ".parquet"
            parquet_path = output_dir / parquet_filename
            
            # Save to Parquet
            df.write_parquet(parquet_path)
            logger.info(f"Successfully converted {zip_path.name} -> {parquet_filename}")
            
        except Exception as e:
            logger.error(f"Error converting {extracted_csv_path} to parquet: {e}")
        finally:
            # Cleanup the extracted CSV aggressively
            if extracted_csv_path.exists():
                os.remove(extracted_csv_path)

if __name__ == "__main__":
    batch_convert()
