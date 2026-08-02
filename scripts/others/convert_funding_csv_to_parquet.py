"""
Universal Historical Funding Rate CSV to Parquet Converter.
Converts any coin's historical CSV funding rate data (BTC, ETH, SOL, etc.) to .parquet format 
and places it in the backtest data directory where build_training_features.py expects it.

Features:
    - Auto-detects symbols from the CSV 'symbol' column or filename.
    - Supports single file conversion or batch scanning of all CSVs in the folder.
    - Guarantees int64 timestamp (Unix milliseconds) normalization for seamless pd.merge_asof joins.

Usage:
    # Convert all CSV files in scripts/others automatically:
    python3 convert_funding_csv_to_parquet.py --all

    # Convert a specific file:
    python3 convert_funding_csv_to_parquet.py --csv eth_funding_rates_2019_2026.csv
"""

import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../backtest/data"))

def process_symbol_data(df: pd.DataFrame, symbol: str):
    """
    Cleans, normalizes, and saves funding rate DataFrame for a specific coin symbol.
    """
    symbol_upper = symbol.upper().strip()
    if not symbol_upper.endswith("USDT") and not symbol_upper.endswith("USD"):
        symbol_upper = f"{symbol_upper}USDT"

    logger.info(f"--- Processing [{symbol_upper}] ({len(df)} raw rows) ---")

    # Ensure timestamp is int64 Unix milliseconds for Dollar Bar compatibility
    if pd.api.types.is_string_dtype(df["fundingTime"]) or pd.api.types.is_object_dtype(df["fundingTime"]):
        dt_series = pd.to_datetime(df["fundingTime"], format="mixed", utc=True)
    elif pd.api.types.is_datetime64_any_dtype(df["fundingTime"]):
        dt_series = df["fundingTime"]
    else:
        dt_series = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)

    df["fundingTime"] = ((dt_series - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["symbol"] = symbol_upper

    # Sort and remove duplicates
    df.sort_values("fundingTime", inplace=True)
    df.drop_duplicates(subset=["fundingTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        logger.warning(f"No valid funding data remained for {symbol_upper} after cleaning.")
        return

    start_dt = pd.to_datetime(df["fundingTime"].iloc[0], unit="ms", utc=True)
    end_dt = pd.to_datetime(df["fundingTime"].iloc[-1], unit="ms", utc=True)
    logger.info(f"[{symbol_upper}] Validated {len(df)} unique records from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}.")

    # Save to scripts/others/
    local_out_path = os.path.join(SCRIPT_DIR, f"{symbol_upper.lower()}_funding_rates.parquet")
    df[["fundingTime", "fundingRate", "symbol"]].to_parquet(local_out_path, index=False)
    logger.info(f"  -> Saved local:    {local_out_path}")

    # Save to backtest/data/
    os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)
    pipeline_out_path = os.path.join(BACKTEST_DATA_DIR, f"funding_rate_{symbol_upper}.parquet")
    df[["fundingTime", "fundingRate", "symbol"]].to_parquet(pipeline_out_path, index=False)
    logger.info(f"  -> Saved pipeline: {pipeline_out_path} ✅")


def convert_csv_to_parquet(csv_file: str, fallback_symbol: str = None):
    csv_path = os.path.join(SCRIPT_DIR, csv_file) if not os.path.isabs(csv_file) else csv_file
    if not os.path.exists(csv_path):
        logger.error(f"Input CSV file not found at: {csv_path}")
        return

    logger.info(f"\n📂 Opening CSV: {os.path.basename(csv_path)}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return

    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        logger.error(f"File {os.path.basename(csv_path)} is missing required columns ('fundingTime', 'fundingRate'). Skipping.")
        return

    # Check if symbol column exists and has multiple coins
    if "symbol" in df.columns and df["symbol"].notna().any():
        unique_symbols = df["symbol"].dropna().unique()
        logger.info(f"Detected {len(unique_symbols)} symbol(s) in CSV: {list(unique_symbols)}")
        for sym in unique_symbols:
            sub_df = df[df["symbol"] == sym].copy()
            process_symbol_data(sub_df, str(sym))
    else:
        # Infer symbol from filename or fallback
        if fallback_symbol:
            sym = fallback_symbol
        else:
            fname = os.path.basename(csv_path).lower()
            if "btc" in fname:
                sym = "BTCUSDT"
            elif "eth" in fname:
                sym = "ETHUSDT"
            elif "sol" in fname:
                sym = "SOLUSDT"
            else:
                sym = "UNKNOWN_USDT"
            logger.info(f"No 'symbol' column found. Inferred symbol '{sym}' from filename.")
        process_symbol_data(df, sym)


def batch_process_all_csvs():
    csv_files = glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {SCRIPT_DIR}")
        return

    logger.info(f"Found {len(csv_files)} CSV file(s) in {SCRIPT_DIR}. Starting batch conversion...")
    for csv_file in sorted(csv_files):
        convert_csv_to_parquet(csv_file)
    logger.info("\n🎉 All CSV files processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Funding Rate CSV to Parquet Converter.")
    parser.add_argument("--csv", type=str, default="all", help="Specific CSV file name in scripts/others/ (or 'all' for batch scan)")
    parser.add_argument("--all", action="store_true", help="Batch convert all CSV files in scripts/others/")
    parser.add_argument("--symbol", type=str, default=None, help="Fallback trading symbol if not detected in CSV")
    args = parser.parse_args()

    if args.all or args.csv.lower() == "all":
        batch_process_all_csvs()
    else:
        convert_csv_to_parquet(args.csv, args.symbol)
