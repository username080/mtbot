import polars as pl
import os
import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

def check_dollar_bars(symbol):
    file_path = os.path.join(DATA_DIR, f"dollar_bars_{symbol}_all.parquet")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"\n--- Checking Dollar Bars for {symbol} ---")
    
    # Read the parquet file
    df = pl.read_parquet(file_path)
    print(f"Total Dollar Bars: {len(df)}")
    
    if len(df) == 0:
        print("Dataframe is empty.")
        return
        
    # Get first 5 rows
    head = df.head(5)
    
    print("First 5 Dollar Bars (Starting Dates):")
    for i, row in enumerate(head.iter_rows(named=True)):
        # Dollar bars timestamp format is usually milliseconds
        ts_open = row.get("open_time")
        ts_close = row.get("close_time")
        
        dt_open = datetime.datetime.fromtimestamp(ts_open / 1000, tz=datetime.timezone.utc) if ts_open else None
        dt_close = datetime.datetime.fromtimestamp(ts_close / 1000, tz=datetime.timezone.utc) if ts_close else None
        
        open_str = dt_open.strftime('%Y-%m-%d %H:%M:%S') if dt_open else "N/A"
        
        print(f"{i+1}. Bar: OpenTime: {open_str} UTC | Open: {row.get('open')} | Close: {row.get('close')} | Vol: {row.get('volume'):.2f}")
        
    print("-" * 50)

if __name__ == "__main__":
    check_dollar_bars("BTCUSDT")
    check_dollar_bars("ETHUSDT")
