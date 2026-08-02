import polars as pl
import os
import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

def check_dates(symbol, interval="4h"):
    file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_klines.parquet")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"\n--- Checking {symbol} ---")
    df = pl.read_parquet(file_path)
    print(f"Total rows: {len(df)}")
    
    if len(df) == 0:
        print("Dataframe is empty.")
        return
        
    # İlk 5 satırı al
    head = df.head(5)
    
    print("First 5 rows (Starting Dates):")
    for i, row in enumerate(head.iter_rows(named=True)):
        ts = row["open_time"]
        dt = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
        print(f"{i+1}. Row: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC | Open: {row['open']} | Close: {row['close']}")
        
    print("-" * 30)

if __name__ == "__main__":
    check_dates("BTCUSDT")
    check_dates("ETHUSDT")
