import urllib.request
import json
import time
import datetime
import polars as pl
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "4h"

# 2019-01-01 00:00:00 UTC
START_TS = int(datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
# 2026-01-01 00:00:00 UTC (exclusive of 2026)
END_TS = int(datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)

# Verilerin kaydedileceği dizin
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_historical_klines(symbol):
    logger.info(f"Downloading {INTERVAL} klines for {symbol} from 2019 to 2026...")
    all_klines = []
    current_start = START_TS
    
    while current_start < END_TS:
        # FAPI (Futures) veya normal API (Spot) endpoint'i
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={INTERVAL}&limit=1000&startTime={current_start}&endTime={END_TS}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            if not data:
                break
                
            all_klines.extend(data)
            last_close_time = data[-1][6]
            last_date_str = datetime.datetime.fromtimestamp(last_close_time/1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')
            
            logger.info(f"[{symbol}] Fetched {len(data)} klines, reached up to: {last_date_str}")
            
            # Eğer limit olan 1000'den az geldiyse, hedeflenen tarihin sonuna ulaşmışızdır.
            if len(data) < 1000:
                break
                
            current_start = last_close_time + 1
            
            # Binance'in IP bazlı bot korumasına takılmamak için 0.5 saniye bekleme.
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    if all_klines:
        logger.info(f"Total {len(all_klines)} klines fetched for {symbol}. Converting to Polars DataFrame...")
        
        # Binance kline sütun isimleri
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore"
        ]
        
        df = pl.DataFrame(all_klines, schema=columns, orient="row")
        
        # Binance verileri metin (string) formatında gönderir, bunları uygun float/int formatlarına çeviriyoruz
        float_cols = ["open", "high", "low", "close", "volume", 
                      "quote_asset_volume", "taker_buy_base_asset_volume", 
                      "taker_buy_quote_asset_volume", "ignore"]
        
        df = df.with_columns([
            pl.col(c).cast(pl.Float64) for c in float_cols
        ])
        
        df = df.with_columns([
            pl.col("open_time").cast(pl.Int64),
            pl.col("close_time").cast(pl.Int64),
            pl.col("number_of_trades").cast(pl.Int64),
        ])
        
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}_{INTERVAL}_klines.parquet")
        df.write_parquet(out_path)
        
        logger.info(f"SUCCESS: Saved {len(df)} records for {symbol} to {out_path}\n")
    else:
        logger.warning(f"No data fetched for {symbol}\n")

if __name__ == "__main__":
    for sym in SYMBOLS:
        download_historical_klines(sym)
