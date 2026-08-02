import time
import requests
import pandas as pd

def fetch_binance_funding_rates(symbol="ETHUSDT", start_year=2019, end_year=2026):
    """
    Fetches historical funding rate data from Binance USDS-M Futures API with pagination.
    """
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_time = int(pd.Timestamp(f"{start_year}-01-01", tz="UTC").timestamp() * 1000)
    end_time = int(pd.Timestamp(f"{end_year}-12-31 23:59:59", tz="UTC").timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"data {params} is not found.")
            break
            
        all_data.extend(data)
        
        # Advance current_start by 1 ms past the last received timestamp to prevent duplicate records
        last_timestamp = data[-1]["fundingTime"]
        if last_timestamp >= current_start and len(data) == 1000:
            current_start = last_timestamp + 1
            time.sleep(0.1)  # Respect API rate limits
        else:
            print("Somehow we are here")
            break
            
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df.sort_values("fundingTime", inplace=True)
        df.drop_duplicates(subset=["fundingTime"], inplace=True)
        
    return df

if __name__ == "__main__":
    df_funding = fetch_binance_funding_rates("ETHUSDT", 2019, 2026)
    df_funding.to_csv("eth_funding_rates_2019_2026.csv", index=False)