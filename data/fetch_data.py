import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_data(symbol="PAXGUSDT", interval="15m", days_back=365):
    print(f"Fetching {days_back} days of {interval} data for {symbol}...")
    
    # Binance API requires timestamps in milliseconds
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    all_klines = []
    limit = 1000  # Binance's maximum candles per single request
    current_start = start_time
    
    while current_start < end_time:
        # Construct the REST API URL
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_start}&limit={limit}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if not data or type(data) is dict: 
                # If dict is returned, it might be an API error message
                print("API returned an error or empty data:", data)
                break
                
            all_klines.extend(data)
            
            # Step forward: set the next start time to the close time of the last fetched candle + 1 millisecond
            current_start = data[-1][6] + 1
            
            print(f"Fetched {len(all_klines)} candles so far...")
            time.sleep(0.1)  # Brief pause to respect API rate limits
            
        except Exception as e:
            print(f"Network error fetching data: {e}")
            break
            
    # Binance's standard 12-column response format
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(all_klines, columns=columns)
    
    # Convert UNIX timestamp to a readable datetime format
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Filter down to only the columns required for feature engineering
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']]
    
    # Convert string price values to floats
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    
    # Drop any potential duplicates caused by API pagination overlaps
    df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Execute the fetch function
    df = fetch_binance_data(symbol="PAXGUSDT", interval="15m", days_back=365)
    
    # Save to a local CSV file
    filename = "paxg_15m_1year.csv"
    df.to_csv(filename, index=False)
    
    print(f"\nSuccess! Dataset securely saved to {filename}.")
    print(f"Total rows fetched: {len(df)} (Expect ~35,040 for 1 year of 15m candles)")
    print("\nFirst 5 rows:")
    print(df.head())