import requests
import pandas as pd

def fetch_ohlcv():
    url = "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_ETH_BTC/history?" \
          "period_id=1DAY&time_start=2022-01-01T00:00:00&time_end=2024-12-09T23:59:59&limit=100000"
    headers = { "X-CoinAPI-Key": "API-KEY" }  # Replace with your API key

    response = requests.get(url, headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        if response.content:
            return response.json()
        else:
            print("Response is empty.")
            return None
    else:
        # Handle other HTTP status codes
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
    
df = pd.DataFrame(fetch_ohlcv())

file_path = r"[file_path]\bitcoin_price.csv"
df.to_csv(file_path)