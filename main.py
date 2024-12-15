from BinanceDataFetcher import BinanceDataFetcher
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

fetcher = BinanceDataFetcher(api_key=api_key, api_secret=api_secret)

symbol = "BTCUSDT"
interval = '5m'
start_date = '2024-12-01'
end_date = '2024-12-17'
csv_filepath = 'futures_data.csv'

fetcher.fetch_futures_data(symbol, interval, start_date, end_date, csv_filepath)