from classes.binance_data_fetcher import BinanceDataFetcher
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# Fetch data
symbol = "SOLUSDT"
interval = '5m'
start_date = '2021-01-01 00:00:00'
end_date = '2025-01-01'
csv_filepath = './data/raw/solusdt_5m_2021_2025.csv'

fetcher = BinanceDataFetcher(api_key=api_key, api_secret=api_secret)
fetcher.fetch_futures_data(symbol, interval, start_date, end_date, csv_filepath)