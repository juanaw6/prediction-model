from classes.binance_data_fetcher import BinanceDataFetcher
from classes.preprocessor import Preprocessor
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# Fetch data
symbol = "ETHUSDT"
interval = '5m'
start_date = '2024-08-01 00:00:00'
end_date = '2024-12-22'
csv_filepath = 'data_fetched.csv'

fetcher = BinanceDataFetcher(api_key=api_key, api_secret=api_secret)
fetcher.fetch_futures_data(symbol, interval, start_date, end_date, csv_filepath)

# Preprocess
df = pd.read_csv(csv_filepath)

preprocessor = Preprocessor()
preprocessor.transform_to_csv(df, 'data_preprocessed.csv')