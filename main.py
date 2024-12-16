from classes.binance_data_fetcher import BinanceDataFetcher
from classes.preprocessor import Preprocessor
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

fetcher = BinanceDataFetcher(api_key=api_key, api_secret=api_secret)

# Retrieval Params
symbol = "BTCUSDT"
interval = '5m'
start_date = '2024-12-16 08:00:00'
end_date = '2024-12-16 16:00:00'
csv_filepath = 'data_fetched.csv'

fetcher.fetch_futures_data(symbol, interval, start_date, end_date, csv_filepath)

# Preprocess
preprocessor = Preprocessor()

df = pd.read_csv(csv_filepath)

preprocessor.transform(df, 'data_preprocessed.csv')