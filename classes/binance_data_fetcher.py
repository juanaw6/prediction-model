from binance.client import Client
import pandas as pd
import time
from typing import List, Optional, Dict

class BinanceDataFetcher:
    """
    A wrapper class to fetch Binance Futures and Spot market data with robust error handling.
    
    Attributes:
        client (Client): The Binance API client.
        interval_durations (Dict[str, int]): Duration of intervals in milliseconds.
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Initializes the BinanceDataFetcher with API credentials.

        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
        """
        self.client = Client(api_key, api_secret)
        
        self.interval_durations: Dict[str, int] = {
            '1m': 1 * 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }

    def _get_binance_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: int, 
        end_time: Optional[int] = None, 
        limit: int = 1500, 
        futures: bool = True, 
        max_retries: int = 3
    ) -> List[List]:
        """
        Fetches kline data from Binance with robust error handling and retry mechanism.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The interval for the kline data (e.g., '5m', '1h').
            start_time (int): Start time in milliseconds.
            end_time (int, optional): End time in milliseconds.
            limit (int, optional): Maximum number of data points to fetch.
            futures (bool, optional): Fetch Futures data if True, Spot data otherwise.
            max_retries (int, optional): Number of retry attempts.

        Returns:
            List[List]: A list of kline data.
        """
        for attempt in range(max_retries):
            try:
                if futures:
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=start_time,
                        endTime=end_time,
                        limit=limit
                    )
                else:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=start_time,
                        endTime=end_time,
                        limit=limit
                    )
                return klines
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error fetching data (attempt {attempt + 1}): {e}. Retrying...")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Failed to fetch data after {max_retries} attempts: {e}")
                    return []

    def _preprocess_data(self, candles: List[List]) -> pd.DataFrame:
        """
        Preprocesses and validates fetched candle data.

        Args:
            candles (List[List]): Raw candle data from Binance.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.
        """
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(candles, columns=columns)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df['changes'] = ((df['close'] - df['open']) / df['open'] * 100).round(3)

        # return df[['timestamp', 'open', 'high', 'low', 'close', 'changes', 'volume']].dropna()
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()

    def fetch_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: str, 
        end_date: str, 
        output_file: str, 
        futures: bool = True
    ) -> pd.DataFrame:
        """
        Fetches market data (Futures or Spot) and saves it as a CSV file.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The interval for the kline data (e.g., '5m', '1h').
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            output_file (str): The path to save the CSV file.
            futures (bool, optional): Fetch Futures data if True, Spot data otherwise.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the fetched data.
        """
        start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)

        candles = []
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            try:
                new_candles = self._get_binance_klines(
                    symbol, 
                    interval, 
                    start_time=current_start, 
                    end_time=end_timestamp, 
                    limit=1500, 
                    futures=futures
                )

                if not new_candles:
                    break

                candles.extend(new_candles)

                last_candle_time = new_candles[-1][0]
                current_start = last_candle_time + self.interval_durations.get(interval, 60000)

                print(f"Fetched up to: {pd.to_datetime(last_candle_time, unit='ms')}")

                time.sleep(0.3)
            except Exception as e:
                print(f"Error in data fetch: {e}")
                break
            
        df = self._preprocess_data(candles)

        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

        return df

    def fetch_futures_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: str, 
        end_date: str, 
        output_file: str
    ) -> pd.DataFrame:
        """
        Fetches Futures market data and saves it as a CSV file.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The interval for the kline data (e.g., '5m', '1h').
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            output_file (str): The path to save the CSV file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the fetched data.
        """
        return self.fetch_data(symbol, interval, start_date, end_date, output_file, futures=True)

    def fetch_spot_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: str, 
        end_date: str, 
        output_file: str
    ) -> pd.DataFrame:
        """
        Fetches Spot market data and saves it as a CSV file.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The interval for the kline data (e.g., '5m', '1h').
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            output_file (str): The path to save the CSV file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the fetched data.
        """
        return self.fetch_data(symbol, interval, start_date, end_date, output_file, futures=False)

# Example Usage
# from apikey import api_key, api_secret
# fetcher = BinanceDataFetcher(api_key, api_secret)
# df_futures = fetcher.fetch_futures_data('BTCUSDT', '5m', '2023-01-01', '2023-02-01', 'btc_futures.csv')
# df_spot = fetcher.fetch_spot_data('BTCUSDT', '5m', '2023-01-01', '2023-02-01', 'btc_spot.csv')