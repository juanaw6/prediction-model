import time
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from matcher_model import calculate_score
from classes.binance_data_fetcher import BinanceDataFetcher
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

load_dotenv()

class PatternMatcherStream:
    def __init__(self, symbol, interval, window_size=3000, update_interval=60):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.fetcher = BinanceDataFetcher(api_key=self.api_key, api_secret=self.api_secret)
        
        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.update_interval = update_interval
        
        self.interval_unit = interval[-1]
        self.interval_value = int(interval[:-1])
        
        self.csv_filepath = f'./data/raw/live_{symbol.lower()}_{interval}.csv'
        self.data_df = None
        self.last_timestamp = None
        
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        
        self._initialize_data()
    
    def _initialize_data(self):
        if os.path.exists(self.csv_filepath):
            try:
                self.data_df = pd.read_csv(self.csv_filepath)
                if isinstance(self.data_df['timestamp'].iloc[0], str):
                    self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
                else:
                    self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'], unit='ms')
                
                self.last_timestamp = self.data_df['timestamp'].max()
                logger.info(f"Loaded existing data until {self.last_timestamp}")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                os.rename(self.csv_filepath, f"{self.csv_filepath}.bak")
                self._fetch_initial_data()
        else:
            self._fetch_initial_data()
    
    def _get_current_candle_start(self):
        now = datetime.now()
        
        if self.interval_unit == 'm':
            minutes_to_subtract = now.minute % self.interval_value
            seconds_to_subtract = now.second
            microseconds_to_subtract = now.microsecond
            
            current_candle_start = now - timedelta(
                minutes=minutes_to_subtract, 
                seconds=seconds_to_subtract,
                microseconds=microseconds_to_subtract
            )
            return current_candle_start
            
        elif self.interval_unit == 'h':
            hours_to_subtract = now.hour % self.interval_value
            minutes_to_subtract = now.minute
            seconds_to_subtract = now.second
            microseconds_to_subtract = now.microsecond
            
            current_candle_start = now - timedelta(
                hours=hours_to_subtract,
                minutes=minutes_to_subtract, 
                seconds=seconds_to_subtract,
                microseconds=microseconds_to_subtract
            )
            return current_candle_start
            
        elif self.interval_unit == 'd':
            return datetime(now.year, now.month, now.day)
        
        return now
    
    def _fetch_initial_data(self):
        logger.info("Fetching initial data...")

        current_candle_start = self._get_current_candle_start()
        end_date = current_candle_start
        
        if self.interval_unit == 'm':
            start_date = end_date - timedelta(minutes=self.interval_value * (self.window_size * 2))
        elif self.interval_unit == 'h':
            start_date = end_date - timedelta(hours=self.interval_value * (self.window_size * 2))
        elif self.interval_unit == 'd':
            start_date = end_date - timedelta(days=self.interval_value * (self.window_size * 2))
        
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Fetching data from {start_str} to {end_str}")
        
        try:
            self.fetcher.fetch_futures_data(
                symbol=self.symbol, 
                interval=self.interval, 
                start_date=start_str, 
                end_date=end_str, 
                output_file=self.csv_filepath,
                verbose=False
            )
            self.data_df = pd.read_csv(self.csv_filepath)
            
            if self.data_df.empty:
                logger.error("No data fetched during initialization")
                raise ValueError("No data fetched")
                
            sample_timestamp = self.data_df['timestamp'].iloc[0]
            logger.info(f"Sample timestamp format: {sample_timestamp} (type: {type(sample_timestamp)})")
            
            if isinstance(sample_timestamp, str):
                self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
            else:
                self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'], unit='ms')
                
            self.data_df = self.data_df.head(-1)
            
            self.last_timestamp = self.data_df['timestamp'].max()
            logger.info(f"Initial data fetched until {self.last_timestamp}")
        except Exception as e:
            logger.error(f"Error fetching initial data: {e}")
            raise
    
    def _fetch_new_data(self):
        if self.last_timestamp is None:
            self._fetch_initial_data()
            return True
        
        current_candle_start = self._get_current_candle_start()
        end_date = current_candle_start
        
        if self.last_timestamp >= current_candle_start:
            logger.info("No new complete candles available since last update")
            return False
        
        buffer_minutes = max(10, self.interval_value * 2)
        start_date = self.last_timestamp - timedelta(minutes=buffer_minutes)

        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        temp_file = f"./data/raw/temp_{self.symbol.lower()}_{self.interval}.csv"
        
        try:
            self.fetcher.fetch_futures_data(
                symbol=self.symbol, 
                interval=self.interval, 
                start_date=start_str, 
                end_date=end_str, 
                output_file=temp_file,
                verbose=False
            )
            
            new_data = pd.read_csv(temp_file)
            if new_data.empty:
                logger.info("No new data available")
                return False
            
            sample_timestamp = new_data['timestamp'].iloc[0]
            
            if isinstance(sample_timestamp, str):
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            else:
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            
            if self.data_df is not None:
                new_data = new_data[new_data['timestamp'] > self.last_timestamp]
            
            if new_data.empty:
                logger.info("No new data after filtering")
                return False
            
            if self.data_df is None:
                self.data_df = new_data
            else:
                self.data_df = pd.concat([self.data_df, new_data], ignore_index=True)
                
            self.data_df = self.data_df.head(-1)

            self.last_timestamp = self.data_df['timestamp'].max()

            self.data_df.to_csv(self.csv_filepath, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Error fetching new data: {e}")
            return False
        finally:

            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def analyze_patterns(self):
        if self.data_df is None or len(self.data_df) < self.window_size:
            logger.warning(f"Not enough data for analysis. Have {len(self.data_df) if self.data_df is not None else 0} rows, need {self.window_size}")
            return None

        recent_data = self.data_df.tail(self.window_size).copy()

        recent_data['change'] = ((recent_data['close'] - recent_data['open']) / recent_data['open']) * 100
        changes = recent_data['change'].values.tolist()

        try:
            start_time = time.time()
            result = calculate_score(changes)
            duration = time.time() - start_time
            latest_candle_time = recent_data['timestamp'].iloc[-1]

            logger.info(f"Match duration : {duration*1000:.0f} ms")
            logger.info(f"Matched date   : {latest_candle_time}")
            
            return result
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return None

    def run(self):
        logger.info(f"Starting pattern matcher for {self.symbol} ({self.interval})")
        
        while True:
            try:
                data_updated = self._fetch_new_data()
                
                if data_updated:
                    result = self.analyze_patterns()
                    if result:
                        logger.info(f"Total score    : {result.total_score}")
                
                current_candle_start = self._get_current_candle_start()
                if self.interval_unit == 'm':
                    next_candle_start = current_candle_start + timedelta(minutes=self.interval_value)
                elif self.interval_unit == 'h':
                    next_candle_start = current_candle_start + timedelta(hours=self.interval_value)
                elif self.interval_unit == 'd':
                    next_candle_start = current_candle_start + timedelta(days=self.interval_value)
                
                now = datetime.now()
                seconds_until_next_candle = (next_candle_start - now).total_seconds()
                
                wait_time = max(10, int(seconds_until_next_candle) + 10)
                wait_time = min(wait_time, self.update_interval)
                
                logger.info(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("Pattern matcher stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        logger.error("Binance API credentials not found in environment variables")
        exit(1)
        
    matcher = PatternMatcherStream(
        symbol="SOLUSDT",
        interval="1m",
        window_size=3000,
        update_interval=60
    )
    matcher.run()