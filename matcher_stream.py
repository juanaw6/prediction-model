import time
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from matcher_model import calculate_score
from classes.binance_data_fetcher import BinanceDataFetcher
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pattern_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

load_dotenv()

class EfficientPatternScanner:
    def __init__(self, symbol, interval, window_size=3000, update_interval=60):
        """
        Initialize the pattern scanner
        
        Args:
            symbol: Trading pair symbol (e.g., 'SOLUSDT')
            interval: Candle timeframe (e.g., '1m', '5m')
            window_size: Number of candles to analyze
            update_interval: Seconds between updates
        """
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.fetcher = BinanceDataFetcher(api_key=self.api_key, api_secret=self.api_secret)
        
        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Calculate time units based on interval (e.g., '1m' = 1 minute)
        self.interval_unit = interval[-1]  # 'm', 'h', 'd'
        self.interval_value = int(interval[:-1])
        
        # Data storage
        self.csv_filepath = f'./data/raw/live_{symbol.lower()}_{interval}.csv'
        self.data_df = None
        self.last_timestamp = None
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        
        # Initialize data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize or load existing data"""
        if os.path.exists(self.csv_filepath):
            try:
                self.data_df = pd.read_csv(self.csv_filepath)
                # Check how timestamp is formatted in the existing CSV
                if isinstance(self.data_df['timestamp'].iloc[0], str):
                    # If timestamp is already a string date, parse it directly
                    self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
                else:
                    # Try with millisecond conversion if it's numeric
                    self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'], unit='ms')
                
                self.last_timestamp = self.data_df['timestamp'].max()
                logger.info(f"Loaded existing data until {self.last_timestamp}")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                # If file exists but is corrupt, rename it and fetch new data
                os.rename(self.csv_filepath, f"{self.csv_filepath}.bak")
                self._fetch_initial_data()
        else:
            self._fetch_initial_data()
    
    def _get_current_candle_start(self):
        """Calculate the start time of the current running candle based on the interval"""
        now = datetime.now()
        
        if self.interval_unit == 'm':
            # Floor to the start of the current minute interval
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
            # Floor to the start of the current hour interval
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
            # Floor to the start of the current day
            return datetime(now.year, now.month, now.day)
        
        # Default fallback
        return now
    
    def _fetch_initial_data(self):
        """Fetch initial data if no existing data"""
        logger.info("Fetching initial data...")
        
        # Calculate end time to exclude current running candle
        current_candle_start = self._get_current_candle_start()
        end_date = current_candle_start  # Exclude current candle
        
        # Calculate how far back to go based on interval and window_size
        if self.interval_unit == 'm':
            start_date = end_date - timedelta(minutes=self.interval_value * (self.window_size * 2))  # buffer
        elif self.interval_unit == 'h':
            start_date = end_date - timedelta(hours=self.interval_value * (self.window_size * 2))
        elif self.interval_unit == 'd':
            start_date = end_date - timedelta(days=self.interval_value * (self.window_size * 2))
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Fetching data from {start_str} to {end_str}")
        
        try:
            self.fetcher.fetch_futures_data(
                self.symbol, 
                self.interval, 
                start_str, 
                end_str, 
                self.csv_filepath
            )
            self.data_df = pd.read_csv(self.csv_filepath)
            
            if self.data_df.empty:
                logger.error("No data fetched during initialization")
                raise ValueError("No data fetched")
                
            # Handle timestamp format - inspect the actual format before converting
            sample_timestamp = self.data_df['timestamp'].iloc[0]
            logger.info(f"Sample timestamp format: {sample_timestamp} (type: {type(sample_timestamp)})")
            
            if isinstance(sample_timestamp, str):
                # Direct parsing for string timestamps
                self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
            else:
                # Numeric timestamps (likely milliseconds)
                self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'], unit='ms')
            
            self.last_timestamp = self.data_df['timestamp'].max()
            logger.info(f"Initial data fetched until {self.last_timestamp}")
        except Exception as e:
            logger.error(f"Error fetching initial data: {e}")
            raise
    
    def _fetch_new_data(self):
        """Fetch only new data since last update"""
        if self.last_timestamp is None:
            self._fetch_initial_data()
            return True
        
        logger.info(f"Fetching new data since {self.last_timestamp}")
        
        # Calculate end time to exclude current running candle
        current_candle_start = self._get_current_candle_start()
        end_date = current_candle_start  # Exclude current candle
        
        # If the last timestamp is already at or after the current candle start,
        # there are no new complete candles to fetch
        if self.last_timestamp >= current_candle_start:
            logger.info("No new complete candles available since last update")
            return False
            
        # Add a small buffer to avoid missing candles (10 minutes or 2 candles, whichever is more)
        buffer_minutes = max(10, self.interval_value * 2)
        start_date = self.last_timestamp - timedelta(minutes=buffer_minutes)
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Fetching data from {start_str} to {end_str}")
        
        # Temporary file for new data
        temp_file = f"./data/raw/temp_{self.symbol.lower()}_{self.interval}.csv"
        
        try:
            self.fetcher.fetch_futures_data(
                self.symbol, 
                self.interval, 
                start_str, 
                end_str, 
                temp_file
            )
            
            # Load new data
            new_data = pd.read_csv(temp_file)
            if new_data.empty:
                logger.info("No new data available")
                return False
            
            # Handle timestamp format - inspect the actual format before converting
            sample_timestamp = new_data['timestamp'].iloc[0]
            
            if isinstance(sample_timestamp, str):
                # Direct parsing for string timestamps
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            else:
                # Numeric timestamps (likely milliseconds)
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            
            # Remove any overlaps with existing data
            if self.data_df is not None:
                new_data = new_data[new_data['timestamp'] > self.last_timestamp]
            
            if new_data.empty:
                logger.info("No new data after filtering")
                return False
            
            logger.info(f"Found {len(new_data)} new candles, latest timestamp: {new_data['timestamp'].max()}")
            
            # Append new data
            if self.data_df is None:
                self.data_df = new_data
            else:
                self.data_df = pd.concat([self.data_df, new_data], ignore_index=True)
            
            # Update last timestamp
            self.last_timestamp = self.data_df['timestamp'].max()
            
            # Save updated data
            self.data_df.to_csv(self.csv_filepath, index=False)
            
            logger.info(f"Added {len(new_data)} new candles. Latest: {self.last_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching new data: {e}")
            return False
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def analyze_patterns(self):
        """Analyze patterns in the most recent window of data"""
        if self.data_df is None or len(self.data_df) < self.window_size:
            logger.warning(f"Not enough data for analysis. Have {len(self.data_df) if self.data_df is not None else 0} rows, need {self.window_size}")
            return None
        
        # Get recent window_size candles
        recent_data = self.data_df.tail(self.window_size).copy()
        
        # Calculate percentage changes
        recent_data['change'] = ((recent_data['close'] - recent_data['open']) / recent_data['open']) * 100
        changes = recent_data['change'].values.tolist()
        
        # Run pattern analysis
        try:
            start_time = time.time()
            result = calculate_score(changes)
            duration = time.time() - start_time
            logger.info(f"Pattern analysis completed in {duration:.2f} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return None
    
    def run(self):
        """Main loop to continuously update and analyze data"""
        logger.info(f"Starting pattern scanner for {self.symbol} ({self.interval})")
        
        while True:
            try:
                # Fetch new data
                data_updated = self._fetch_new_data()
                
                # Analyze if we have new data or first run
                if data_updated:
                    result = self.analyze_patterns()
                    if result:
                        logger.info(f"Total score: {result.total_score}")
                
                # Determine wait time until next candle closes
                current_candle_start = self._get_current_candle_start()
                if self.interval_unit == 'm':
                    next_candle_start = current_candle_start + timedelta(minutes=self.interval_value)
                elif self.interval_unit == 'h':
                    next_candle_start = current_candle_start + timedelta(hours=self.interval_value)
                elif self.interval_unit == 'd':
                    next_candle_start = current_candle_start + timedelta(days=self.interval_value)
                
                now = datetime.now()
                seconds_until_next_candle = (next_candle_start - now).total_seconds()
                
                # Add 10 seconds buffer after candle close for data to be available
                wait_time = max(10, int(seconds_until_next_candle) + 10)
                wait_time = min(wait_time, self.update_interval)  # Don't wait longer than update_interval
                
                logger.info(f"Next candle closes at {next_candle_start}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("Pattern scanner stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                # Wait a bit before retrying to avoid spamming errors
                time.sleep(60)


if __name__ == "__main__":
    # Check Binance API credentials
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        logger.error("Binance API credentials not found in environment variables")
        exit(1)
        
    scanner = EfficientPatternScanner(
        symbol="SOLUSDT",
        interval="1m",
        window_size=3000,
        update_interval=60  # Update every minute for 1m candles
    )
    scanner.run()