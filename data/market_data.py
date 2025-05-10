"""
Market Data Module
Handles fetching and processing market data from API providers.
Primary data source: Twelve Data API
"""
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple

from config.api_keys import TWELVE_DATA_API_KEY
from config.config import (
    SYMBOL, DEFAULT_TIMEFRAME, API_REQUEST_TIMEOUT, 
    API_MAX_RETRIES, API_RETRY_DELAY, DATA_DIR
)
from utils.logger import setup_logger
from utils.helpers import retry_function, format_timestamp

# Set up logger
logger = setup_logger("market_data")

class MarketData:
    """
    Class for fetching and processing market data from API providers.
    """
    def __init__(self, 
                 symbol: str = SYMBOL,
                 timeframe: str = DEFAULT_TIMEFRAME,
                 api_key: str = TWELVE_DATA_API_KEY):
        """
        Initialize the MarketData class.
        
        Args:
            symbol (str, optional): The trading symbol. Defaults to SYMBOL from config.
            timeframe (str, optional): The timeframe for data. Defaults to DEFAULT_TIMEFRAME from config.
            api_key (str, optional): The API key. Defaults to TWELVE_DATA_API_KEY from config.
        """
        self.symbol = symbol.replace("/", "")  # Remove slash for API
        self.timeframe = timeframe
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.cache_dir = os.path.join(DATA_DIR, "market_data_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Twelve Data API.
        
        Args:
            endpoint (str): The API endpoint.
            params (Dict[str, Any]): The request parameters.
            
        Returns:
            Dict[str, Any]: The API response.
        """
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key
        
        try:
            response = retry_function(
                requests.get,
                max_retries=API_MAX_RETRIES,
                retry_delay=API_RETRY_DELAY,
                url=url,
                params=params,
                timeout=API_REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    
                    # Check if API returned an error message
                    if isinstance(json_response, dict) and json_response.get("status") == "error":
                        error_msg = json_response.get("message", "Unknown API error")
                        logger.error(f"API error: {error_msg}")
                        return {"status": "error", "message": error_msg}
                    
                    return json_response
                except ValueError as e:
                    logger.error(f"Invalid JSON response: {e}")
                    return {"status": "error", "message": f"Invalid JSON response: {response.text[:100]}..."}
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return {"status": "error", "message": f"API request failed with status {response.status_code}: {response.text[:100]}..."}
                
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return {"status": "error", "message": f"API request failed: {e}"}
    
    def get_time_series(self, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        outputsize: int = 5000) -> pd.DataFrame:
        """
        Get historical time series data.
        
        Args:
            start_date (Optional[str], optional): The start date in format YYYY-MM-DD. 
                Defaults to None (uses outputsize).
            end_date (Optional[str], optional): The end date in format YYYY-MM-DD. 
                Defaults to None (uses current date).
            outputsize (int, optional): The number of data points to return if start_date is None. 
                Defaults to 5000.
                
        Returns:
            pd.DataFrame: The historical price data.
        """
        # Check if we need to use date range or outputsize
        if start_date and end_date:
            # Use date range
            params = {
                "symbol": self.symbol,
                "interval": self.timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "format": "JSON"
            }
            cache_key = f"{self.symbol}_{self.timeframe}_{start_date}_{end_date}.csv"
        else:
            # Use outputsize
            params = {
                "symbol": self.symbol,
                "interval": self.timeframe,
                "outputsize": outputsize,
                "format": "JSON"
            }
            cache_key = f"{self.symbol}_{self.timeframe}_{outputsize}.csv"
        
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Check if we have cached data that's recent enough (less than 1 hour old)
        if os.path.exists(cache_path):
            file_mod_time = os.path.getmtime(cache_path)
            if time.time() - file_mod_time < 3600:  # 1 hour in seconds
                logger.info(f"Loading cached data from {cache_path}")
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        # If not cached or cache is too old, fetch from API
        logger.info(f"Fetching time series data for {self.symbol} ({self.timeframe})")
        response = self._make_request("time_series", params)
        
        if "values" in response:
            # Process the data
            data = response["values"]
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            
            # Set volume to 0 if not available
            if "volume" not in df.columns:
                df["volume"] = 0
            else:
                df["volume"] = pd.to_numeric(df["volume"])
            
            # Set index to datetime
            df = df.set_index("datetime")
            
            # Sort by datetime (newest data last)
            df = df.sort_index()
            
            # Save to cache
            df.to_csv(cache_path)
            
            return df
        else:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"Failed to get time series data: {error_msg}")
            
            # If we have cached data, use it as fallback
            if os.path.exists(cache_path):
                logger.warning(f"Using cached data as fallback")
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            
            # Otherwise, return empty DataFrame
            return pd.DataFrame()
    
    def get_latest_price(self) -> Dict[str, Any]:
        """
        Get the latest price data.
        
        Returns:
            Dict[str, Any]: The latest price data.
        """
        logger.info(f"Fetching latest price for {self.symbol}")
        
        params = {
            "symbol": self.symbol,
            "format": "JSON"
        }
        
        response = self._make_request("price", params)
        
        if "price" in response:
            return {
                "symbol": self.symbol,
                "price": float(response["price"]),
                "timestamp": datetime.now().timestamp()
            }
        else:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"Failed to get latest price: {error_msg}")
            return {"error": error_msg}
    
    def get_forex_pairs(self) -> List[Dict[str, str]]:
        """
        Get a list of available forex pairs.
        
        Returns:
            List[Dict[str, str]]: The list of available forex pairs.
        """
        logger.info("Fetching available forex pairs")
        
        params = {
            "format": "JSON"
        }
        
        response = self._make_request("forex_pairs", params)
        
        if "data" in response:
            return response["data"]
        else:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"Failed to get forex pairs: {error_msg}")
            return []

class SyntheticDataGenerator:
    """
    Class for generating synthetic market data when API fails.
    """
    def __init__(self, 
                 symbol: str = SYMBOL,
                 timeframe: str = DEFAULT_TIMEFRAME,
                 volatility: float = 0.0001,  # Daily volatility (~10 pips)
                 drift: float = 0.0001):      # Annual drift (~250 pips)
        """
        Initialize the SyntheticDataGenerator class.
        
        Args:
            symbol (str, optional): The trading symbol. Defaults to SYMBOL from config.
            timeframe (str, optional): The timeframe for data. Defaults to DEFAULT_TIMEFRAME from config.
            volatility (float, optional): The daily volatility. Defaults to 0.0001 (10 pips).
            drift (float, optional): The annual drift. Defaults to 0.0001 (25 pips).
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.volatility = volatility
        self.drift = drift
        
        # Timeframe settings
        self.timeframe_minutes = self._get_timeframe_minutes()
        self.bars_per_day = 1440 // self.timeframe_minutes  # 1440 minutes in a day
        
        # Adjust volatility and drift for the timeframe
        self.tf_volatility = self.volatility * np.sqrt(self.timeframe_minutes / 1440)
        self.tf_drift = self.drift * (self.timeframe_minutes / (252 * 1440))
    
    def _get_timeframe_minutes(self) -> int:
        """
        Convert timeframe string to minutes.
        
        Returns:
            int: The timeframe in minutes.
        """
        unit = self.timeframe[-1]
        value = int(self.timeframe[:-1])
        
        if unit == "m":
            return value
        elif unit == "h":
            return value * 60
        elif unit == "d":
            return value * 1440
        else:
            return 60  # Default to 1h
    
    def generate_data(self, 
                      start_date: str, 
                      end_date: str, 
                      initial_price: float = 1.1000) -> pd.DataFrame:
        """
        Generate synthetic market data for the given date range.
        
        Args:
            start_date (str): The start date in format YYYY-MM-DD.
            end_date (str): The end date in format YYYY-MM-DD.
            initial_price (float, optional): The initial price. Defaults to 1.1000.
            
        Returns:
            pd.DataFrame: The synthetic price data.
        """
        logger.info(f"Generating synthetic data for {self.symbol} from {start_date} to {end_date}")
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate datetime index based on timeframe
        date_range = pd.date_range(start=start, end=end, freq=self._get_pandas_freq())
        
        # Filter out weekends
        date_range = date_range[date_range.dayofweek < 5]  # 0-4 are Monday-Friday
        
        # Number of bars
        n_bars = len(date_range)
        
        # Generate price series using geometric Brownian motion
        np.random.seed(42)  # For reproducibility
        
        # Generate daily returns
        daily_returns = np.random.normal(
            loc=self.tf_drift,
            scale=self.tf_volatility,
            size=n_bars
        )
        
        # Calculate price series
        price_series = np.zeros(n_bars)
        price_series[0] = initial_price
        
        for i in range(1, n_bars):
            price_series[i] = price_series[i-1] * (1 + daily_returns[i])
        
        # Generate OHLC data
        data = {
            "datetime": date_range,
            "open": price_series.copy(),
            "high": np.zeros(n_bars),
            "low": np.zeros(n_bars),
            "close": np.zeros(n_bars),
            "volume": np.random.randint(1000, 10000, size=n_bars)
        }
        
        # Generate realistic OHLC data
        for i in range(n_bars):
            # Random intrabar movement (typically 5-10 pips for EUR/USD)
            intrabar_range = np.random.uniform(0.0005, 0.0010)
            
            # Open price
            open_price = data["open"][i]
            
            # Determine close direction and magnitude
            close_direction = 1 if np.random.random() > 0.5 else -1
            close_change = np.random.uniform(0, intrabar_range) * close_direction
            close_price = open_price * (1 + close_change)
            
            # Determine high and low - ensure high is always highest and low is always lowest
            if close_price > open_price:
                high_price = max(open_price, close_price) + np.random.uniform(0.00001, intrabar_range / 2)
                low_price = min(open_price, close_price) - np.random.uniform(0.00001, intrabar_range / 2)
            else:
                high_price = max(open_price, close_price) + np.random.uniform(0.00001, intrabar_range / 2)
                low_price = min(open_price, close_price) - np.random.uniform(0.00001, intrabar_range / 2)
            
            # Assign values, ensuring high > open/close > low
            data["close"][i] = close_price
            data["high"][i] = high_price
            data["low"][i] = low_price
            
            # Double-check to ensure high and low are valid
            if data["high"][i] <= max(open_price, close_price):
                data["high"][i] = max(open_price, close_price) + 0.00001
                
            if data["low"][i] >= min(open_price, close_price):
                data["low"][i] = min(open_price, close_price) - 0.00001
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.set_index("datetime")
        
        return df
    
    def _get_pandas_freq(self) -> str:
        """
        Get the pandas frequency string for the timeframe.
        
        Returns:
            str: The pandas frequency string.
        """
        unit = self.timeframe[-1]
        value = int(self.timeframe[:-1])
        
        freq_map = {
            "m": "min",
            "h": "H",
            "d": "D"
        }
        
        return f"{value}{freq_map.get(unit, 'H')}"

def get_data_fallback(symbol: str = SYMBOL, 
                     timeframe: str = DEFAULT_TIMEFRAME,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     outputsize: int = 5000) -> pd.DataFrame:
    """
    Get market data with fallback to synthetic data if API fails.
    
    Args:
        symbol (str, optional): The trading symbol. Defaults to SYMBOL from config.
        timeframe (str, optional): The timeframe for data. Defaults to DEFAULT_TIMEFRAME from config.
        start_date (Optional[str], optional): The start date in format YYYY-MM-DD. 
            Defaults to None.
        end_date (Optional[str], optional): The end date in format YYYY-MM-DD. 
            Defaults to None.
        outputsize (int, optional): The number of data points to return if start_date is None. 
            Defaults to 5000.
            
    Returns:
        pd.DataFrame: The market data.
    """
    # Try to get data from API
    market_data = MarketData(symbol=symbol, timeframe=timeframe)
    df = market_data.get_time_series(start_date=start_date, end_date=end_date, outputsize=outputsize)
    
    # If API fails, generate synthetic data
    if df.empty:
        logger.warning(f"Failed to get data from API, generating synthetic data")
        
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=outputsize * int(timeframe[:-1]) // 24)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate synthetic data
        synthetic_data = SyntheticDataGenerator(symbol=symbol, timeframe=timeframe)
        df = synthetic_data.generate_data(start_date=start_date, end_date=end_date)
        
        # Add a column to flag synthetic data
        df["synthetic"] = True
    else:
        # Flag real data
        df["synthetic"] = False
    
    return df