"""
Helper Functions
Utility functions used throughout the application.
"""
import os
import json
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional

from utils.logger import setup_logger

logger = setup_logger("helpers")

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime.datetime:
    """
    Convert a timestamp to a datetime object.
    
    Args:
        timestamp (Union[int, float]): The timestamp to convert.
        
    Returns:
        datetime.datetime: The converted datetime object.
    """
    return datetime.datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime.datetime) -> int:
    """
    Convert a datetime object to a timestamp.
    
    Args:
        dt (datetime.datetime): The datetime object to convert.
        
    Returns:
        int: The converted timestamp.
    """
    return int(dt.timestamp())

def format_timestamp(timestamp: Union[int, float], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp as a string.
    
    Args:
        timestamp (Union[int, float]): The timestamp to format.
        format_str (str, optional): The format string. Defaults to "%Y-%m-%d %H:%M:%S".
        
    Returns:
        str: The formatted timestamp.
    """
    return timestamp_to_datetime(timestamp).strftime(format_str)

def format_price(price: float, decimals: int = 5) -> str:
    """
    Format a price with the appropriate number of decimal places.
    
    Args:
        price (float): The price to format.
        decimals (int, optional): The number of decimal places. Defaults to 5.
        
    Returns:
        str: The formatted price.
    """
    return f"{price:.{decimals}f}"

def calculate_pip_value(symbol: str, price: float, lot_size: float = 1.0) -> float:
    """
    Calculate the pip value for a given currency pair and price.
    
    Args:
        symbol (str): The currency pair symbol.
        price (float): The current price.
        lot_size (float, optional): The lot size. Defaults to 1.0.
        
    Returns:
        float: The pip value in the account currency.
    """
    pip_values = {
        "EUR/USD": 10.0,
        "GBP/USD": 10.0,
        "USD/JPY": 1000.0 / price,
        "USD/CHF": 10.0 / price,
        "AUD/USD": 10.0,
        "USD/CAD": 10.0 / price,
        "NZD/USD": 10.0,
    }
    
    # Standardize symbol format
    symbol = symbol.upper().replace("/", "")
    standard_symbol = f"{symbol[:3]}/{symbol[3:]}"
    
    # Use the pip value for the given symbol, or default to EUR/USD
    pip_value = pip_values.get(standard_symbol, 10.0)
    
    # Adjust for lot size
    pip_value *= lot_size
    
    return pip_value

def pip_difference(start_price: float, end_price: float, symbol: str = "EUR/USD") -> float:
    """
    Calculate the difference in pips between two prices.
    
    Args:
        start_price (float): The starting price.
        end_price (float): The ending price.
        symbol (str, optional): The currency pair symbol. Defaults to "EUR/USD".
        
    Returns:
        float: The difference in pips.
    """
    # For most currency pairs, 1 pip = 0.0001
    # For JPY pairs, 1 pip = 0.01
    pip_size = 0.0001
    if "JPY" in symbol:
        pip_size = 0.01
    
    return (end_price - start_price) / pip_size

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path (Union[str, Path]): The path to the JSON file.
        
    Returns:
        Dict[str, Any]: The loaded JSON data.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data (Dict[str, Any]): The data to save.
        file_path (Union[str, Path]): The path to the JSON file.
        
    Returns:
        bool: True if the data was saved successfully, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def retry_function(func, max_retries: int = 3, retry_delay: int = 1, *args, **kwargs):
    """
    Retry a function execution with exponential backoff.
    
    Args:
        func: The function to execute.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
        retry_delay (int, optional): The initial delay between retries in seconds. Defaults to 1.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Any: The result of the function execution.
        
    Raises:
        Exception: If the function fails after the maximum number of retries.
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Function {func.__name__} failed, retry {retries}/{max_retries}: {e}")
            
            if retries == max_retries:
                break
                
            # Exponential backoff
            sleep_time = retry_delay * (2 ** retries)
            time.sleep(sleep_time)
            retries += 1
    
    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {last_exception}")
    raise last_exception

def is_market_open(timestamp: Optional[Union[int, float]] = None) -> bool:
    """
    Check if the forex market is open at the given timestamp.
    
    Args:
        timestamp (Optional[Union[int, float]], optional): The timestamp to check. 
            Defaults to None (current time).
        
    Returns:
        bool: True if the market is open, False otherwise.
    """
    from config.config import FOREX_MARKET_HOURS
    
    if timestamp is None:
        dt = datetime.datetime.now()
    else:
        dt = timestamp_to_datetime(timestamp)
    
    # Forex market is 24/5, closed on weekends
    weekday = dt.weekday()  # Monday is 0, Sunday is 6
    
    # Check if it's weekend
    if weekday == 5:  # Saturday
        return False
    
    if weekday == 6:  # Sunday
        # Market opens Sunday evening
        sunday_open = datetime.datetime.strptime(FOREX_MARKET_HOURS["sunday_open"], "%H:%M").time()
        return dt.time() >= sunday_open
    
    if weekday == 4:  # Friday
        # Market closes Friday evening
        friday_close = datetime.datetime.strptime(FOREX_MARKET_HOURS["friday_close"], "%H:%M").time()
        return dt.time() <= friday_close
    
    # Check daily maintenance window
    maintenance_start = datetime.datetime.strptime(FOREX_MARKET_HOURS["daily_maintenance"][0], "%H:%M").time()
    maintenance_end = datetime.datetime.strptime(FOREX_MARKET_HOURS["daily_maintenance"][1], "%H:%M").time()
    
    if maintenance_start <= dt.time() <= maintenance_end:
        return False
    
    # If we've made it here, the market is open
    return True

def get_timeframe_minutes(timeframe: str) -> int:
    """
    Convert a timeframe string to minutes.
    
    Args:
        timeframe (str): The timeframe string (e.g., "1m", "1h", "1d").
        
    Returns:
        int: The timeframe in minutes.
    """
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    
    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 1440  # 24 * 60
    elif unit == "w":
        return value * 10080  # 7 * 24 * 60
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def resample_dataframe(df: pd.DataFrame, source_timeframe: str, target_timeframe: str) -> pd.DataFrame:
    """
    Resample a dataframe to a different timeframe.
    
    Args:
        df (pd.DataFrame): The dataframe to resample.
        source_timeframe (str): The original timeframe.
        target_timeframe (str): The target timeframe.
        
    Returns:
        pd.DataFrame: The resampled dataframe.
    """
    # Convert source and target timeframes to pandas frequency strings
    freq_map = {
        "m": "min",
        "h": "H",
        "d": "D",
        "w": "W",
    }
    
    source_unit = source_timeframe[-1].lower()
    source_value = int(source_timeframe[:-1])
    source_freq = f"{source_value}{freq_map[source_unit]}"
    
    target_unit = target_timeframe[-1].lower()
    target_value = int(target_timeframe[:-1])
    target_freq = f"{target_value}{freq_map[target_unit]}"
    
    # Set index to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.DatetimeIndex(df.index))
    
    # Resample
    resampled = df.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled

def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Calculate the Average True Range (ATR) for a dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe with OHLC data.
        period (int, optional): The ATR period. Defaults to 14.
        
    Returns:
        np.ndarray: The ATR values.
    """
    high = df['high'].values
    low = df['low'].values
    close = np.concatenate(([0], df['close'].values[:-1]))
    
    # Calculate True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - close)
    tr3 = np.abs(low - close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using simple moving average
    atr = np.zeros_like(true_range)
    atr[:period] = np.nan
    atr[period] = np.mean(true_range[1:period+1])
    
    # Calculate smoothed ATR
    for i in range(period + 1, len(true_range)):
        atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
    
    return atr

def calculate_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Calculate the maximum drawdown and drawdown duration from an equity curve.
    
    Args:
        equity_curve (np.ndarray): The equity curve.
        
    Returns:
        tuple: (maximum drawdown as percentage, maximum drawdown duration in bars)
    """
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve - running_max) / running_max * 100
    
    # Find the maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Calculate drawdown duration
    drawdown_duration = 0
    current_duration = 0
    peak = equity_curve[0]
    
    for value in equity_curve:
        if value >= peak:
            peak = value
            current_duration = 0
        else:
            current_duration += 1
            drawdown_duration = max(drawdown_duration, current_duration)
    
    return max_drawdown, drawdown_duration

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio of a returns series.
    
    Args:
        returns (np.ndarray): The returns series.
        risk_free_rate (float, optional): The risk-free rate. Defaults to 0.0.
        annualization_factor (int, optional): The annualization factor. Defaults to 252 (trading days).
        
    Returns:
        float: The Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)
    return sharpe

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sortino ratio of a returns series.
    
    Args:
        returns (np.ndarray): The returns series.
        risk_free_rate (float, optional): The risk-free rate. Defaults to 0.0.
        annualization_factor (int, optional): The annualization factor. Defaults to 252 (trading days).
        
    Returns:
        float: The Sortino ratio.
    """
    excess_returns = returns - risk_free_rate
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(annualization_factor)
    return sortino

def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate the profit factor of a returns series.
    
    Args:
        returns (np.ndarray): The returns series.
        
    Returns:
        float: The profit factor.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return gains / losses

def convert_to_relative_path(absolute_path: str) -> str:
    """
    Convert an absolute path to a relative path from the current working directory.
    
    Args:
        absolute_path (str): The absolute path.
        
    Returns:
        str: The relative path.
    """
    return os.path.relpath(absolute_path, os.getcwd())