"""
Base Strategy Class
Defines the base class for all trading strategies.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Any, Optional, Tuple
from abc import ABC, abstractmethod

from utils.logger import setup_logger
from data.sentiment import SentimentAnalyzer

logger = setup_logger("strategy")

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        Initialize the Strategy class.
        
        Args:
            name (str): The name of the strategy.
            params (Dict[str, Any]): The strategy parameters.
        """
        self.name = name
        self.params = params
        self.signals = []
        self.current_position = None
        self.use_sentiment = True  # Flag to incorporate sentiment analysis
        
        logger.info(f"Initialized {name} strategy with parameters: {params}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given data.
        
        Args:
            df (pd.DataFrame): The OHLCV data.
            
        Returns:
            pd.DataFrame: The data with signals added.
        """
        pass
    
    def apply_sentiment_filter(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply sentiment filter to trading signals.
    
        Args:
            df (pd.DataFrame): The dataframe with signals.
            symbol (str): The trading symbol.
        
        Returns:
            pd.DataFrame: The dataframe with sentiment-adjusted signals.
        """
        if not self.use_sentiment:
            return df
    
        # If we don't have signal column, return the dataframe as is
        if 'signal' not in df.columns:
            return df
    
        # Get current sentiment
        try:
            sentiment_analyzer = SentimentAnalyzer(symbol=symbol)
            sentiment = sentiment_analyzer.get_sentiment_signal()
        
            # Check if sentiment is a dictionary before proceeding
            if not isinstance(sentiment, dict):
                logger.warning(f"Invalid sentiment data type: {type(sentiment)}")
                return df
            
            # Create a new column for sentiment-adjusted signals if it doesn't exist
            if 'sentiment_signal' not in df.columns:
                df['sentiment_signal'] = df['signal'].copy()
        
            # Only adjust recent signals (last day)
            if len(df) > 0:
                recent_mask = df.index >= df.index[-1] - pd.Timedelta(days=1)
            
                # Apply sentiment filter
                if sentiment.get('signal') == 'buy':
                    # Strengthen buy signals, weaken sell signals
                    buy_mask = recent_mask & (df['signal'] > 0)
                    sell_mask = recent_mask & (df['signal'] < 0)
                
                    if buy_mask.any():
                        df.loc[buy_mask, 'sentiment_signal'] = df.loc[buy_mask, 'signal'] * (1 + sentiment.get('confidence', 0) * 0.5)
                
                    if sell_mask.any():
                        df.loc[sell_mask, 'sentiment_signal'] = df.loc[sell_mask, 'signal'] * (1 - sentiment.get('confidence', 0) * 0.5)
            
                elif sentiment.get('signal') == 'sell':
                    # Strengthen sell signals, weaken buy signals
                    buy_mask = recent_mask & (df['signal'] > 0)
                    sell_mask = recent_mask & (df['signal'] < 0)
                
                    if sell_mask.any():
                        df.loc[sell_mask, 'sentiment_signal'] = df.loc[sell_mask, 'signal'] * (1 + sentiment.get('confidence', 0) * 0.5)
                
                    if buy_mask.any():
                        df.loc[buy_mask, 'sentiment_signal'] = df.loc[buy_mask, 'signal'] * (1 - sentiment.get('confidence', 0) * 0.5)
            
                else:
                    # Neutral sentiment, no adjustment
                    if recent_mask.any():
                        df.loc[recent_mask, 'sentiment_signal'] = df.loc[recent_mask, 'signal']
            
                # For older signals, just copy the original signal
                older_mask = ~recent_mask
                if older_mask.any():
                    df.loc[older_mask, 'sentiment_signal'] = df.loc[older_mask, 'signal']
            
                # Log sentiment impact
                logger.info(f"Applied sentiment filter: {sentiment.get('signal', 'neutral')} with confidence {sentiment.get('confidence', 0):.2f}")
        
        except Exception as e:
            logger.error(f"Error applying sentiment filter: {e}")
            # If there's an error, just use the original signals
            if 'sentiment_signal' not in df.columns:
                df['sentiment_signal'] = df['signal'].copy()
    
        return df
    
    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """
        Calculate stop loss price for a position.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            
        Returns:
            float: The stop loss price.
        """
        stop_loss_pct = self.params.get('stop_loss', 0.003)  # Default to 0.3% (30 pips for EUR/USD)
        
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """
        Calculate take profit price for a position.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            
        Returns:
            float: The take profit price.
        """
        profit_target_pct = self.params.get('profit_target', 0.006)  # Default to 0.6% (60 pips for EUR/USD)
        
        if position_type == 'long':
            return entry_price * (1 + profit_target_pct)
        else:  # short
            return entry_price * (1 - profit_target_pct)
    
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Update the strategy parameters.
        
        Args:
            new_params (Dict[str, Any]): The new parameters.
        """
        self.params.update(new_params)
        logger.info(f"Updated {self.name} strategy parameters: {new_params}")
    
    def get_position_status(self) -> Dict[str, Any]:
        """
        Get the current position status.
        
        Returns:
            Dict[str, Any]: The position status.
        """
        return self.current_position or {}

    def set_position(self, position_data: Dict[str, Any]) -> None:
        """
        Set the current position.
        
        Args:
            position_data (Dict[str, Any]): The position data.
        """
        self.current_position = position_data
    
    def clear_position(self) -> None:
        """
        Clear the current position.
        """
        self.current_position = None
    
    def should_exit_position(self, current_price: float, current_time: datetime) -> Tuple[bool, str]:
        """
        Check if the current position should be exited.
        
        Args:
            current_price (float): The current market price.
            current_time (datetime): The current time.
            
        Returns:
            Tuple[bool, str]: (exit_flag, reason)
        """
        if not self.current_position:
            return False, ""
        
        entry_price = self.current_position['entry_price']
        position_type = self.current_position['type']
        entry_time = self.current_position['entry_time']
        stop_loss = self.current_position['stop_loss']
        take_profit = self.current_position['take_profit']
        
        # Check for stop loss
        if position_type == 'long' and current_price <= stop_loss:
            return True, "stop_loss"
        if position_type == 'short' and current_price >= stop_loss:
            return True, "stop_loss"
        
        # Check for take profit
        if position_type == 'long' and current_price >= take_profit:
            return True, "take_profit"
        if position_type == 'short' and current_price <= take_profit:
            return True, "take_profit"
        
        # Check for time-based exit
        max_holding_period = self.params.get('max_holding_period', 0)
        if max_holding_period > 0:
            holding_time = (current_time - entry_time).total_seconds() / 3600  # In hours
            if holding_time >= max_holding_period:
                return True, "time_exit"
        
        return False, ""
    
    def update_trailing_stop(self, current_price: float) -> None:
        """
        Update the trailing stop loss.
        
        Args:
            current_price (float): The current market price.
        """
        if not self.current_position:
            return
        
        # Check if trailing stop is enabled
        trailing_stop = self.params.get('trailing_stop', False)
        if not trailing_stop:
            return
        
        # Validate inputs
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            logger.error(f"Invalid current price for trailing stop update: {current_price}")
            return
            
        # Get position data
        entry_price = self.current_position.get('entry_price')
        position_type = self.current_position.get('type')
        current_stop = self.current_position.get('stop_loss')
        
        # Validate position data
        if (not isinstance(entry_price, (int, float)) or 
            not isinstance(current_stop, (int, float)) or
            position_type not in ['long', 'short']):
            logger.error(f"Invalid position data for trailing stop: entry_price={entry_price}, "
                         f"type={position_type}, stop_loss={current_stop}")
            return
        
        # Parameters for trailing stop
        activation_threshold = self.params.get('trailing_stop_activation', 0.002)  # 0.2% profit to activate
        trailing_distance = self.params.get('trailing_stop_distance', 0.0015)  # 0.15% trailing distance
        
        # Ensure trailing_distance is positive
        trailing_distance = max(0.0001, trailing_distance)
        
        # Calculate profit percentage
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            new_stop = current_price * (1 - trailing_distance)
            
            # Only update if in profit and new stop is higher than current stop
            if profit_pct >= activation_threshold and new_stop > current_stop:
                # Add some validation to prevent unreasonable stop loss levels
                if new_stop < current_price * 0.95:  # Ensure stop is not more than 5% away
                    new_stop = current_price * 0.95
                
                self.current_position['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop to {new_stop:.5f}")
                
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
            new_stop = current_price * (1 + trailing_distance)
            
            # Only update if in profit and new stop is lower than current stop
            if profit_pct >= activation_threshold and new_stop < current_stop:
                # Add some validation to prevent unreasonable stop loss levels
                if new_stop > current_price * 1.05:  # Ensure stop is not more than 5% away
                    new_stop = current_price * 1.05
                
                self.current_position['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop to {new_stop:.5f}")
    
    def get_signal_metadata(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Get metadata for a signal at a specific index.
        
        Args:
            df (pd.DataFrame): The dataframe with signals.
            index (int): The index of the signal.
            
        Returns:
            Dict[str, Any]: The signal metadata.
        """
        return {
            "strategy": self.name,
            "signal_strength": df['signal'].iloc[index] if 'signal' in df.columns else 0,
            "sentiment_adjusted": df['sentiment_signal'].iloc[index] if 'sentiment_signal' in df.columns else 0,
            "parameters": self.params
        }