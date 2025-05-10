"""
Moving Average Crossover Strategy
Implements a trading strategy based on moving average crossovers.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional

from strategies.base import Strategy
from config.parameters import MA_CROSSOVER_PARAMS
from utils.logger import setup_logger

logger = setup_logger("ma_crossover")

class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy class.
    """
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters. Defaults to None.
        """
        # Use default parameters if none provided
        if params is None:
            params = MA_CROSSOVER_PARAMS.copy()
        
        super().__init__("Moving Average Crossover", params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            df (pd.DataFrame): The OHLCV data.
            
        Returns:
            pd.DataFrame: The data with signals added.
        """
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Get parameters
        fast_period = self.params['fast_ma_period']
        slow_period = self.params['slow_ma_period']
        ma_type = self.params['ma_type']
        
        # Calculate moving averages
        if ma_type == 'SMA':
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        elif ma_type == 'EMA':
            df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        elif ma_type == 'WMA':
            # Weighted moving average
            weights_fast = np.arange(1, fast_period + 1)
            weights_slow = np.arange(1, slow_period + 1)
            
            df['fast_ma'] = df['close'].rolling(window=fast_period).apply(
                lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True
            )
            df['slow_ma'] = df['close'].rolling(window=slow_period).apply(
                lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True
            )
        else:
            logger.warning(f"Unknown MA type: {ma_type}, defaulting to EMA")
            df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate crossover signals
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['prev_ma_diff'] = df['ma_diff'].shift(1)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals based on crossovers
        # Buy signal: fast MA crosses above slow MA
        df.loc[(df['ma_diff'] > 0) & (df['prev_ma_diff'] <= 0), 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        df.loc[(df['ma_diff'] < 0) & (df['prev_ma_diff'] >= 0), 'signal'] = -1
        
        # Calculate signal strength based on the slope of the fast MA
        df['fast_ma_slope'] = df['fast_ma'].diff(3) / df['fast_ma'].shift(3)
        
        # Scale the signals by the slope to get signal strength
        df.loc[df['signal'] != 0, 'signal'] = df.loc[df['signal'] != 0, 'signal'] * (
            1 + df.loc[df['signal'] != 0, 'fast_ma_slope'].abs() * 10
        )
        
        # Add exit signals based on parameters
        exit_after_bars = self.params.get('exit_after_bars', 0)
        if exit_after_bars > 0:
            # For each entry signal, add an exit signal after the specified number of bars
            for i in range(len(df)):
                if df['signal'].iloc[i] != 0:
                    exit_index = min(i + exit_after_bars, len(df) - 1)
                    if df['signal'].iloc[exit_index] == 0:  # Only add exit if no other signal there
                        df.loc[df.index[exit_index], 'signal'] = -df['signal'].iloc[i]  # Opposite of entry signal
        
        # Apply sentiment filter if enabled
        if self.use_sentiment:
            symbol = "EUR/USD"  # Default symbol
            df = self.apply_sentiment_filter(df, symbol)
        
        # Calculate additional indicators for analysis
        df['ma_spread'] = (df['ma_diff'] / df['close']) * 10000  # Spread in pips
        
        # Calculate trend strength
        df['trend_strength'] = df['ma_diff'].abs() / df['slow_ma'] * 100
        
        # Add position columns
        df['position'] = 0
        current_position = 0
        
        for i in range(len(df)):
            # Update position based on signals
            if df['signal'].iloc[i] > self.params.get('signal_threshold', 0):
                current_position = 1  # Long
            elif df['signal'].iloc[i] < -self.params.get('signal_threshold', 0):
                current_position = -1  # Short
            
            # If using sentiment-adjusted signals, check those too
            if 'sentiment_signal' in df.columns:
                sentiment_signal = df['sentiment_signal'].iloc[i]
                if sentiment_signal > self.params.get('signal_threshold', 0):
                    current_position = 1  # Long
                elif sentiment_signal < -self.params.get('signal_threshold', 0):
                    current_position = -1  # Short
            
            df.loc[df.index[i], 'position'] = current_position
        
        return df
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, df: pd.DataFrame = None) -> float:
        """
        Calculate stop loss price based on strategy parameters and recent volatility.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            df (pd.DataFrame, optional): Recent price data for volatility calculation.
            
        Returns:
            float: The stop loss price.
        """
        stop_loss_pct = self.params.get('stop_loss', 0.003)  # Default to 0.3%
        
        # If we have price data, adjust stop loss based on volatility
        if df is not None and len(df) > 20:
            # Calculate ATR for volatility-based stop loss
            high = df['high'].values[-20:]
            low = df['low'].values[-20:]
            close = np.concatenate(([df['close'].values[-21]], df['close'].values[-20:-1]))
            
            # Calculate True Range
            tr1 = np.abs(high - low)
            tr2 = np.abs(high - close)
            tr3 = np.abs(low - close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range)
            
            # Convert ATR to percentage
            atr_pct = atr / entry_price
            
            # Use larger of fixed stop loss or ATR-based stop loss
            stop_loss_pct = max(stop_loss_pct, atr_pct * 1.5)
        
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: str, df: pd.DataFrame = None) -> float:
        """
        Calculate take profit price based on strategy parameters and recent volatility.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            df (pd.DataFrame, optional): Recent price data for volatility calculation.
            
        Returns:
            float: The take profit price.
        """
        profit_target_pct = self.params.get('profit_target', 0.005)  # Default to 0.5%
        
        # If we have price data, adjust profit target based on volatility
        if df is not None and len(df) > 20:
            # Calculate ATR for volatility-based profit target
            high = df['high'].values[-20:]
            low = df['low'].values[-20:]
            close = np.concatenate(([df['close'].values[-21]], df['close'].values[-20:-1]))
            
            # Calculate True Range
            tr1 = np.abs(high - low)
            tr2 = np.abs(high - close)
            tr3 = np.abs(low - close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range)
            
            # Convert ATR to percentage
            atr_pct = atr / entry_price
            
            # Use larger of fixed profit target or ATR-based profit target
            profit_target_pct = max(profit_target_pct, atr_pct * 2.5)
        
        if position_type == 'long':
            return entry_price * (1 + profit_target_pct)
        else:  # short
            return entry_price * (1 - profit_target_pct)