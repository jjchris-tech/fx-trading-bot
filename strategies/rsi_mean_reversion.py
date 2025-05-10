"""
RSI Mean Reversion Strategy
Implements a trading strategy based on RSI and price's reversion to the mean.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional

from strategies.base import Strategy
from config.parameters import RSI_MEAN_REVERSION_PARAMS
from utils.logger import setup_logger

logger = setup_logger("rsi_mean_reversion")

class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion strategy class.
    """
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the RSI Mean Reversion strategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters. Defaults to None.
        """
        # Use default parameters if none provided
        if params is None:
            params = RSI_MEAN_REVERSION_PARAMS.copy()
        
        super().__init__("RSI Mean Reversion", params)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): The price series.
            period (int, optional): The RSI period. Defaults to 14.
            
        Returns:
            pd.Series: The RSI values.
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI with protection against division by zero
        rs = pd.Series(index=prices.index, dtype=float)
        for i in range(len(avg_gain)):
            if avg_loss.iloc[i] == 0:
                rs.iloc[i] = 100.0  # If no losses, RS is effectively "infinite"
            else:
                rs.iloc[i] = avg_gain.iloc[i] / avg_loss.iloc[i]
        
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI mean reversion.
        
        Args:
            df (pd.DataFrame): The OHLCV data.
            
        Returns:
            pd.DataFrame: The data with signals added.
        """
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Get parameters
        rsi_period = self.params['rsi_period']
        overbought = self.params['overbought_threshold']
        oversold = self.params['oversold_threshold']
        mean_period = self.params['mean_period']
        counter_trend_factor = self.params['counter_trend_factor']
        exit_rsi = self.params['exit_rsi_level']
        confirmation_candles = self.params.get('confirmation_candles', 1)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=rsi_period)
        
        # Calculate moving average for mean reversion
        df['mean'] = df['close'].rolling(window=mean_period).mean()
        
        # Calculate percentage deviation from mean
        df['deviation'] = (df['close'] - df['mean']) / df['mean'] * 100
        
        # Initialize signal column
        df['signal'] = 0
        
        # Initialize confirmation counters
        oversold_count = 0
        overbought_count = 0
        
        # Generate signals based on RSI levels and mean reversion
        for i in range(1, len(df)):
            current_rsi = df['rsi'].iloc[i]
            current_deviation = df['deviation'].iloc[i]
            
            # Check for oversold condition
            if current_rsi <= oversold:
                oversold_count += 1
                overbought_count = 0
            # Check for overbought condition
            elif current_rsi >= overbought:
                overbought_count += 1
                oversold_count = 0
            else:
                # Reset counters if RSI is in the normal range
                oversold_count = 0
                overbought_count = 0
            
            # Generate buy signal: RSI is oversold and price is below mean
            if (oversold_count >= confirmation_candles and 
                df['close'].iloc[i] < df['mean'].iloc[i] and
                current_deviation < -counter_trend_factor):
                df.loc[df.index[i], 'signal'] = 1
                
                # Calculate signal strength based on RSI and deviation
                signal_strength = (oversold - current_rsi) / oversold * abs(current_deviation) / counter_trend_factor
                df.loc[df.index[i], 'signal'] = 1 + signal_strength
                
                # Reset counter after signal
                oversold_count = 0
            
            # Generate sell signal: RSI is overbought and price is above mean
            elif (overbought_count >= confirmation_candles and 
                  df['close'].iloc[i] > df['mean'].iloc[i] and
                  current_deviation > counter_trend_factor):
                df.loc[df.index[i], 'signal'] = -1
                
                # Calculate signal strength based on RSI and deviation
                signal_strength = (current_rsi - overbought) / (100 - overbought) * abs(current_deviation) / counter_trend_factor
                df.loc[df.index[i], 'signal'] = -1 - signal_strength
                
                # Reset counter after signal
                overbought_count = 0
            
            # Generate exit signals based on RSI level
            if df['position'].iloc[i-1] == 1 and current_rsi >= exit_rsi:  # Exit long position
                df.loc[df.index[i], 'signal'] = -0.5  # Use smaller value to indicate exit
            elif df['position'].iloc[i-1] == -1 and current_rsi <= exit_rsi:  # Exit short position
                df.loc[df.index[i], 'signal'] = 0.5  # Use smaller value to indicate exit
        
        # Apply sentiment filter if enabled
        if self.use_sentiment:
            symbol = "EUR/USD"  # Default symbol
            df = self.apply_sentiment_filter(df, symbol)
        
        # Add position columns
        df['position'] = 0
        current_position = 0
        
        for i in range(len(df)):
            # Update position based on signals
            if df['signal'].iloc[i] > 0.5:
                current_position = 1  # Long
            elif df['signal'].iloc[i] < -0.5:
                current_position = -1  # Short
            elif df['signal'].iloc[i] != 0:
                # Exit signals (values between -0.5 and 0.5)
                current_position = 0
            
            # If using sentiment-adjusted signals, check those too
            if 'sentiment_signal' in df.columns:
                sentiment_signal = df['sentiment_signal'].iloc[i]
                if sentiment_signal > 0.5:
                    current_position = 1  # Long
                elif sentiment_signal < -0.5:
                    current_position = -1  # Short
            
            df.loc[df.index[i], 'position'] = current_position
        
        # Calculate additional indicators for analysis
        df['rsi_slope'] = df['rsi'].diff(3)
        df['mean_gap'] = df['close'] - df['mean']
        
        # Identify divergences (price makes new high/low but RSI doesn't)
        df['price_higher_high'] = False
        df['price_lower_low'] = False
        df['rsi_higher_high'] = False
        df['rsi_lower_low'] = False
        
        # Look for divergences in a 10-bar window
        window = 10
        for i in range(window, len(df)):
            # Check if price made a higher high
            if df['high'].iloc[i] > df['high'].iloc[i-window:i].max():
                df.loc[df.index[i], 'price_higher_high'] = True
            
            # Check if price made a lower low
            if df['low'].iloc[i] < df['low'].iloc[i-window:i].min():
                df.loc[df.index[i], 'price_lower_low'] = True
            
            # Check if RSI made a higher high
            if df['rsi'].iloc[i] > df['rsi'].iloc[i-window:i].max():
                df.loc[df.index[i], 'rsi_higher_high'] = True
            
            # Check if RSI made a lower low
            if df['rsi'].iloc[i] < df['rsi'].iloc[i-window:i].min():
                df.loc[df.index[i], 'rsi_lower_low'] = True
        
        # Identify bearish divergence (price higher high but RSI not higher high)
        df['bearish_divergence'] = df['price_higher_high'] & ~df['rsi_higher_high']
        
        # Identify bullish divergence (price lower low but RSI not lower low)
        df['bullish_divergence'] = df['price_lower_low'] & ~df['rsi_lower_low']
        
        # Strengthen signals on divergence
        for i in range(len(df)):
            if df['signal'].iloc[i] < 0 and df['bearish_divergence'].iloc[i]:
                df.loc[df.index[i], 'signal'] = df['signal'].iloc[i] * 1.5
            elif df['signal'].iloc[i] > 0 and df['bullish_divergence'].iloc[i]:
                df.loc[df.index[i], 'signal'] = df['signal'].iloc[i] * 1.5
        
        return df
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, df: pd.DataFrame = None) -> float:
        """
        Calculate stop loss price based on strategy parameters.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            df (pd.DataFrame, optional): Recent price data for additional analysis.
            
        Returns:
            float: The stop loss price.
        """
        stop_loss_pct = self.params.get('stop_loss', 0.002)  # Default to 0.2%
        
        # If we have price data, calculate stop loss based on recent volatility
        if df is not None and len(df) > 20:
            # Use the mean as a reference for stop loss
            mean = df['mean'].iloc[-1]
            
            if position_type == 'long':
                # Stop below the mean or a percentage below entry, whichever is higher
                mean_stop = mean * 0.9995  # Small buffer below mean
                pct_stop = entry_price * (1 - stop_loss_pct)
                return max(mean_stop, pct_stop)
            else:  # short
                # Stop above the mean or a percentage above entry, whichever is lower
                mean_stop = mean * 1.0005  # Small buffer above mean
                pct_stop = entry_price * (1 + stop_loss_pct)
                return min(mean_stop, pct_stop)
        
        # Default calculation if no price data available
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)