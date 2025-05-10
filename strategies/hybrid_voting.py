"""
Hybrid Voting Strategy
Combines multiple strategies using a weighted voting system.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional

from strategies.base import Strategy
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from config.parameters import HYBRID_VOTING_PARAMS
from utils.logger import setup_logger
from utils.helpers import calculate_atr

logger = setup_logger("hybrid_voting")

class HybridVotingStrategy(Strategy):
    """
    Hybrid Voting Strategy combining multiple strategies.
    """
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Hybrid Voting Strategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters. Defaults to None.
        """
        # Use default parameters if none provided
        if params is None:
            params = HYBRID_VOTING_PARAMS.copy()
        
        super().__init__("Hybrid Voting", params)
        
        # Initialize component strategies
        self.strategies = {
            "ma_crossover": MovingAverageCrossover(),
            "rsi_mean_reversion": RSIMeanReversion()
        }
    
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Update the strategy parameters and propagate to component strategies.
        
        Args:
            new_params (Dict[str, Any]): The new parameters.
        """
        super().update_parameters(new_params)
        
        # Update component strategies if their parameters are included
        for strategy_name, strategy in self.strategies.items():
            strategy_prefix = f"{strategy_name}_"
            strategy_params = {}
            
            # Extract parameters for this strategy
            for key, value in new_params.items():
                if key.startswith(strategy_prefix):
                    param_name = key[len(strategy_prefix):]
                    strategy_params[param_name] = value
            
            # Update strategy parameters if any were found
            if strategy_params:
                strategy.update_parameters(strategy_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by combining multiple strategies.
    
        Args:
            df (pd.DataFrame): The OHLCV data.
        
        Returns:
            pd.DataFrame: The data with signals added.
        """
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
    
        # Initialize position column
        df['position'] = 0
    
        # Make sure signal column is a float type to avoid type errors
        df['signal'] = 0.0
    
        # Apply each strategy and get signals
        strategy_signals = {}
    
        for name, strategy in self.strategies.items():
            # Generate signals for this strategy
            strategy_df = strategy.generate_signals(df)
        
            # Extract signal and position columns
            strategy_signals[f"{name}_signal"] = strategy_df['signal']
            strategy_signals[f"{name}_position"] = strategy_df['position']
        
            # Copy additional columns for analysis
            for col in strategy_df.columns:
                if col not in df.columns and col not in ['signal', 'position']:
                    df[f"{name}_{col}"] = strategy_df[col]
    
        # Add strategy signal columns to the dataframe
        for col_name, values in strategy_signals.items():
            df[col_name] = values
    
        # Get strategy weights
        ma_weight = self.params.get('ma_crossover_weight', 1.0)
        rsi_weight = self.params.get('rsi_mean_reversion_weight', 1.0)
    
        # Calculate weighted vote
        df['vote'] = (
            df['ma_crossover_signal'] * ma_weight +
            df['rsi_mean_reversion_signal'] * rsi_weight
        ) / (ma_weight + rsi_weight)
    
        # Apply voting threshold to generate signals - IMPROVED THRESHOLD
        voting_threshold = self.params.get('voting_threshold', 0.5)
    
        # Generate buy signals - fix dtype incompatibility
        buy_mask = df['vote'] > voting_threshold
        if buy_mask.any():
            df.loc[buy_mask, 'signal'] = df.loc[buy_mask, 'vote'].astype(float)
    
        # Generate sell signals - fix dtype incompatibility
        sell_mask = df['vote'] < -voting_threshold
        if sell_mask.any():
            df.loc[sell_mask, 'signal'] = df.loc[sell_mask, 'vote'].astype(float)
    
        # Add trend filter
        self._add_trend_filter(df)
    
        # Apply volatility filter 
        self._apply_volatility_filter(df)
    
        # Apply sentiment filter if enabled
        try:
            if self.use_sentiment:
                symbol = "EUR/USD"  # Default symbol
                df = self.apply_sentiment_filter(df, symbol)
        except Exception as e:
            logger.error(f"Error in sentiment filter: {e}")
            # Ensure sentiment_signal exists 
            if 'sentiment_signal' not in df.columns:
                df['sentiment_signal'] = df['signal'].copy()
    
        # Calculate positions based on signals with improved logic
        self._calculate_positions(df)
    
        # Add additional analysis columns
        self._add_analysis_columns(df)
    
        return df
    
    def _add_trend_filter(self, df: pd.DataFrame) -> None:
        """
        Add a trend filter based on moving averages
        """
        # Calculate 50-period and 200-period EMAs for trend direction
        df['trend_ma50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['trend_ma200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Determine trend direction
        df['trend_direction'] = np.where(df['trend_ma50'] > df['trend_ma200'], 1, 
                                        np.where(df['trend_ma50'] < df['trend_ma200'], -1, 0))
        
        # Filter signals against the trend - only take signals in the direction of the trend
        for i in range(len(df)):
            if df['signal'].iloc[i] > 0 and df['trend_direction'].iloc[i] < 0:
                df.loc[df.index[i], 'signal'] = 0  # Cancel buy signal in downtrend
            elif df['signal'].iloc[i] < 0 and df['trend_direction'].iloc[i] > 0:
                df.loc[df.index[i], 'signal'] = 0  # Cancel sell signal in uptrend
    
    def _apply_volatility_filter(self, df: pd.DataFrame) -> None:
        """
        Apply volatility filter to signals
        """
        atr_period = self.params.get('atr_filter_period', 14)
        atr_multiplier = self.params.get('atr_filter_multiplier', 0.7)
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, period=atr_period)
        
        # Calculate ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Calculate historical ATR percentile
        df['atr_percentile'] = df['atr_pct'].rolling(window=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Filter out signals during extreme volatility (too high or too low)
        high_volatility = df['atr_percentile'] > 0.8
        low_volatility = df['atr_percentile'] < 0.2
        
        # Cancel signals during extreme volatility
        df.loc[high_volatility | low_volatility, 'signal'] = 0
        
        # Strengthen signals during medium volatility (more ideal conditions)
        medium_volatility = (df['atr_percentile'] > 0.4) & (df['atr_percentile'] <= 0.6)
        df.loc[medium_volatility & (df['signal'] != 0), 'signal'] = df.loc[medium_volatility & (df['signal'] != 0), 'signal'] * 1.2
    
    def _calculate_positions(self, df: pd.DataFrame) -> None:
        """
        Calculate positions based on signals with improved logic
        """
        current_position = 0
        
        # Improved parameters for position management
        min_bars_between_trades = 3  # Min bars to wait before new trade
        last_signal_bar = -min_bars_between_trades - 1
        
        # Signal threshold with hysteresis (different thresholds for entry and exit)
        entry_threshold = 0.5  # Higher threshold for entry
        exit_threshold = 0.3   # Lower threshold for exit
        
        for i in range(len(df)):
            # Check for exit first
            if current_position == 1 and df['signal'].iloc[i] < -exit_threshold:  # Exit long
                current_position = 0
                last_signal_bar = i
            elif current_position == -1 and df['signal'].iloc[i] > exit_threshold:  # Exit short
                current_position = 0
                last_signal_bar = i
            # Then check for entry, but only if enough time has passed since last trade
            elif current_position == 0 and (i - last_signal_bar) > min_bars_between_trades:
                if df['signal'].iloc[i] > entry_threshold:  # Long entry
                    current_position = 1
                    last_signal_bar = i
                elif df['signal'].iloc[i] < -entry_threshold:  # Short entry
                    current_position = -1
                    last_signal_bar = i
            
            # If using sentiment-adjusted signals, check those too
            if 'sentiment_signal' in df.columns and current_position == 0:
                sentiment_signal = df['sentiment_signal'].iloc[i]
                if sentiment_signal > entry_threshold:
                    current_position = 1
                elif sentiment_signal < -entry_threshold:
                    current_position = -1
            
            df.loc[df.index[i], 'position'] = current_position
    
    def _add_analysis_columns(self, df: pd.DataFrame) -> None:
        """
        Add additional columns for analysis
        """
        # Calculate agreement between strategies
        df['strategy_agreement'] = (
            (df['ma_crossover_position'] == df['rsi_mean_reversion_position']) & 
            (df['ma_crossover_position'] != 0)
        ).astype(int)
        
        # Calculate conviction score
        df['conviction'] = abs(df['vote'])
        
        # Calculate confirmation level - IMPROVED CONFIRMATION
        minimum_confirmation = self.params.get('minimum_confirmation', 0.5)  # Increased from 0.3 to 0.5
        df['confirmation'] = 0.0
        
        for i in range(1, len(df)):
            if df['signal'].iloc[i] != 0:
                # Check how many previous bars support the signal
                prev_signals = df['vote'].iloc[max(0, i-5):i]
                same_direction = (prev_signals * df['signal'].iloc[i] > 0).mean()
                
                if same_direction >= minimum_confirmation:
                    df.loc[df.index[i], 'confirmation'] = same_direction
                else:
                    # Weak confirmation, reduce signal strength significantly
                    df.loc[df.index[i], 'signal'] = df['signal'].iloc[i] * 0.3
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, df: pd.DataFrame = None) -> float:
        """
        Calculate stop loss price based on ATR and key levels.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            df (pd.DataFrame, optional): Recent price data for ATR calculation.
            
        Returns:
            float: The stop loss price.
        """
        # Default stop loss from parameters - IMPROVED RISK MANAGEMENT
        stop_loss_pct = self.params.get('stop_loss', 0.003)  # Default 0.3% (30 pips for EUR/USD)
        
        # If we have price data, calculate ATR-based stop loss
        if df is not None and len(df) > 20:
            # Calculate ATR if needed
            if 'atr' not in df.columns:
                atr = calculate_atr(df)[-1]
            else:
                atr = df['atr'].iloc[-1]
            
            # Stop loss based on ATR (2.5 x ATR) - WIDENED STOPS
            atr_multiplier = 2.5  # Increased from 2.0 to 2.5
            
            if position_type == 'long':
                atr_stop = entry_price - (atr * atr_multiplier)
                pct_stop = entry_price * (1 - stop_loss_pct)
                
                # Use the higher of the two (smaller distance from entry)
                return max(atr_stop, pct_stop)
            else:  # short
                atr_stop = entry_price + (atr * atr_multiplier)
                pct_stop = entry_price * (1 + stop_loss_pct)
                
                # Use the lower of the two (smaller distance from entry)
                return min(atr_stop, pct_stop)
        
        # Default calculation if no price data available
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: str, df: pd.DataFrame = None) -> float:
        """
        Calculate take profit price based on ATR and risk-reward ratio.
        
        Args:
            entry_price (float): The entry price.
            position_type (str): The position type ('long' or 'short').
            df (pd.DataFrame, optional): Recent price data for ATR calculation.
            
        Returns:
            float: The take profit price.
        """
        # Default take profit from parameters - INCREASED FOR BETTER REWARD
        profit_target_pct = self.params.get('profit_target', 0.008)  # Increased from 0.005 to 0.008
        
        # If we have price data, calculate ATR-based take profit
        if df is not None and len(df) > 20:
            # Calculate ATR if needed
            if 'atr' not in df.columns:
                atr = calculate_atr(df)[-1]
            else:
                atr = df['atr'].iloc[-1]
                
            # Take profit based on ATR - IMPROVED RISK/REWARD
            atr_multiplier = 4.0  # Increased from 3.0 to 4.0 for better risk/reward
            
            if position_type == 'long':
                return entry_price + (atr * atr_multiplier)
            else:  # short
                return entry_price - (atr * atr_multiplier)
        
        # Default calculation if no price data available
        if position_type == 'long':
            return entry_price * (1 + profit_target_pct)
        else:  # short
            return entry_price * (1 - profit_target_pct)