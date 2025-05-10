"""
Position Sizing Module
Handles position sizing and risk management for trades.
"""
import numpy as np
from typing import Dict, List, Union, Any, Optional, Tuple

from config.config import POSITION_SIZE_TYPE, ATR_RISK_FACTOR
from utils.logger import setup_logger
from utils.helpers import calculate_atr

logger = setup_logger("position_sizing")

def calculate_position_size(capital: float,
                            risk_percentage: float,
                            entry_price: float,
                            stop_loss: float,
                            symbol: str = "EUR/USD",
                            position_size_type: str = POSITION_SIZE_TYPE,
                            atr_risk_factor: float = ATR_RISK_FACTOR,
                            atr_value: Optional[float] = None,
                            df: Optional[Any] = None) -> float:
    """
    Calculate position size based on capital, risk, and position type.
    
    Args:
        capital (float): Available capital for trading.
        risk_percentage (float): Risk percentage per trade (e.g., 0.02 for 2%).
        entry_price (float): Entry price for the trade.
        stop_loss (float): Stop loss price for the trade.
        symbol (str, optional): Trading symbol. Defaults to "EUR/USD".
        position_size_type (str, optional): Position sizing method. Defaults to POSITION_SIZE_TYPE from config.
        atr_risk_factor (float, optional): ATR risk factor. Defaults to ATR_RISK_FACTOR from config.
        atr_value (Optional[float], optional): ATR value if already calculated. Defaults to None.
        df (Optional[Any], optional): Price dataframe for ATR calculation. Defaults to None.
        
    Returns:
        float: Position size in units of the base currency.
    """
    # Calculate risk amount in account currency
    risk_amount = capital * risk_percentage
    
    # Calculate stop loss in pips
    pip_size = 0.0001  # For most forex pairs
    if "JPY" in symbol:
        pip_size = 0.01
    
    # Calculate stop loss distance in absolute terms
    if entry_price > stop_loss:  # Long position
        stop_distance = entry_price - stop_loss
    else:  # Short position
        stop_distance = stop_loss - entry_price
    
    # Ensure stop_distance is not zero to prevent division by zero
    if stop_distance <= 0:
        logger.warning(f"Invalid stop distance: {stop_distance}. Using minimum distance of 0.0001.")
        stop_distance = 0.0001  # Minimum valid stop distance (1 pip for most pairs)
    
    # Use specified position sizing method
    if position_size_type == "fixed":
        # Fixed percentage of capital
        position_size = (capital * risk_percentage) / entry_price
        
    elif position_size_type == "percent":
        # Risk a percentage of capital based on stop distance
        position_size = risk_amount / stop_distance
        
    elif position_size_type == "atr":
        # ATR-based position sizing
        if atr_value is None and df is not None:
            # Calculate ATR if not provided
            atr_value = calculate_atr(df)[-1]
        
        if atr_value and atr_value > 0:
            # Use ATR for stop distance
            stop_distance = max(stop_distance, atr_value * atr_risk_factor)
        
        # Calculate position size based on risk amount and stop distance
        position_size = risk_amount / stop_distance
    
    else:
        # Default to percent-based sizing
        logger.warning(f"Unknown position size type: {position_size_type}, defaulting to percent-based")
        position_size = risk_amount / stop_distance
    
    # Account for lot sizing in forex (standard lot = 100,000 units)
    lot_size = position_size / 100000
    
    # Round down to nearest 0.01 lot (micro lot = 1,000 units)
    lot_size = np.floor(lot_size * 100) / 100
    
    # Convert back to units
    position_size = lot_size * 100000
    
    logger.info(f"Position size calculated: {position_size:.2f} units ({lot_size:.2f} lots)")
    
    return position_size

def calculate_optimal_take_profit(entry_price: float,
                                 stop_loss: float,
                                 position_type: str,
                                 min_risk_reward: float = 2.0) -> float:
    """
    Calculate optimal take profit level based on risk-reward ratio.
    
    Args:
        entry_price (float): Entry price for the trade.
        stop_loss (float): Stop loss price for the trade.
        position_type (str): Position type ('long' or 'short').
        min_risk_reward (float, optional): Minimum risk-reward ratio. Defaults to 2.0.
        
    Returns:
        float: Calculated take profit price.
    """
    # Calculate stop loss distance
    if position_type == 'long':
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * min_risk_reward)
    else:  # short
        stop_distance = stop_loss - entry_price
        take_profit = entry_price - (stop_distance * min_risk_reward)
    
    return take_profit

def calculate_risk_per_trade(capital: float,
                            max_drawdown: float = 0.2,
                            win_rate: float = 0.5,
                            risk_reward: float = 2.0,
                            consecutive_losses: int = 5) -> float:
    """
    Calculate optimal risk percentage per trade based on expected statistics.
    
    Args:
        capital (float): Trading capital.
        max_drawdown (float, optional): Maximum acceptable drawdown. Defaults to 0.2 (20%).
        win_rate (float, optional): Expected win rate. Defaults to 0.5 (50%).
        risk_reward (float, optional): Expected risk-reward ratio. Defaults to 2.0.
        consecutive_losses (int, optional): Number of consecutive losses to withstand. Defaults to 5.
        
    Returns:
        float: Recommended risk percentage per trade.
    """
    # Calculate expected value per trade
    expected_value = (win_rate * risk_reward) - (1 - win_rate)
    
    # Calculate Kelly criterion (optimal risk percentage)
    kelly = (expected_value) / (risk_reward)
    
    # Calculate maximum risk based on consecutive losses
    max_loss_risk = 1 - ((1 - max_drawdown) ** (1 / consecutive_losses))
    
    # Use the more conservative of the two
    optimal_risk = min(kelly, max_loss_risk)
    
    # Cap the risk at 5% as a safety measure
    optimal_risk = min(optimal_risk, 0.05)
    
    logger.info(f"Calculated optimal risk per trade: {optimal_risk:.2%}")
    
    return optimal_risk

def calculate_max_positions(capital: float,
                           risk_per_trade: float,
                           correlation_factor: float = 0.8) -> int:
    """
    Calculate maximum number of simultaneous positions based on risk.
    
    Args:
        capital (float): Trading capital.
        risk_per_trade (float): Risk percentage per trade.
        correlation_factor (float, optional): Correlation factor between trades. Defaults to 0.8.
        
    Returns:
        int: Maximum number of positions.
    """
    # More correlated trades (higher factor) means fewer positions
    max_positions = int(1 / (risk_per_trade * correlation_factor))
    
    # Ensure at least 1 position
    max_positions = max(1, max_positions)
    
    logger.info(f"Calculated maximum positions: {max_positions}")
    
    return max_positions

def calculate_martingale_size(base_size: float,
                             consecutive_losses: int,
                             factor: float = 1.5,
                             max_multiplier: float = 4.0) -> float:
    """
    Calculate position size using a martingale strategy.
    WARNING: Martingale strategies can lead to rapid capital depletion.
    Use with extreme caution.
    
    Args:
        base_size (float): Base position size.
        consecutive_losses (int): Number of consecutive losses.
        factor (float, optional): Multiplier factor. Defaults to 1.5.
        max_multiplier (float, optional): Maximum multiplier. Defaults to 4.0.
        
    Returns:
        float: Calculated position size.
    """
    # Calculate multiplier
    multiplier = factor ** consecutive_losses
    
    # Cap the multiplier
    multiplier = min(multiplier, max_multiplier)
    
    # Calculate new position size
    new_size = base_size * multiplier
    
    logger.warning(f"Using martingale sizing after {consecutive_losses} losses: {multiplier:.2f}x multiplier")
    
    return new_size

def calculate_anti_martingale_size(base_size: float,
                                  consecutive_wins: int,
                                  factor: float = 1.3,
                                  max_multiplier: float = 3.0) -> float:
    """
    Calculate position size using an anti-martingale strategy.
    
    Args:
        base_size (float): Base position size.
        consecutive_wins (int): Number of consecutive wins.
        factor (float, optional): Multiplier factor. Defaults to 1.3.
        max_multiplier (float, optional): Maximum multiplier. Defaults to 3.0.
        
    Returns:
        float: Calculated position size.
    """
    # Calculate multiplier
    multiplier = factor ** consecutive_wins
    
    # Cap the multiplier
    multiplier = min(multiplier, max_multiplier)
    
    # Calculate new position size
    new_size = base_size * multiplier
    
    logger.info(f"Using anti-martingale sizing after {consecutive_wins} wins: {multiplier:.2f}x multiplier")
    
    return new_size

def adjust_position_for_volatility(base_size: float,
                                  current_atr: float,
                                  average_atr: float) -> float:
    """
    Adjust position size based on current volatility.
    
    Args:
        base_size (float): Base position size.
        current_atr (float): Current ATR value.
        average_atr (float): Average ATR value.
        
    Returns:
        float: Adjusted position size.
    """
    # Calculate volatility ratio
    volatility_ratio = average_atr / current_atr if current_atr > 0 else 1.0
    
    # Adjust position size (more volatility = smaller position)
    adjusted_size = base_size * volatility_ratio
    
    # Ensure position size doesn't go below 20% of base size
    adjusted_size = max(adjusted_size, base_size * 0.2)
    
    logger.info(f"Volatility adjustment: {volatility_ratio:.2f}x -> {adjusted_size:.2f} units")
    
    return adjusted_size

def adjust_position_for_trend_strength(base_size: float,
                                      trend_strength: float,
                                      max_adjustment: float = 1.5) -> float:
    """
    Adjust position size based on trend strength.
    
    Args:
        base_size (float): Base position size.
        trend_strength (float): Trend strength indicator (0 to 1).
        max_adjustment (float, optional): Maximum adjustment factor. Defaults to 1.5.
        
    Returns:
        float: Adjusted position size.
    """
    # Calculate adjustment factor
    adjustment = 1.0 + (trend_strength * (max_adjustment - 1.0))
    
    # Apply adjustment
    adjusted_size = base_size * adjustment
    
    logger.info(f"Trend strength adjustment: {adjustment:.2f}x -> {adjusted_size:.2f} units")
    
    return adjusted_size

def position_size_by_confidence(capital: float,
                               risk_percentage: float,
                               confidence: float,
                               entry_price: float,
                               stop_loss: float,
                               min_confidence_factor: float = 0.5,
                               max_confidence_factor: float = 1.5) -> float:
    """
    Calculate position size adjusted by signal confidence.
    
    Args:
        capital (float): Trading capital.
        risk_percentage (float): Base risk percentage.
        confidence (float): Signal confidence (0 to 1).
        entry_price (float): Entry price for the trade.
        stop_loss (float): Stop loss price for the trade.
        min_confidence_factor (float, optional): Minimum confidence factor. Defaults to 0.5.
        max_confidence_factor (float, optional): Maximum confidence factor. Defaults to 1.5.
        
    Returns:
        float: Position size adjusted by confidence.
    """
    # Calculate confidence factor
    confidence_factor = min_confidence_factor + confidence * (max_confidence_factor - min_confidence_factor)
    
    # Adjust risk percentage
    adjusted_risk = risk_percentage * confidence_factor
    
    # Calculate base position size
    position_size = calculate_position_size(capital, adjusted_risk, entry_price, stop_loss)
    
    logger.info(f"Confidence-adjusted position: {confidence:.2f} confidence -> {confidence_factor:.2f}x factor")
    
    return position_size