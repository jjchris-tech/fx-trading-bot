"""
Execution Simulator
Simulates trade execution with realistic market conditions.
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple
import random

from config.config import (
    SPREAD, COMMISSION, SLIPPAGE, EXECUTION_DELAY,
    SYMBOL, INITIAL_CAPITAL, RISK_PER_TRADE
)
from utils.logger import setup_logger
from utils.helpers import pip_difference
from execution.position_sizing import calculate_position_size
from data.market_data import MarketData
from alerts.discord import send_trade_alert

logger = setup_logger("simulator")

class ExecutionSimulator:
    """
    Simulates trade execution with realistic market conditions.
    """
    def __init__(self, 
                 symbol: str = SYMBOL,
                 initial_capital: float = INITIAL_CAPITAL,
                 risk_per_trade: float = RISK_PER_TRADE,
                 spread: float = SPREAD,
                 commission: float = COMMISSION,
                 slippage: float = SLIPPAGE,
                 execution_delay: float = EXECUTION_DELAY):
        """
        Initialize the ExecutionSimulator.
        
        Args:
            symbol (str, optional): The trading symbol. Defaults to SYMBOL from config.
            initial_capital (float, optional): The initial capital. Defaults to INITIAL_CAPITAL from config.
            risk_per_trade (float, optional): The risk per trade as a percentage. Defaults to RISK_PER_TRADE from config.
            spread (float, optional): The spread in decimal. Defaults to SPREAD from config.
            commission (float, optional): The commission in decimal. Defaults to COMMISSION from config.
            slippage (float, optional): The slippage in decimal. Defaults to SLIPPAGE from config.
            execution_delay (float, optional): The execution delay in seconds. Defaults to EXECUTION_DELAY from config.
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread = spread
        self.commission = commission
        self.slippage = slippage
        self.execution_delay = execution_delay
        
        # Position tracking
        self.current_position = None
        self.positions_history = []
        self.equity_curve = [initial_capital]
        self.trade_log = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pips = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.max_capital = initial_capital
        
        logger.info(f"ExecutionSimulator initialized with {initial_capital:.2f} capital")
    
    def calculate_entry_price(self, market_price: float, position_type: str) -> float:
        """
        Calculate the entry price including spread and slippage.
        
        Args:
            market_price (float): The market price.
            position_type (str): The position type ('long' or 'short').
            
        Returns:
            float: The entry price.
        """
        # Apply spread
        if position_type == 'long':
            price = market_price + (self.spread / 2)
        else:  # short
            price = market_price - (self.spread / 2)
        
        # Apply slippage (random in range of 0 to max slippage)
        slippage = random.uniform(0, self.slippage)
        
        if position_type == 'long':
            price += slippage
        else:  # short
            price -= slippage
        
        return price
    
    def calculate_exit_price(self, market_price: float, position_type: str) -> float:
        """
        Calculate the exit price including spread and slippage.
        
        Args:
            market_price (float): The market price.
            position_type (str): The position type ('long' or 'short').
            
        Returns:
            float: The exit price.
        """
        # Apply spread
        if position_type == 'long':
            price = market_price - (self.spread / 2)
        else:  # short
            price = market_price + (self.spread / 2)
        
        # Apply slippage (random in range of 0 to max slippage)
        slippage = random.uniform(0, self.slippage)
        
        if position_type == 'long':
            price -= slippage
        else:  # short
            price += slippage
        
        return price
    
    def enter_position(self, 
                      signal_time: datetime,
                      market_price: float,
                      position_type: str,
                      stop_loss: float, 
                      take_profit: float,
                      position_size: Optional[float] = None,
                      strategy: str = "unknown",
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enter a new position.
        
        Args:
            signal_time (datetime): The time of the signal.
            market_price (float): The current market price.
            position_type (str): The position type ('long' or 'short').
            stop_loss (float): The stop loss price.
            take_profit (float): The take profit price.
            position_size (Optional[float], optional): The position size. 
                Defaults to None (calculated based on risk).
            strategy (str, optional): The strategy name. Defaults to "unknown".
            metadata (Optional[Dict[str, Any]], optional): Additional metadata. Defaults to None.
            
        Returns:
            Dict[str, Any]: The position data.
        """
        # Check if already in a position
        if self.current_position is not None:
            logger.warning("Cannot enter position: already in a position")
            return None
        
        # Simulate execution delay
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)
        
        # Calculate entry price
        entry_price = self.calculate_entry_price(market_price, position_type)
        
        # Calculate position size if not provided
        if position_size is None:
            position_size = calculate_position_size(
                capital=self.capital,
                risk_percentage=self.risk_per_trade,
                entry_price=entry_price,
                stop_loss=stop_loss,
                symbol=self.symbol
            )
        
        # SAFETY CHECK: Maximum position size cap
        max_position_value = self.capital * 0.5  # Max 50% of capital in position value
        max_position_size = max_position_value / entry_price
        position_size = min(position_size, max_position_size)
        
        # SAFETY CHECK: Sanity limit
        if position_size > 1000000:  # 10 standard lots
            logger.warning(f"Position size suspiciously large: {position_size:.2f} units. Capping at 10 standard lots.")
            position_size = 1000000  # 10 standard lots
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Calculate commission cost
        commission_cost = position_value * self.commission
        
        # Update capital
        self.capital -= commission_cost
        
        # Create position object
        position = {
            "id": len(self.positions_history) + 1,
            "symbol": self.symbol,
            "type": position_type,
            "size": position_size,
            "entry_price": entry_price,
            "entry_time": signal_time,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "pnl": 0.0,
            "pnl_pips": 0.0,
            "commission": commission_cost,
            "strategy": strategy,
            "metadata": metadata or {}
        }
        
        # Set current position
        self.current_position = position
        
        logger.info(f"Entered {position_type} position: {position_size} units at {entry_price:.5f}")
        
        # Send Discord alert
        send_trade_alert(
            trade_type="ENTRY",
            position_type=position_type.upper(),
            symbol=self.symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            pnl=0.0,
            strategy=strategy
        )
        
        return position
    
    def exit_position(self, 
                     signal_time: datetime,
                     market_price: float,
                     reason: str = "signal") -> Dict[str, Any]:
        """
        Exit the current position.
        
        Args:
            signal_time (datetime): The time of the signal.
            market_price (float): The current market price.
            reason (str, optional): The reason for exiting. Defaults to "signal".
            
        Returns:
            Dict[str, Any]: The updated position data.
        """
        # Check if in a position
        if self.current_position is None:
            logger.warning("Cannot exit position: no position to exit")
            return None
        
        # Simulate execution delay
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)
        
        # Get position data
        position = self.current_position
        position_type = position["type"]
        entry_price = position["entry_price"]
        position_size = position["size"]
        
        # Calculate exit price
        exit_price = self.calculate_exit_price(market_price, position_type)
        
        # Calculate position value
        position_value = position_size * exit_price
        
        # Calculate commission cost
        commission_cost = position_value * self.commission
        
        # Calculate PnL
        if position_type == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size
        
        # Subtract commission costs
        pnl -= (position["commission"] + commission_cost)
        
        # Calculate pips difference
        pips = pip_difference(entry_price, exit_price, self.symbol)
        if position_type == 'short':
            pips = -pips
        
        # Update position data
        position["exit_price"] = exit_price
        position["exit_time"] = signal_time
        position["exit_reason"] = reason
        position["pnl"] = pnl
        position["pnl_pips"] = pips
        position["commission"] += commission_cost
        
        # Update capital and performance metrics
        self.capital += pnl  # Only add the profit/loss to capital, not the full position value
        
        self.total_trades += 1
        self.total_pips += pips
        self.total_profit += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update max capital and drawdown
        if self.capital > self.max_capital:
            self.max_capital = self.capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.max_capital - self.capital) / self.max_capital * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Add to positions history
        self.positions_history.append(position)
        
        # Update equity curve
        self.equity_curve.append(self.capital)
        
        # Add to trade log
        trade_log_entry = {
            "id": position["id"],
            "symbol": self.symbol,
            "type": position_type,
            "entry_time": position["entry_time"],
            "entry_price": entry_price,
            "exit_time": signal_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl": pnl,
            "pnl_pips": pips,
            "position_size": position_size,
            "strategy": position["strategy"]
        }
        self.trade_log.append(trade_log_entry)
        
        logger.info(f"Exited {position_type} position: {pnl:.2f} USD ({pips:.1f} pips)")
        
        # Send Discord alert
        send_trade_alert(
            trade_type="EXIT",
            position_type=position_type.upper(),
            symbol=self.symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl=pnl,
            pnl_pips=pips,
            strategy=position["strategy"],
            exit_reason=reason
        )
        
        # Clear current position
        self.current_position = None
        
        return position
    
    def update_position(self, current_time: datetime, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Update the current position and check for stop loss or take profit.
        
        Args:
            current_time (datetime): The current time.
            current_price (float): The current market price.
            
        Returns:
            Optional[Dict[str, Any]]: The exited position data if stop loss or take profit was hit,
                None otherwise.
        """
        # Check if in a position
        if self.current_position is None:
            return None
        
        position = self.current_position
        position_type = position["type"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Apply spread to current price
        if position_type == 'long':
            effective_price = current_price - (self.spread / 2)  # Bid price for long positions
        else:  # short
            effective_price = current_price + (self.spread / 2)  # Ask price for short positions
        
        # Check for stop loss
        if (position_type == 'long' and effective_price <= stop_loss) or \
           (position_type == 'short' and effective_price >= stop_loss):
            return self.exit_position(current_time, current_price, "stop_loss")
        
        # Check for take profit
        if (position_type == 'long' and effective_price >= take_profit) or \
           (position_type == 'short' and effective_price <= take_profit):
            return self.exit_position(current_time, current_price, "take_profit")
        
        return None
    
    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """
        Get the current position.
        
        Returns:
            Optional[Dict[str, Any]]: The current position or None if no position.
        """
        return self.current_position
    
    def get_positions_history(self) -> List[Dict[str, Any]]:
        """
        Get the positions history.
        
        Returns:
            List[Dict[str, Any]]: The positions history.
        """
        return self.positions_history
    
    def get_equity_curve(self) -> List[float]:
        """
        Get the equity curve.
        
        Returns:
            List[float]: The equity curve.
        """
        return self.equity_curve
    
    def get_trade_log(self) -> List[Dict[str, Any]]:
        """
        Get the trade log.
        
        Returns:
            List[Dict[str, Any]]: The trade log.
        """
        return self.trade_log
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of trading performance.
        
        Returns:
            Dict[str, Any]: The performance summary.
        """
        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate average trade
        avg_trade = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        total_wins = sum(p["pnl"] for p in self.positions_history if p["pnl"] > 0)
        total_losses = abs(sum(p["pnl"] for p in self.positions_history if p["pnl"] < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate return
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(self.positions_history) > 1:
            returns = [(p["pnl"] / self.initial_capital) for p in self.positions_history]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate position holding times
        holding_times = []
        for p in self.positions_history:
            if p["entry_time"] and p["exit_time"]:
                holding_time = (p["exit_time"] - p["entry_time"]).total_seconds() / 3600  # in hours
                holding_times.append(holding_time)
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Summary of exit reasons
        exit_reasons = {}
        for p in self.positions_history:
            reason = p["exit_reason"]
            if reason in exit_reasons:
                exit_reasons[reason] += 1
            else:
                exit_reasons[reason] = 1
        
        # Calculate strategy performance
        strategy_performance = {}
        for p in self.positions_history:
            strategy = p["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                    "pips": 0
                }
            
            strategy_performance[strategy]["trades"] += 1
            if p["pnl"] > 0:
                strategy_performance[strategy]["wins"] += 1
            else:
                strategy_performance[strategy]["losses"] += 1
            
            strategy_performance[strategy]["pnl"] += p["pnl"]
            strategy_performance[strategy]["pips"] += p["pnl_pips"]
        
        # Add win rates to strategy performance
        for strategy in strategy_performance:
            stats = strategy_performance[strategy]
            stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_pips": self.total_pips,
            "total_profit": self.total_profit,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "total_return_pct": total_return,
            "max_drawdown_pct": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_holding_time_hours": avg_holding_time,
            "exit_reasons": exit_reasons,
            "strategy_performance": strategy_performance
        }
    
    def simulate_backtest(self, 
                         df: pd.DataFrame, 
                         strategy, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate a backtest on historical data.
        
        Args:
            df (pd.DataFrame): The historical OHLCV data.
            strategy: The trading strategy object.
            start_date (Optional[str], optional): The start date for the backtest. 
                Defaults to None (use all data).
            end_date (Optional[str], optional): The end date for the backtest. 
                Defaults to None (use all data).
            
        Returns:
            pd.DataFrame: The backtest results.
        """
        # Reset simulator state
        self.capital = self.initial_capital
        self.current_position = None
        self.positions_history = []
        self.equity_curve = [self.initial_capital]
        self.trade_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pips = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.max_capital = self.initial_capital
        
        # Filter data by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Add some debugging info
        logger.info(f"Backtesting with {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        # Generate signals
        df = strategy.generate_signals(df)
        
        # Debug signals
        signal_count = (df['signal'] != 0).sum()
        logger.info(f"Generated {signal_count} signals")
        
        # Add equity column to track capital
        df['equity'] = self.initial_capital
        
        # Simulate trades
        current_position = None
        
        for i in range(1, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Update the equity for this bar
            df.loc[df.index[i], 'equity'] = self.capital
            
            # Check if we have an open position
            if current_position:
                # Extract stop loss and take profit levels
                stop_loss = current_position["stop_loss"]
                take_profit = current_position["take_profit"]
                
                # Validate stop loss and take profit levels
                if not isinstance(stop_loss, (int, float)) or not isinstance(take_profit, (int, float)):
                    logger.error(f"Invalid stop loss or take profit: SL={stop_loss}, TP={take_profit}")
                    # Exit at current price to prevent further issues
                    self.exit_position(current_time, current_price, "error")
                    current_position = None
                    continue
                
                # Check low and high to determine if price moved through stop loss or take profit during the bar
                low_price = df['low'].iloc[i]
                high_price = df['high'].iloc[i]
                
                # For long positions
                if current_position["type"] == 'long':
                    if low_price <= stop_loss:
                        # Stop loss hit - exit at stop loss price
                        self.exit_position(current_time, stop_loss, "stop_loss")
                        current_position = None
                    elif high_price >= take_profit:
                        # Take profit hit - exit at take profit price
                        self.exit_position(current_time, take_profit, "take_profit")
                        current_position = None
                    else:
                        # Check for exit signal
                        if df['signal'].iloc[i] < -0.3:  # More sensitive threshold
                            self.exit_position(current_time, current_price, "signal")
                            current_position = None
                
                # For short positions
                elif current_position["type"] == 'short':
                    if high_price >= stop_loss:
                        # Stop loss hit - exit at stop loss price
                        self.exit_position(current_time, stop_loss, "stop_loss")
                        current_position = None
                    elif low_price <= take_profit:
                        # Take profit hit - exit at take profit price
                        self.exit_position(current_time, take_profit, "take_profit")
                        current_position = None
                    else:
                        # Check for exit signal
                        if df['signal'].iloc[i] > 0.3:  # More sensitive threshold
                            self.exit_position(current_time, current_price, "signal")
                            current_position = None
            
            # Check for entry signals if no position
            if not current_position:
                if df['signal'].iloc[i] > 0.3:  # More sensitive threshold
                    # Calculate stop loss and take profit
                    stop_loss = strategy.calculate_stop_loss(current_price, 'long', df.iloc[:i+1])
                    take_profit = strategy.calculate_take_profit(current_price, 'long', df.iloc[:i+1])
                    
                    # Validate stop loss is below entry and take profit is above entry
                    if stop_loss >= current_price:
                        logger.error(f"Invalid stop loss for long: {stop_loss} >= {current_price}")
                        continue
                        
                    if take_profit <= current_price:
                        logger.error(f"Invalid take profit for long: {take_profit} <= {current_price}")
                        continue
                    
                    # Enter position
                    current_position = self.enter_position(
                        signal_time=current_time,
                        market_price=current_price,
                        position_type='long',
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=strategy.name,
                        metadata=strategy.get_signal_metadata(df, i)
                    )
                    
                elif df['signal'].iloc[i] < -0.3:  # More sensitive threshold
                    # Calculate stop loss and take profit
                    stop_loss = strategy.calculate_stop_loss(current_price, 'short', df.iloc[:i+1])
                    take_profit = strategy.calculate_take_profit(current_price, 'short', df.iloc[:i+1])
                    
                    # Validate stop loss is above entry and take profit is below entry
                    if stop_loss <= current_price:
                        logger.error(f"Invalid stop loss for short: {stop_loss} <= {current_price}")
                        continue
                        
                    if take_profit >= current_price:
                        logger.error(f"Invalid take profit for short: {take_profit} >= {current_price}")
                        continue
                    
                    # Enter position
                    current_position = self.enter_position(
                        signal_time=current_time,
                        market_price=current_price,
                        position_type='short',
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=strategy.name,
                        metadata=strategy.get_signal_metadata(df, i)
                    )
        
        # Close any open position at the end of the backtest
        if current_position:
            self.exit_position(df.index[-1], df['close'].iloc[-1], "end_of_backtest")
        
        # Update final equity
        df.loc[df.index[-1], 'equity'] = self.capital
        
        # Add trade markers to the dataframe
        df['trade_entry'] = None
        df['trade_exit'] = None
        df['trade_type'] = None
        
        for position in self.positions_history:
            entry_time = position['entry_time']
            exit_time = position['exit_time']
            
            # Find the nearest index in the dataframe
            entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
            exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]
            
            # Mark entry and exit points
            df.loc[df.index[entry_idx], 'trade_entry'] = position['entry_price']
            df.loc[df.index[exit_idx], 'trade_exit'] = position['exit_price']
            df.loc[df.index[entry_idx], 'trade_type'] = position['type']
        
        logger.info(f"Backtest completed: {self.total_trades} trades, {self.total_profit:.2f} USD profit")
        
        return df