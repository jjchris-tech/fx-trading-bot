"""
Performance Metrics Module
Calculates and tracks various trading performance metrics.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
import json
import os
from pathlib import Path

from config.config import INITIAL_CAPITAL, DATA_DIR, REPORTS_DIR
from utils.logger import setup_logger
from utils.helpers import (
    calculate_drawdown, calculate_sharpe_ratio, 
    calculate_sortino_ratio, calculate_profit_factor
)

logger = setup_logger("metrics")

class PerformanceMetrics:
    """
    Calculates and tracks various trading performance metrics.
    """
    def __init__(self, 
                 initial_capital: float = INITIAL_CAPITAL,
                 symbol: str = "EUR/USD",
                 trade_log: Optional[List[Dict[str, Any]]] = None,
                 equity_curve: Optional[List[float]] = None):
        """
        Initialize the PerformanceMetrics class.
        
        Args:
            initial_capital (float, optional): The initial capital. Defaults to INITIAL_CAPITAL from config.
            symbol (str, optional): The trading symbol. Defaults to "EUR/USD".
            trade_log (Optional[List[Dict[str, Any]]], optional): The trade log. Defaults to None.
            equity_curve (Optional[List[float]], optional): The equity curve. Defaults to None.
        """
        self.initial_capital = initial_capital
        self.symbol = symbol
        self.trade_log = trade_log or []
        self.equity_curve = equity_curve or [initial_capital]
        
        # Metrics storage
        self.metrics = {}
        
        # Metrics history
        self.metrics_history = []
        
        # Storage directory
        self.metrics_dir = Path(REPORTS_DIR) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def update_trade_log(self, trade_log: List[Dict[str, Any]]) -> None:
        """
        Update the trade log.
        
        Args:
            trade_log (List[Dict[str, Any]]): The new trade log.
        """
        self.trade_log = trade_log
    
    def update_equity_curve(self, equity_curve: List[float]) -> None:
        """
        Update the equity curve.
        
        Args:
            equity_curve (List[float]): The new equity curve.
        """
        self.equity_curve = equity_curve
    
    def calculate_metrics(self, 
                        recalculate: bool = True, 
                        period_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            recalculate (bool, optional): Whether to recalculate metrics. Defaults to True.
            period_days (Optional[int], optional): Number of days to calculate metrics for. 
                Defaults to None (all data).
                
        Returns:
            Dict[str, Any]: The calculated metrics.
        """
        if not recalculate and self.metrics:
            return self.metrics
        
        # Filter trade log by period if specified
        filtered_trade_log = self.trade_log
        
        if period_days is not None and self.trade_log:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            filtered_trade_log = [
                trade for trade in self.trade_log
                if isinstance(trade.get("exit_time"), datetime) and trade["exit_time"] >= cutoff_date
            ]
        
        # Calculate metrics based on trade log
        metrics = self._calculate_metrics_from_trades(filtered_trade_log)
        
        # Calculate metrics based on equity curve
        equity_metrics = self._calculate_metrics_from_equity()
        metrics.update(equity_metrics)
        
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Store metrics
        self.metrics = metrics
        
        # Add to history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_metrics_from_trades(self, trade_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on the trade log.
        
        Args:
            trade_log (List[Dict[str, Any]]): The trade log.
            
        Returns:
            Dict[str, Any]: The calculated metrics.
        """
        metrics = {
            "total_trades": len(trade_log),
            "initial_capital": self.initial_capital,
            "current_capital": self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }
        
        if not trade_log:
            # Return default metrics if no trades
            metrics.update({
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "total_profit": 0,
                "avg_trade": 0,
                "profit_factor": 0,
                "total_return_pct": 0,
                "avg_holding_time_hours": 0,
            })
            return metrics
        
        # Calculate basic metrics
        winning_trades = sum(1 for trade in trade_log if trade.get("pnl", 0) > 0)
        losing_trades = sum(1 for trade in trade_log if trade.get("pnl", 0) < 0)
        
        win_rate = (winning_trades / len(trade_log) * 100) if trade_log else 0
        
        total_pips = sum(trade.get("pnl_pips", 0) for trade in trade_log)
        total_profit = sum(trade.get("pnl", 0) for trade in trade_log)
        
        avg_trade = total_profit / len(trade_log) if trade_log else 0
        
        # Calculate profit factor
        total_wins = sum(trade.get("pnl", 0) for trade in trade_log if trade.get("pnl", 0) > 0)
        total_losses = abs(sum(trade.get("pnl", 0) for trade in trade_log if trade.get("pnl", 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate return
        total_return = (metrics["current_capital"] - self.initial_capital) / self.initial_capital * 100
        
        # Calculate holding times
        holding_times = []
        
        for trade in trade_log:
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            
            if entry_time and exit_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                
                holding_time = (exit_time - entry_time).total_seconds() / 3600  # in hours
                holding_times.append(holding_time)
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Calculate strategy performance
        strategy_performance = {}
        
        for trade in trade_log:
            strategy = trade.get("strategy", "unknown")
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                    "pips": 0
                }
            
            strategy_performance[strategy]["trades"] += 1
            
            if trade.get("pnl", 0) > 0:
                strategy_performance[strategy]["wins"] += 1
            elif trade.get("pnl", 0) < 0:
                strategy_performance[strategy]["losses"] += 1
            
            strategy_performance[strategy]["pnl"] += trade.get("pnl", 0)
            strategy_performance[strategy]["pips"] += trade.get("pnl_pips", 0)
        
        # Add win rates to strategy performance
        for strategy in strategy_performance:
            stats = strategy_performance[strategy]
            stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        
        # Update metrics
        metrics.update({
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pips": total_pips,
            "total_profit": total_profit,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "total_return_pct": total_return,
            "avg_holding_time_hours": avg_holding_time,
            "strategy_performance": strategy_performance
        })
        
        return metrics
    
    def _calculate_metrics_from_equity(self) -> Dict[str, Any]:
        """
        Calculate metrics based on the equity curve.
        
        Returns:
            Dict[str, Any]: The calculated metrics.
        """
        metrics = {}
        
        if not self.equity_curve or len(self.equity_curve) < 2:
            metrics.update({
                "max_drawdown_pct": 0,
                "max_drawdown_duration": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "volatility": 0
            })
            return metrics
        
        # Convert equity curve to numpy array
        equity_array = np.array(self.equity_curve)
        
        # Calculate drawdown
        max_drawdown, max_drawdown_duration = calculate_drawdown(equity_array)
        
        # Calculate returns
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate metrics
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        volatility = np.std(returns) * 100  # Convert to percentage
        
        # Update metrics
        metrics.update({
            "max_drawdown_pct": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "volatility": volatility
        })
        
        return metrics
    
    def calculate_drawdown_profile(self) -> List[Dict[str, Any]]:
        """
        Calculate a detailed drawdown profile.
        
        Returns:
            List[Dict[str, Any]]: The drawdown profile.
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return []
        
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        # Find drawdown periods
        in_drawdown = False
        drawdown_periods = []
        current_period = None
        
        for i in range(len(drawdown)):
            if drawdown[i] < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                current_period = {
                    "start_index": i,
                    "start_equity": equity_array[i],
                    "peak_equity": running_max[i],
                    "current_drawdown": drawdown[i]
                }
            elif drawdown[i] < 0 and in_drawdown:
                # Continuing drawdown
                if drawdown[i] < current_period["current_drawdown"]:
                    # New max drawdown in this period
                    current_period["current_drawdown"] = drawdown[i]
                    current_period["min_equity"] = equity_array[i]
                    current_period["min_equity_index"] = i
            elif drawdown[i] == 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                current_period["end_index"] = i
                current_period["end_equity"] = equity_array[i]
                current_period["duration"] = i - current_period["start_index"]
                current_period["recovery_duration"] = i - current_period.get("min_equity_index", current_period["start_index"])
                
                # Add to periods
                drawdown_periods.append(current_period)
                current_period = None
        
        # Check if still in drawdown at the end
        if in_drawdown:
            current_period["end_index"] = len(drawdown) - 1
            current_period["end_equity"] = equity_array[-1]
            current_period["duration"] = current_period["end_index"] - current_period["start_index"]
            current_period["recovery_duration"] = 0  # Still recovering
            
            # Add to periods
            drawdown_periods.append(current_period)
        
        return drawdown_periods
    
    def calculate_monthly_returns(self) -> Dict[str, float]:
        """
        Calculate monthly returns.
        
        Returns:
            Dict[str, float]: The monthly returns.
        """
        if not self.trade_log:
            return {}
        
        # Convert trade log to dataframe
        trades_df = pd.DataFrame(self.trade_log)
        
        # Ensure exit_time is datetime
        if 'exit_time' in trades_df.columns:
            if trades_df['exit_time'].dtype == 'object':
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Extract month and year
            trades_df['month_year'] = trades_df['exit_time'].dt.strftime('%Y-%m')
            
            # Group by month and sum PnL
            monthly_pnl = trades_df.groupby('month_year')['pnl'].sum()
            
            # Convert to dictionary
            return monthly_pnl.to_dict()
        
        return {}
    
    def calculate_daily_returns(self) -> Dict[str, float]:
        """
        Calculate daily returns.
        
        Returns:
            Dict[str, float]: The daily returns.
        """
        if not self.trade_log:
            return {}
        
        # Convert trade log to dataframe
        trades_df = pd.DataFrame(self.trade_log)
        
        # Ensure exit_time is datetime
        if 'exit_time' in trades_df.columns:
            if trades_df['exit_time'].dtype == 'object':
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Extract date
            trades_df['date'] = trades_df['exit_time'].dt.strftime('%Y-%m-%d')
            
            # Group by date and sum PnL
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            
            # Convert to dictionary
            return daily_pnl.to_dict()
        
        return {}
    
    def calculate_periodic_metrics(self, period: str = 'weekly') -> List[Dict[str, Any]]:
        """
        Calculate metrics for each period.
        
        Args:
            period (str, optional): The period ('daily', 'weekly', or 'monthly'). Defaults to 'weekly'.
            
        Returns:
            List[Dict[str, Any]]: The periodic metrics.
        """
        if not self.trade_log:
            return []
        
        # Convert trade log to dataframe
        trades_df = pd.DataFrame(self.trade_log)
        
        # Ensure exit_time is datetime
        if 'exit_time' in trades_df.columns:
            if trades_df['exit_time'].dtype == 'object':
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Add period column
            if period == 'daily':
                trades_df['period'] = trades_df['exit_time'].dt.strftime('%Y-%m-%d')
            elif period == 'weekly':
                trades_df['period'] = trades_df['exit_time'].dt.strftime('%Y-%W')
            elif period == 'monthly':
                trades_df['period'] = trades_df['exit_time'].dt.strftime('%Y-%m')
            else:
                raise ValueError(f"Invalid period: {period}")
            
            # Group by period
            grouped = trades_df.groupby('period')
            
            # Calculate metrics for each period
            periodic_metrics = []
            
            for period_name, group in grouped:
                pnl = group['pnl'].sum()
                pips = group['pnl_pips'].sum() if 'pnl_pips' in group.columns else 0
                
                winning_trades = (group['pnl'] > 0).sum()
                losing_trades = (group['pnl'] < 0).sum()
                total_trades = len(group)
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                periodic_metrics.append({
                    'period': period_name,
                    'pnl': pnl,
                    'pips': pips,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'total_trades': total_trades,
                    'win_rate': win_rate
                })
            
            return periodic_metrics
        
        return []
    
    def save_metrics(self, file_name: Optional[str] = None) -> str:
        """
        Save metrics to a file.
        
        Args:
            file_name (Optional[str], optional): The file name. Defaults to None (auto-generated).
            
        Returns:
            str: The path to the saved file.
        """
        if not self.metrics:
            self.calculate_metrics()
        
        # Generate file name if not provided
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"metrics_{timestamp}.json"
        
        # Ensure file has .json extension
        if not file_name.endswith(".json"):
            file_name += ".json"
        
        # Create full path
        file_path = self.metrics_dir / file_name
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=4, default=str)
        
        logger.info(f"Saved metrics to {file_path}")
        
        return str(file_path)
    
    def load_metrics(self, file_path: str) -> Dict[str, Any]:
        """
        Load metrics from a file.
        
        Args:
            file_path (str): The path to the metrics file.
            
        Returns:
            Dict[str, Any]: The loaded metrics.
        """
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            self.metrics = metrics
            logger.info(f"Loaded metrics from {file_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics from {file_path}: {e}")
            return {}
    
    def get_latest_metrics_file(self) -> Optional[str]:
        """
        Get the path to the latest metrics file.
        
        Returns:
            Optional[str]: The path to the latest metrics file, or None if not found.
        """
        try:
            # Get all metrics files
            metrics_files = list(self.metrics_dir.glob("metrics_*.json"))
            
            if not metrics_files:
                return None
            
            # Sort by modification time (newest first)
            metrics_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Return the newest
            return str(metrics_files[0])
        except Exception as e:
            logger.error(f"Error getting latest metrics file: {e}")
            return None
    
    def get_metrics_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get metrics history for the specified number of days.
        
        Args:
            days (int, optional): The number of days. Defaults to 30.
            
        Returns:
            List[Dict[str, Any]]: The metrics history.
        """
        try:
            # Get all metrics files
            metrics_files = list(self.metrics_dir.glob("metrics_*.json"))
            
            if not metrics_files:
                return []
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter files by modification time
            recent_files = [f for f in metrics_files if f.stat().st_mtime >= cutoff_date.timestamp()]
            
            # Sort by modification time (oldest first)
            recent_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Load metrics from each file
            history = []
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)
                    
                    history.append(metrics)
                except Exception as e:
                    logger.error(f"Error loading metrics from {file_path}: {e}")
            
            return history
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []