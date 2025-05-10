"""
Main Application
The main entry point for the FX Trading Bot.
"""
import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
import threading
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
from pathlib import Path

from config.config import (
    SYMBOL, TIMEFRAMES, DEFAULT_TIMEFRAME, INITIAL_CAPITAL,
    RISK_PER_TRADE, MAX_OPEN_POSITIONS, DEFAULT_STRATEGY,
    OPTIMIZATION_INTERVAL, OPTIMIZATION_PERIOD
)
from config.parameters import STRATEGY_PARAMS
from data.market_data import get_data_fallback, MarketData
from data.sentiment import SentimentAnalyzer
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.hybrid_voting import HybridVotingStrategy
from execution.simulator import ExecutionSimulator
from execution.position_sizing import calculate_position_size
from optimization.optimizer import StrategyOptimizer, WalkForwardOptimizer
from reporting.metrics import PerformanceMetrics
from reporting.pdf_generator import ReportGenerator
from reporting.dashboard import Dashboard
from alerts.discord import (
    send_trade_alert, send_error_alert, 
    send_optimization_alert, send_status_alert,
    send_performance_alert
)
from utils.logger import setup_logger

logger = setup_logger("main")

class TradingBot:
    """
    Main trading bot class that manages all components.
    """
    def __init__(self, 
                 symbol: str = SYMBOL,
                 timeframe: str = DEFAULT_TIMEFRAME,
                 strategy_name: str = DEFAULT_STRATEGY,
                 initial_capital: float = INITIAL_CAPITAL,
                 risk_per_trade: float = RISK_PER_TRADE,
                 max_positions: int = MAX_OPEN_POSITIONS,
                 use_dashboard: bool = True,
                 auto_optimize: bool = True,
                 backtest_mode: bool = False):
        """
        Initialize the TradingBot.
        
        Args:
            symbol (str, optional): Trading symbol. Defaults to SYMBOL from config.
            timeframe (str, optional): Trading timeframe. Defaults to DEFAULT_TIMEFRAME from config.
            strategy_name (str, optional): Strategy name. Defaults to DEFAULT_STRATEGY from config.
            initial_capital (float, optional): Initial capital. Defaults to INITIAL_CAPITAL from config.
            risk_per_trade (float, optional): Risk per trade. Defaults to RISK_PER_TRADE from config.
            max_positions (int, optional): Maximum open positions. Defaults to MAX_OPEN_POSITIONS from config.
            use_dashboard (bool, optional): Whether to use dashboard. Defaults to True.
            auto_optimize (bool, optional): Whether to auto-optimize. Defaults to True.
            backtest_mode (bool, optional): Whether in backtest mode. Defaults to False.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.use_dashboard = use_dashboard
        self.auto_optimize = auto_optimize
        self.backtest_mode = backtest_mode
        
        # Storage directories
        self.data_dir = Path("data_files")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.simulator = ExecutionSimulator(
            symbol=symbol,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade
        )
        
        # Initialize strategy
        self.strategy = self._create_strategy(strategy_name)
        
        # Initialize dashboard
        self.dashboard = Dashboard(auto_open=use_dashboard) if use_dashboard else None
        
        # Initialize metrics
        self.metrics = PerformanceMetrics(
            initial_capital=initial_capital,
            symbol=symbol,
            trade_log=self.simulator.get_trade_log(),
            equity_curve=self.simulator.get_equity_curve()
        )
        
        # Initialize report generator
        self.report_generator = ReportGenerator(
            metrics=self.metrics,
            trade_log=self.simulator.get_trade_log(),
            equity_curve=self.simulator.get_equity_curve()
        )
        
        # Optimization settings
        self.last_optimization_time = None
        
        # Market data cache
        self.market_data_cache = None
        self.last_market_data_time = None
        
        # Bot status
        self.running = False
        self.stopping = False
        self.main_thread = None
        
        # Performance tracking
        self.performance_summary = {}
        
        logger.info(f"Initialized trading bot for {symbol} ({timeframe})")
    
    def _create_strategy(self, strategy_name: str) -> Any:
        """
        Create a strategy instance based on strategy name.
        
        Args:
            strategy_name (str): The strategy name.
            
        Returns:
            Any: The strategy instance.
        """
        if strategy_name == "ma_crossover":
            return MovingAverageCrossover()
        elif strategy_name == "rsi_mean_reversion":
            return RSIMeanReversion()
        elif strategy_name == "hybrid_voting":
            return HybridVotingStrategy()
        else:
            logger.warning(f"Unknown strategy: {strategy_name}, using Hybrid Voting")
            return HybridVotingStrategy()
    
    def optimize_strategy(self, force: bool = False) -> bool:
        """
        Optimize the strategy parameters.
        
        Args:
            force (bool, optional): Whether to force optimization. Defaults to False.
            
        Returns:
            bool: True if optimization was performed, False otherwise.
        """
        if not self.auto_optimize and not force:
            return False
        
        # Check if optimization is needed
        current_time = datetime.now()
        
        if not force and self.last_optimization_time:
            time_since_last = (current_time - self.last_optimization_time).total_seconds()
            
            if time_since_last < OPTIMIZATION_INTERVAL * 24 * 3600:  # Convert days to seconds
                logger.info(f"Skipping optimization, last one was {time_since_last / 3600:.1f} hours ago")
                return False
        
        logger.info(f"Starting optimization for {self.strategy_name}")
        
        try:
            # Run walk-forward optimization
            optimizer = WalkForwardOptimizer(
                strategy_name=self.strategy_name,
                in_sample_days=90,
                out_sample_days=30,
                timeframe=self.timeframe,
                n_windows=3,
                trials_per_window=50
            )
            
            # Get robust parameters
            wfo_result = optimizer.run_walk_forward_optimization()
            
            if wfo_result and "robust_parameters" in wfo_result:
                # Apply parameters to strategy
                self.strategy.update_parameters(wfo_result["robust_parameters"])
                
                # Send alert
                send_optimization_alert(
                    strategy=self.strategy_name,
                    best_params=wfo_result["robust_parameters"],
                    performance=wfo_result["overall_metrics"],
                    optimization_type="Walk-Forward"
                )
                
                # Update last optimization time
                self.last_optimization_time = current_time
                
                logger.info(f"Optimization completed successfully")
                return True
            else:
                logger.warning("Optimization did not return robust parameters")
                return False
                
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            send_error_alert(
                error_message=f"Optimization failed: {e}",
                error_type="Optimization Error",
                details={"strategy": self.strategy_name}
            )
            return False
    
    def get_market_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get market data with caching.
        
        Args:
            force_refresh (bool, optional): Whether to force refresh. Defaults to False.
            
        Returns:
            pd.DataFrame: The market data.
        """
        current_time = datetime.now()
        
        # Determine if cache is valid
        cache_valid = (
            not force_refresh and
            self.market_data_cache is not None and
            self.last_market_data_time is not None and
            (current_time - self.last_market_data_time).total_seconds() < 300  # 5 minutes
        )
        
        if cache_valid:
            return self.market_data_cache
        
        # Fetch fresh data
        try:
            logger.info(f"Fetching market data for {self.symbol} ({self.timeframe})")
            
            # Get data
            df = get_data_fallback(
                symbol=self.symbol,
                timeframe=self.timeframe,
                outputsize=1000  # Last 1000 bars
            )
            
            # Update cache
            self.market_data_cache = df
            self.last_market_data_time = current_time
            
            # Update dashboard with latest price
            if self.use_dashboard and self.dashboard:
                self.dashboard.update_market_data(df['close'].iloc[-1], self.symbol)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            
            # Return cache if available
            if self.market_data_cache is not None:
                logger.warning("Using cached market data")
                return self.market_data_cache
            
            # Create empty dataframe if no cache
            logger.error("No market data available")
            return pd.DataFrame()
    
    def get_sentiment_data(self) -> Dict[str, Any]:
        """
        Get sentiment data for the current symbol.
        
        Returns:
            Dict[str, Any]: The sentiment data.
        """
        try:
            logger.info(f"Fetching sentiment data for {self.symbol}")
            
            # Create sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer(symbol=self.symbol)
            
            # Get sentiment signal
            sentiment_data = sentiment_analyzer.get_sentiment_signal()
            
            # Update dashboard
            if self.use_dashboard and self.dashboard:
                self.dashboard.update_sentiment_data(sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            return {
                "symbol": self.symbol,
                "signal": "neutral",
                "strength": 0,
                "confidence": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def process_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process signals from the strategy.
        
        Args:
            df (pd.DataFrame): The market data.
            
        Returns:
            Dict[str, Any]: The signal data.
        """
        if df.empty:
            return {"signal": 0, "strength": 0}
        
        # Generate signals
        df_with_signals = self.strategy.generate_signals(df)
        
        # Get the latest signal
        latest_signal = df_with_signals['signal'].iloc[-1] if 'signal' in df_with_signals.columns else 0
        
        # Get sentiment-adjusted signal if available
        if 'sentiment_signal' in df_with_signals.columns:
            sentiment_signal = df_with_signals['sentiment_signal'].iloc[-1]
        else:
            sentiment_signal = latest_signal
        
        # Determine signal strength
        signal_strength = abs(sentiment_signal)
        
        # Get position
        position = df_with_signals['position'].iloc[-1] if 'position' in df_with_signals.columns else 0
        
        # Get additional metadata
        metadata = self.strategy.get_signal_metadata(df_with_signals, -1)
        
        return {
            "signal": sentiment_signal,
            "strength": signal_strength,
            "position": position,
            "metadata": metadata
        }
    
    def check_and_place_trades(self) -> bool:
        """
        Check for new signals and place trades accordingly.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get market data
            df = self.get_market_data()
            
            if df.empty:
                logger.error("No market data available, skipping trade check")
                return False
            
            # Validate dataframe has required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Market data missing required columns: {missing_columns}")
                return False
            
            # Get current price (with validation)
            try:
                current_price = df['close'].iloc[-1]
                if not pd.isna(current_price) and current_price <= 0:
                    logger.error(f"Invalid price data: {current_price}")
                    return False
            except Exception as e:
                logger.error(f"Error getting current price: {e}")
                return False
                
            current_time = datetime.now()
            
            # Update any existing position
            current_position = self.simulator.get_current_position()
            
            if current_position:
                try:
                    # Check trailing stop
                    self.strategy.update_trailing_stop(current_price)
                    
                    # Check for exit
                    exit_flag, exit_reason = self.strategy.should_exit_position(current_price, current_time)
                    
                    if exit_flag:
                        # Exit position
                        position_result = self.simulator.exit_position(current_time, current_price, exit_reason)
                        
                        if position_result:
                            # Update metrics
                            self.update_metrics()
                            
                            # Generate report if needed
                            self.generate_report_if_needed()
                            
                            logger.info(f"Exited position: {exit_reason}")
                        else:
                            logger.error("Failed to exit position")
                        
                        return True
                    
                    # Update position in simulator
                    self.simulator.update_position(current_time, current_price)
                    
                    # Already in a position, no need to check for new trades
                    return True
                except Exception as e:
                    logger.error(f"Error updating position: {e}")
                    # Try to exit position to prevent further issues
                    try:
                        self.simulator.exit_position(current_time, current_price, "error")
                        logger.info("Exited position due to error")
                    except:
                        logger.error("Failed to exit position after error")
                    return False
            
            # Process signals
            signal_data = self.process_signals(df)
            
            # Check if we have a valid signal
            signal = signal_data.get("signal", 0)
            signal_strength = signal_data.get("strength", 0)
            
            # Trading logic
            if signal > 0.5:  # Buy signal
                try:
                    # Calculate stop loss and take profit
                    stop_loss = self.strategy.calculate_stop_loss(current_price, 'long', df)
                    take_profit = self.strategy.calculate_take_profit(current_price, 'long', df)
                    
                    # Validate stop loss and take profit
                    if not isinstance(stop_loss, (int, float)) or not isinstance(take_profit, (int, float)):
                        logger.error(f"Invalid stop loss or take profit: SL={stop_loss}, TP={take_profit}")
                        return False
                    
                    if stop_loss >= current_price or take_profit <= current_price:
                        logger.error(f"Invalid stop loss or take profit levels for long position: SL={stop_loss}, TP={take_profit}, Price={current_price}")
                        return False
                    
                    # Enter position
                    position = self.simulator.enter_position(
                        signal_time=current_time,
                        market_price=current_price,
                        position_type='long',
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=self.strategy_name,
                        metadata=signal_data.get("metadata", {})
                    )
                    
                    if position:
                        # Update metrics
                        self.update_metrics()
                        logger.info(f"Entered LONG position at {current_price:.5f}")
                    else:
                        logger.error("Failed to enter long position")
                    
                    return True
                except Exception as e:
                    logger.error(f"Error entering long position: {e}")
                    return False
                
            elif signal < -0.5:  # Sell signal
                try:
                    # Calculate stop loss and take profit
                    stop_loss = self.strategy.calculate_stop_loss(current_price, 'short', df)
                    take_profit = self.strategy.calculate_take_profit(current_price, 'short', df)
                    
                    # Validate stop loss and take profit
                    if not isinstance(stop_loss, (int, float)) or not isinstance(take_profit, (int, float)):
                        logger.error(f"Invalid stop loss or take profit: SL={stop_loss}, TP={take_profit}")
                        return False
                    
                    if stop_loss <= current_price or take_profit >= current_price:
                        logger.error(f"Invalid stop loss or take profit levels for short position: SL={stop_loss}, TP={take_profit}, Price={current_price}")
                        return False
                    
                    # Enter position
                    position = self.simulator.enter_position(
                        signal_time=current_time,
                        market_price=current_price,
                        position_type='short',
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=self.strategy_name,
                        metadata=signal_data.get("metadata", {})
                    )
                    
                    if position:
                        # Update metrics
                        self.update_metrics()
                        logger.info(f"Entered SHORT position at {current_price:.5f}")
                    else:
                        logger.error("Failed to enter short position")
                    
                    return True
                except Exception as e:
                    logger.error(f"Error entering short position: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking and placing trades: {e}")
            send_error_alert(
                error_message=f"Error checking and placing trades: {e}",
                error_type="Trading Error",
                details={"symbol": self.symbol, "timeframe": self.timeframe}
            )
            return False
    
    def update_metrics(self) -> None:
        """
        Update and save performance metrics.
        """
        try:
            # Update metrics with latest data
            self.metrics.update_trade_log(self.simulator.get_trade_log())
            self.metrics.update_equity_curve(self.simulator.get_equity_curve())
            
            # Calculate metrics
            metrics = self.metrics.calculate_metrics()
            
            # Update dashboard
            if self.use_dashboard and self.dashboard:
                self.dashboard.update_trade_log(self.simulator.get_trade_log())
                self.dashboard.update_equity_curve(self.simulator.get_equity_curve())
                self.dashboard.update_metrics(metrics)
            
            # Save metrics
            self.metrics.save_metrics()
            
            # Update performance summary
            self.performance_summary = metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def generate_report_if_needed(self) -> None:
        """
        Generate a PDF report if needed.
        """
        try:
            if self.report_generator.should_generate_report():
                logger.info("Generating PDF report")
                
                # Update report generator with latest data
                self.report_generator.update_trade_log(self.simulator.get_trade_log())
                self.report_generator.update_equity_curve(self.simulator.get_equity_curve())
                
                # Generate report
                report_file = self.report_generator.generate_report()
                
                logger.info(f"Generated report: {report_file}")
                
                # Send performance alert
                send_performance_alert(self.performance_summary)
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def main_loop(self) -> None:
        """
        Main trading loop.
        """
        logger.info("Starting main trading loop")
        
        # Send status alert
        send_status_alert(
            status="Bot started",
            details={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_name
            }
        )
        
        # Initialize loop variables
        last_optimization_check = datetime.now()
        last_report_check = datetime.now()
        
        while self.running and not self.stopping:
            try:
                # Check for trades
                self.check_and_place_trades()
                
                # Check if optimization is needed
                if self.auto_optimize:
                    current_time = datetime.now()
                    time_since_last_check = (current_time - last_optimization_check).total_seconds()
                    
                    if time_since_last_check > 3600:  # Check every hour
                        logger.info("Checking if optimization is needed")
                        self.optimize_strategy()
                        last_optimization_check = current_time
                
                # Check if report generation is needed
                current_time = datetime.now()
                time_since_last_check = (current_time - last_report_check).total_seconds()
                
                if time_since_last_check > 3600:  # Check every hour
                    logger.info("Checking if report generation is needed")
                    self.generate_report_if_needed()
                    last_report_check = current_time
                
                # Update sentiment data occasionally
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    self.get_sentiment_data()
                
                # Sleep before next iteration
                time.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(30)  # Longer sleep on error
                
            # Check if stopping
            if self.stopping:
                logger.info("Stopping main loop")
                break
        
        logger.info("Main loop stopped")
    
    def start(self) -> None:
        """
        Start the trading bot.
        """
        if self.running:
            logger.warning("Bot is already running")
            return
        
        # Set running flag
        self.running = True
        self.stopping = False
        
        # Start dashboard
        if self.use_dashboard and self.dashboard:
            self.dashboard.start()
        
        # Apply optimization if auto_optimize is enabled
        if self.auto_optimize:
            self.optimize_strategy()
        
        # Start main thread
        self.main_thread = threading.Thread(target=self.main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        logger.info("Trading bot started")
    
    def stop(self) -> None:
        """
        Stop the trading bot.
        """
        if not self.running:
            logger.warning("Bot is not running")
            return
        
        # Set stopping flag
        self.stopping = True
        
        # Wait for main thread to stop
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
        
        # Stop dashboard
        if self.use_dashboard and self.dashboard:
            self.dashboard.stop()
        
        # Reset flags
        self.running = False
        self.stopping = False
        
        # Send status alert
        send_status_alert(
            status="Bot stopped",
            details={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_name
            }
        )
        
        logger.info("Trading bot stopped")
    
    def run_backtest(self, 
                   start_date: str,
                   end_date: str,
                   generate_report: bool = True) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            start_date (str): Start date in format YYYY-MM-DD.
            end_date (str): End date in format YYYY-MM-DD.
            generate_report (bool, optional): Whether to generate a PDF report. Defaults to True.
            
        Returns:
            Dict[str, Any]: The backtest results.
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        try:
            # Get historical data
            df = get_data_fallback(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.error("No data available for backtest")
                return {"error": "No data available"}
            
            # Run backtest
            backtest_df = self.simulator.simulate_backtest(df, self.strategy)
            
            # Get performance metrics
            performance = self.simulator.get_performance_summary()
            
            logger.info(f"Backtest completed: {performance['total_trades']} trades, {performance['total_profit']:.2f} USD profit")
            
            # Update metrics
            self.update_metrics()
            
            # Generate report if requested
            if generate_report:
                self.generate_report_if_needed()
            
            # Create backtest results
            results = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_name,
                "start_date": start_date,
                "end_date": end_date,
                "performance": performance,
                "trade_count": performance["total_trades"],
                "win_rate": performance["win_rate"],
                "profit": performance["total_profit"],
                "profit_factor": performance["profit_factor"],
                "max_drawdown": performance["max_drawdown_pct"]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {"error": str(e)}
    
    def optimize_and_backtest(self, 
                             start_date: str,
                             end_date: str,
                             optimization_period: int = OPTIMIZATION_PERIOD,
                             generate_report: bool = True) -> Dict[str, Any]:
        """
        Optimize the strategy and run a backtest.
        
        Args:
            start_date (str): Start date in format YYYY-MM-DD.
            end_date (str): End date in format YYYY-MM-DD.
            optimization_period (int, optional): Optimization period in days. 
                Defaults to OPTIMIZATION_PERIOD from config.
            generate_report (bool, optional): Whether to generate a PDF report. Defaults to True.
            
        Returns:
            Dict[str, Any]: The backtest results.
        """
        logger.info(f"Running optimization and backtest from {start_date} to {end_date}")
        
        try:
            # Run optimization
            self.optimize_strategy(force=True)
            
            # Run backtest
            results = self.run_backtest(
                start_date=start_date,
                end_date=end_date,
                generate_report=generate_report
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error running optimization and backtest: {e}")
            return {"error": str(e)}

def signal_handler(signum, frame):
    """
    Handle termination signals.
    """
    logger.info(f"Received signal {signum}, shutting down...")
    
    # Attempt to gracefully stop bots
    for bot in running_bots:
        try:
            bot.stop()
        except:
            pass
    
    sys.exit(0)

# Global list of running bots
running_bots = []

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FX Trading Bot")
    
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="Trading timeframe")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY, help="Trading strategy")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE, help="Risk per trade")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--no-optimize", action="store_true", help="Disable auto-optimization")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--optimize", action="store_true", help="Run optimization before backtest")
    parser.add_argument("--no-report", action="store_true", help="Disable PDF report generation")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create trading bot
    bot = TradingBot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        use_dashboard=not args.no_dashboard,
        auto_optimize=not args.no_optimize,
        backtest_mode=args.backtest
    )
    
    # Add to global list
    running_bots.append(bot)
    
    # Check mode
    if args.backtest:
        # Check required arguments
        if not args.start_date or not args.end_date:
            logger.error("Backtest mode requires --start-date and --end-date")
            sys.exit(1)
        
        # Run backtest or optimization+backtest
        if args.optimize:
            results = bot.optimize_and_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                generate_report=not args.no_report
            )
        else:
            results = bot.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                generate_report=not args.no_report
            )
        
        # Print results summary
        print("\nBacktest Results:")
        print(f"Symbol: {results.get('symbol', args.symbol)}")
        print(f"Timeframe: {results.get('timeframe', args.timeframe)}")
        print(f"Strategy: {results.get('strategy', args.strategy)}")
        print(f"Period: {results.get('start_date', args.start_date)} to {results.get('end_date', args.end_date)}")
        print("-" * 40)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Total Trades: {results.get('trade_count', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"Profit: ${results.get('profit', 0):.2f}")
            print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        
        # Keep dashboard running if enabled
        if not args.no_dashboard:
            print("\nPress Ctrl+C to exit...")
            
            while True:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
        
    else:
        # Start trading bot
        bot.start()
        
        print(f"\nFX Trading Bot started for {args.symbol} ({args.timeframe})")
        print(f"Strategy: {args.strategy}")
        print(f"Initial Capital: ${args.capital:.2f}")
        print(f"Risk per Trade: {args.risk:.1%}")
        print("\nPress Ctrl+C to stop...")
        
        # Keep main thread alive
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
        
        # Stop bot
        bot.stop()
        print("\nFX Trading Bot stopped")
    
    # Clean up
    running_bots.remove(bot)

if __name__ == "__main__":
    main()