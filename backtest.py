"""
Backtest Script
Script for backtesting trading strategies.
"""
import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
from pathlib import Path

from config.config import (
    SYMBOL, TIMEFRAMES, DEFAULT_TIMEFRAME, INITIAL_CAPITAL,
    RISK_PER_TRADE, DATA_DIR, REPORTS_DIR
)
from config.parameters import STRATEGY_PARAMS
from data.market_data import get_data_fallback
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.hybrid_voting import HybridVotingStrategy
from execution.simulator import ExecutionSimulator
from optimization.optimizer import StrategyOptimizer, WalkForwardOptimizer
from reporting.metrics import PerformanceMetrics
from reporting.pdf_generator import ReportGenerator
from reporting.dashboard import Dashboard
from utils.logger import setup_logger

logger = setup_logger("backtest")

def create_strategy(strategy_name: str) -> Any:
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

def run_backtest(
    symbol: str = SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategy_name: str = "hybrid_voting",
    initial_capital: float = INITIAL_CAPITAL,
    risk_per_trade: float = RISK_PER_TRADE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    optimize: bool = False,
    generate_report: bool = False,
    show_dashboard: bool = False
) -> Dict[str, Any]:
    """
    Run a backtest.
    
    Args:
        symbol (str, optional): Trading symbol. Defaults to SYMBOL from config.
        timeframe (str, optional): Trading timeframe. Defaults to DEFAULT_TIMEFRAME from config.
        strategy_name (str, optional): Strategy name. Defaults to "hybrid_voting".
        initial_capital (float, optional): Initial capital. Defaults to INITIAL_CAPITAL from config.
        risk_per_trade (float, optional): Risk per trade. Defaults to RISK_PER_TRADE from config.
        start_date (Optional[str], optional): Start date in format YYYY-MM-DD. Defaults to None.
        end_date (Optional[str], optional): End date in format YYYY-MM-DD. Defaults to None.
        optimize (bool, optional): Whether to optimize strategy. Defaults to False.
        generate_report (bool, optional): Whether to generate a PDF report. Defaults to False.
        show_dashboard (bool, optional): Whether to show dashboard. Defaults to False.
        
    Returns:
        Dict[str, Any]: The backtest results.
    """
    logger.info(f"Running backtest for {symbol} ({timeframe}) with {strategy_name} strategy")
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create strategy
    strategy = create_strategy(strategy_name)
    
    # Optimize if requested
    if optimize:
        logger.info("Optimizing strategy parameters")
        
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            period_days=90,
            timeframe=timeframe
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        if result and "best_params" in result:
            # Apply best parameters
            strategy.update_parameters(result["best_params"])
            logger.info(f"Applied optimized parameters: {result['best_params']}")
    
    # Create simulator
    simulator = ExecutionSimulator(
        symbol=symbol,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade
    )
    
    # Get data
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    df = get_data_fallback(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        logger.error("No data available for backtest")
        return {"error": "No data available"}
    
    # Run backtest
    logger.info("Running backtest simulation")
    
    backtest_df = simulator.simulate_backtest(df, strategy)
    
    # Get performance metrics
    performance = simulator.get_performance_summary()
    
    # Initialize metrics and report generator
    metrics = PerformanceMetrics(
        initial_capital=initial_capital,
        symbol=symbol,
        trade_log=simulator.get_trade_log(),
        equity_curve=simulator.get_equity_curve()
    )
    
    report_generator = ReportGenerator(
        metrics=metrics,
        trade_log=simulator.get_trade_log(),
        equity_curve=simulator.get_equity_curve()
    )
    
    # Generate report if requested
    if generate_report:
        logger.info("Generating PDF report")
        
        report_file = report_generator.generate_report(
            title=f"Backtest Report: {symbol} ({timeframe})",
            period="all",
            include_trades=True,
            report_type="detailed"
        )
        
        logger.info(f"Generated report: {report_file}")
    
    # Show dashboard if requested
    if show_dashboard:
        logger.info("Starting dashboard")
        
        dashboard = Dashboard(auto_open=True)
        
        # Update dashboard data
        dashboard.update_trade_log(simulator.get_trade_log())
        dashboard.update_equity_curve(simulator.get_equity_curve())
        dashboard.update_metrics(metrics.calculate_metrics())
        
        # Start dashboard
        dashboard.start()
    
    # Create results
    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_capital": performance["current_capital"],
        "trade_count": performance["total_trades"],
        "winning_trades": performance["winning_trades"],
        "losing_trades": performance["losing_trades"],
        "win_rate": performance["win_rate"],
        "profit": performance["total_profit"],
        "return_pct": performance["total_return_pct"],
        "max_drawdown": performance["max_drawdown_pct"],
        "profit_factor": performance["profit_factor"],
        "sharpe_ratio": performance.get("sharpe_ratio", 0),
        "strategy_params": strategy.params
    }
    
    logger.info(f"Backtest completed: {results['trade_count']} trades, ${results['profit']:.2f} profit")
    
    return results

def compare_strategies(
    symbol: str = SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategies: List[str] = ["ma_crossover", "rsi_mean_reversion", "hybrid_voting"],
    initial_capital: float = INITIAL_CAPITAL,
    risk_per_trade: float = RISK_PER_TRADE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    optimize: bool = False
) -> List[Dict[str, Any]]:
    """
    Compare multiple strategies on the same data.
    
    Args:
        symbol (str, optional): Trading symbol. Defaults to SYMBOL from config.
        timeframe (str, optional): Trading timeframe. Defaults to DEFAULT_TIMEFRAME from config.
        strategies (List[str], optional): List of strategy names. 
            Defaults to ["ma_crossover", "rsi_mean_reversion", "hybrid_voting"].
        initial_capital (float, optional): Initial capital. Defaults to INITIAL_CAPITAL from config.
        risk_per_trade (float, optional): Risk per trade. Defaults to RISK_PER_TRADE from config.
        start_date (Optional[str], optional): Start date in format YYYY-MM-DD. Defaults to None.
        end_date (Optional[str], optional): End date in format YYYY-MM-DD. Defaults to None.
        optimize (bool, optional): Whether to optimize strategies. Defaults to False.
        
    Returns:
        List[Dict[str, Any]]: The comparison results.
    """
    logger.info(f"Comparing strategies for {symbol} ({timeframe})")
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Run backtest for each strategy
    results = []
    
    for strategy_name in strategies:
        logger.info(f"Testing strategy: {strategy_name}")
        
        result = run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            start_date=start_date,
            end_date=end_date,
            optimize=optimize,
            generate_report=False,
            show_dashboard=False
        )
        
        results.append(result)
    
    # Create comparison charts
    create_comparison_charts(results)
    
    return results

def compare_parameters(
    symbol: str = SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategy_name: str = "hybrid_voting",
    param_name: str = "profit_target",
    param_values: List[Any] = None,
    initial_capital: float = INITIAL_CAPITAL,
    risk_per_trade: float = RISK_PER_TRADE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Compare different parameter values for a strategy.
    
    Args:
        symbol (str, optional): Trading symbol. Defaults to SYMBOL from config.
        timeframe (str, optional): Trading timeframe. Defaults to DEFAULT_TIMEFRAME from config.
        strategy_name (str, optional): Strategy name. Defaults to "hybrid_voting".
        param_name (str, optional): Parameter name to vary. Defaults to "profit_target".
        param_values (List[Any], optional): List of parameter values to test. 
            Defaults to None (auto-generated).
        initial_capital (float, optional): Initial capital. Defaults to INITIAL_CAPITAL from config.
        risk_per_trade (float, optional): Risk per trade. Defaults to RISK_PER_TRADE from config.
        start_date (Optional[str], optional): Start date in format YYYY-MM-DD. Defaults to None.
        end_date (Optional[str], optional): End date in format YYYY-MM-DD. Defaults to None.
        
    Returns:
        List[Dict[str, Any]]: The comparison results.
    """
    logger.info(f"Comparing parameter values for {strategy_name} ({param_name})")
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get default parameter values if not provided
    if param_values is None:
        # Get parameter from STRATEGY_PARAMS
        strategy_params = STRATEGY_PARAMS.get(strategy_name, {})
        default_value = strategy_params.get(param_name, 0)
        
        # Generate values around default
        if isinstance(default_value, int):
            param_values = [max(1, int(default_value * 0.5)),
                           int(default_value * 0.75),
                           default_value,
                           int(default_value * 1.25),
                           int(default_value * 1.5)]
        elif isinstance(default_value, float):
            param_values = [default_value * 0.5,
                           default_value * 0.75,
                           default_value,
                           default_value * 1.25,
                           default_value * 1.5]
        else:
            logger.error(f"Cannot auto-generate values for {param_name}")
            return []
    
    # Run backtest for each parameter value
    results = []
    
    for value in param_values:
        logger.info(f"Testing {param_name} = {value}")
        
        # Create strategy
        strategy = create_strategy(strategy_name)
        
        # Update parameter
        params = strategy.params.copy()
        params[param_name] = value
        strategy.update_parameters(params)
        
        # Create simulator
        simulator = ExecutionSimulator(
            symbol=symbol,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade
        )
        
        # Get data
        df = get_data_fallback(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            logger.error("No data available for backtest")
            continue
        
        # Run backtest
        backtest_df = simulator.simulate_backtest(df, strategy)
        
        # Get performance metrics
        performance = simulator.get_performance_summary()
        
        # Create result
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "param_name": param_name,
            "param_value": value,
            "trade_count": performance["total_trades"],
            "win_rate": performance["win_rate"],
            "profit": performance["total_profit"],
            "return_pct": performance["total_return_pct"],
            "max_drawdown": performance["max_drawdown_pct"],
            "profit_factor": performance["profit_factor"]
        }
        
        results.append(result)
    
    # Create parameter comparison chart
    create_parameter_chart(results, param_name)
    
    return results

def create_comparison_charts(results: List[Dict[str, Any]]) -> None:
    """
    Create comparison charts for multiple strategies.
    
    Args:
        results (List[Dict[str, Any]]): The comparison results.
    """
    # Set up figure
    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background')
    
    # Extract data
    strategies = [r["strategy"] for r in results]
    returns = [r["return_pct"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    profit_factors = [r["profit_factor"] for r in results]
    trade_counts = [r["trade_count"] for r in results]
    
    # Create subplots
    plt.subplot(2, 3, 1)
    bars = plt.bar(strategies, returns)
    for i, bar in enumerate(bars):
        if returns[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.title('Total Return (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 2)
    bars = plt.bar(strategies, drawdowns)
    for bar in bars:
        bar.set_color('red')
    plt.title('Max Drawdown (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 3)
    bars = plt.bar(strategies, win_rates)
    for bar in bars:
        bar.set_color('blue')
    plt.title('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 4)
    bars = plt.bar(strategies, profit_factors)
    for i, bar in enumerate(bars):
        if profit_factors[i] >= 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.title('Profit Factor')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 5)
    bars = plt.bar(strategies, trade_counts)
    for bar in bars:
        bar.set_color('purple')
    plt.title('Trade Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Create equity curves subplot
    plt.subplot(2, 3, 6)
    
    # Get equity curves (need to run backtests again)
    symbol = results[0]["symbol"]
    timeframe = results[0]["timeframe"]
    start_date = results[0]["start_date"]
    end_date = results[0]["end_date"]
    
    # Get data once
    df = get_data_fallback(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if not df.empty:
        for result in results:
            strategy_name = result["strategy"]
            strategy = create_strategy(strategy_name)
            
            # Apply parameters if available
            if "strategy_params" in result:
                strategy.update_parameters(result["strategy_params"])
            
            simulator = ExecutionSimulator(symbol=symbol)
            backtest_df = simulator.simulate_backtest(df, strategy)
            
            # Normalize equity curve
            equity = np.array(simulator.get_equity_curve())
            equity = equity / equity[0] * 100  # Convert to percentage of initial
            
            plt.plot(equity, label=strategy_name)
    
    plt.title('Equity Curves (% of Initial)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(exist_ok=True)
    plt.savefig(reports_dir / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # Show figure
    plt.show()

def create_parameter_chart(results: List[Dict[str, Any]], param_name: str) -> None:
    """
    Create parameter comparison chart.
    
    Args:
        results (List[Dict[str, Any]]): The parameter comparison results.
        param_name (str): The parameter name.
    """
    # Set up figure
    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background')
    
    # Extract data
    param_values = [str(r["param_value"]) for r in results]
    returns = [r["return_pct"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    profit_factors = [r["profit_factor"] for r in results]
    trade_counts = [r["trade_count"] for r in results]
    
    # Create subplots
    plt.subplot(2, 3, 1)
    bars = plt.bar(param_values, returns)
    for i, bar in enumerate(bars):
        if returns[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.title('Total Return (%)')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 2)
    bars = plt.bar(param_values, drawdowns)
    for bar in bars:
        bar.set_color('red')
    plt.title('Max Drawdown (%)')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 3)
    bars = plt.bar(param_values, win_rates)
    for bar in bars:
        bar.set_color('blue')
    plt.title('Win Rate (%)')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 4)
    bars = plt.bar(param_values, profit_factors)
    for i, bar in enumerate(bars):
        if profit_factors[i] >= 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.title('Profit Factor')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 5)
    bars = plt.bar(param_values, trade_counts)
    for bar in bars:
        bar.set_color('purple')
    plt.title('Trade Count')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    # Create combined metric subplot
    plt.subplot(2, 3, 6)
    
    # Calculate combined score
    # Normalize metrics
    norm_returns = np.array(returns) / np.max(np.abs(returns)) if np.max(np.abs(returns)) > 0 else np.zeros_like(returns)
    norm_drawdowns = 1 - np.array(drawdowns) / np.max(drawdowns) if np.max(drawdowns) > 0 else np.ones_like(drawdowns)
    norm_win_rates = np.array(win_rates) / 100
    norm_profit_factors = np.array(profit_factors) / np.max(profit_factors) if np.max(profit_factors) > 0 else np.zeros_like(profit_factors)
    
    # Combined score (weighted average)
    combined_scores = (
        0.35 * norm_returns +
        0.25 * norm_drawdowns +
        0.20 * norm_win_rates +
        0.20 * norm_profit_factors
    )
    
    bars = plt.bar(param_values, combined_scores)
    for i, bar in enumerate(bars):
        bar.set_color('cyan')
    plt.title('Combined Score')
    plt.xlabel(param_name)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(exist_ok=True)
    plt.savefig(reports_dir / f"parameter_comparison_{param_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # Show figure
    plt.show()

def validate_date_format(date_str):
    """
    Validate that a date string is in YYYY-MM-DD format.
    
    Args:
        date_str (str): Date string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FX Trading Bot Backtest")
    
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, choices=TIMEFRAMES, help="Trading timeframe")
    parser.add_argument("--strategy", type=str, default="hybrid_voting", 
                       choices=["ma_crossover", "rsi_mean_reversion", "hybrid_voting"], 
                       help="Trading strategy")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE, help="Risk per trade")
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--optimize", action="store_true", help="Optimize strategy parameters")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard")
    parser.add_argument("--compare", action="store_true", help="Compare strategies")
    parser.add_argument("--param", type=str, help="Parameter to compare")
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Validate date formats
    if not validate_date_format(args.start_date):
        print(f"Error: Start date '{args.start_date}' is not in YYYY-MM-DD format")
        sys.exit(1)
        
    if not validate_date_format(args.end_date):
        print(f"Error: End date '{args.end_date}' is not in YYYY-MM-DD format")
        sys.exit(1)
        
    # Validate date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    if start_date > end_date:
        print(f"Error: Start date {args.start_date} is after end date {args.end_date}")
        sys.exit(1)
        
    if end_date > datetime.now() + timedelta(days=1):
        print(f"Warning: End date {args.end_date} is in the future. Using current date instead.")
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Check mode
    if args.compare:
        # Compare strategies
        results = compare_strategies(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategies=["ma_crossover", "rsi_mean_reversion", "hybrid_voting"],
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            start_date=args.start_date,
            end_date=args.end_date,
            optimize=args.optimize
        )
        
        # Print summary
        print("\nStrategy Comparison:")
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print("-" * 40)
        
        for result in results:
            print(f"Strategy: {result['strategy']}")
            print(f"  Return: {result['return_pct']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Trades: {result['trade_count']}")
            print()
        
    elif args.param:
        # Compare parameter values
        results = compare_parameters(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy_name=args.strategy,
            param_name=args.param,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print summary
        print(f"\nParameter Comparison ({args.param}):")
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Strategy: {args.strategy}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print("-" * 40)
        
        for result in results:
            print(f"{args.param}: {result['param_value']}")
            print(f"  Return: {result['return_pct']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Trades: {result['trade_count']}")
            print()
        
    else:
        # Run single backtest
        result = run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy_name=args.strategy,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            start_date=args.start_date,
            end_date=args.end_date,
            optimize=args.optimize,
            generate_report=args.report,
            show_dashboard=args.dashboard
        )
        
        # Print results
        print("\nBacktest Results:")
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Period: {result['start_date']} to {result['end_date']}")
        print("-" * 40)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Initial Capital: ${result['initial_capital']:.2f}")
            print(f"Final Capital: ${result['final_capital']:.2f}")
            print(f"Total Return: {result['return_pct']:.2f}%")
            print(f"Profit: ${result['profit']:.2f}")
            print(f"Trades: {result['trade_count']}")
            print(f"Win Rate: {result['win_rate']:.2f}%")
            print(f"Profit Factor: {result['profit_factor']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        
        # Keep dashboard running if enabled
        if args.dashboard:
            print("\nPress Ctrl+C to exit...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    main()