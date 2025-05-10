"""
Optimizer Module
Implements hyperparameter optimization using Optuna.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import optuna
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple, Callable
import joblib
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config.config import (
    OPTIMIZATION_TRIALS, OPTIMIZATION_PERIOD, OPTIMIZATION_INTERVAL,
    DATA_DIR, SYMBOL
)
from config.parameters import STRATEGY_PARAMS
from config.parameters import STRATEGY_HYPERPARAMS
from utils.logger import setup_logger
from execution.simulator import ExecutionSimulator
from data.market_data import get_data_fallback
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.hybrid_voting import HybridVotingStrategy
from utils.helpers import calculate_sharpe_ratio, calculate_profit_factor

logger = setup_logger("optimizer")

class StrategyOptimizer:
    """
    Optimizes strategy parameters using Optuna.
    """
    def __init__(self, 
                 strategy_name: str,
                 study_name: Optional[str] = None,
                 trials: int = OPTIMIZATION_TRIALS,
                 period_days: int = OPTIMIZATION_PERIOD,
                 timeframe: str = "1h"):
        """
        Initialize the StrategyOptimizer.
        
        Args:
            strategy_name (str): The name of the strategy to optimize.
            study_name (Optional[str], optional): The name of the optimization study.
                Defaults to None (auto-generated).
            trials (int, optional): The number of optimization trials.
                Defaults to OPTIMIZATION_TRIALS from config.
            period_days (int, optional): The number of days to use for optimization.
                Defaults to OPTIMIZATION_PERIOD from config.
            timeframe (str, optional): The timeframe for data. Defaults to "1h".
        """
        self.strategy_name = strategy_name
        self.study_name = study_name or f"{strategy_name}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trials = trials
        self.period_days = period_days
        self.timeframe = timeframe
        
        # Storage directory for studies
        self.storage_dir = os.path.join(DATA_DIR, "optimization")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create strategy instance based on name
        self.strategy = self._create_strategy_instance(strategy_name)
        
        # Get hyperparameter space
        self.param_space = STRATEGY_HYPERPARAMS.get(strategy_name, {})
        
        logger.info(f"Initialized optimizer for {strategy_name} strategy")
    
    def _create_strategy_instance(self, strategy_name: str) -> Any:
        """
        Create a strategy instance based on the strategy name.
        
        Args:
            strategy_name (str): The name of the strategy.
            
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
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _get_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generate parameters for a trial.
        
        Args:
            trial (optuna.Trial): The Optuna trial.
            
        Returns:
            Dict[str, Any]: The generated parameters.
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config.get("type")
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config.get("low"), 
                    param_config.get("high"),
                    step=param_config.get("step", 1)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config.get("low"),
                    param_config.get("high"),
                    step=param_config.get("step")
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config.get("choices", [])
                )
            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")
        
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial (optuna.Trial): The Optuna trial.
            
        Returns:
            float: The objective value to maximize.
        """
        # Get parameters for this trial
        params = self._get_parameters(trial)
        
        # Update strategy with these parameters
        self.strategy.update_parameters(params)
        
        # Get data for the optimization period
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.period_days)).strftime("%Y-%m-%d")
        
        df = get_data_fallback(
            symbol=SYMBOL,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Skip trial if not enough data
        if len(df) < 100:
            logger.warning(f"Not enough data for trial: {len(df)} bars")
            return float("-inf")
        
        # Create a simulator for backtesting
        simulator = ExecutionSimulator(symbol=SYMBOL)
        
        # Run backtest
        backtest_df = simulator.simulate_backtest(df, self.strategy)
        
        # Get performance metrics
        performance = simulator.get_performance_summary()
        
        # Calculate objective value based on multiple metrics
        # We want to maximize this value
        
        # Minimum number of trades required
        min_trades = 20
        if performance["total_trades"] < min_trades:
            return float("-inf")
        
        # Calculate Sharpe ratio if not in performance summary
        if "sharpe_ratio" not in performance or performance["sharpe_ratio"] == 0:
            # Calculate daily returns from equity curve
            daily_returns = np.diff(backtest_df['equity'].values) / backtest_df['equity'].values[:-1]
            sharpe = calculate_sharpe_ratio(daily_returns)
        else:
            sharpe = performance["sharpe_ratio"]
        
        # Profit factor
        profit_factor = performance["profit_factor"]
        
        # Return
        total_return = performance["total_return_pct"] / 100  # Convert to decimal
        
        # Win rate
        win_rate = performance["win_rate"] / 100  # Convert to decimal
        
        # Drawdown
        max_drawdown = performance["max_drawdown_pct"] / 100  # Convert to decimal
        
        # Profit per trade
        avg_trade = performance["avg_trade"] / performance["initial_capital"]
        
        # Calculate objective based on weighted combination of metrics
        # Adjust weights to prioritize certain metrics
        weights = {
            "sharpe": 0.3,
            "profit_factor": 0.2,
            "return": 0.2,
            "win_rate": 0.1,
            "drawdown": 0.1,
            "avg_trade": 0.1
        }
        
        objective = (
            weights["sharpe"] * sharpe +
            weights["profit_factor"] * min(profit_factor, 5) +  # Cap profit factor to prevent outliers
            weights["return"] * min(total_return, 1.0) +  # Cap return
            weights["win_rate"] * win_rate +
            weights["drawdown"] * (1 - max_drawdown) +  # Lower drawdown is better
            weights["avg_trade"] * min(avg_trade * 100, 1.0)  # Cap avg trade
        )
        
        # Log trial results
        logger.info(
            f"Trial {trial.number}: "
            f"Trades={performance['total_trades']}, "
            f"Return={total_return:.4f}, "
            f"PF={profit_factor:.2f}, "
            f"Sharpe={sharpe:.2f}, "
            f"Drawdown={max_drawdown:.4f}, "
            f"WinRate={win_rate:.4f}, "
            f"AvgTrade={avg_trade:.4f}, "
            f"Objective={objective:.4f}"
        )
        
        return objective
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dict[str, Any]: The best parameters and performance metrics.
        """
        logger.info(f"Starting optimization for {self.strategy_name} with {self.trials} trials")
        start_time = time.time()
        
        # Create storage path
        storage_path = os.path.join(self.storage_dir, f"{self.study_name}.pkl")
        
        # Create sampler and pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        
        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self._objective, n_trials=self.trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Save study
        joblib.dump(study, storage_path)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {study.best_value:.4f}")
        
        # Create result dictionary
        result = {
            "strategy": self.strategy_name,
            "study_name": self.study_name,
            "best_params": best_params,
            "best_value": study.best_value,
            "trials": self.trials,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "storage_path": storage_path
        }
        
        return result
    
    def apply_best_parameters(self, strategy: Any = None) -> Dict[str, Any]:
        """
        Apply the best parameters from a previous optimization to a strategy.
        
        Args:
            strategy (Any, optional): The strategy instance to update. 
                Defaults to None (use the optimizer's strategy).
                
        Returns:
            Dict[str, Any]: The best parameters.
        """
        # Get the strategy to update
        target_strategy = strategy or self.strategy
        
        # Load the latest study for this strategy if no specific study name provided
        storage_path = self._find_latest_study()
        
        if not storage_path:
            logger.warning(f"No optimization study found for {self.strategy_name}")
            return {}
        
        # Load study
        try:
            study = joblib.load(storage_path)
            best_params = study.best_params
            
            # Update strategy
            target_strategy.update_parameters(best_params)
            
            logger.info(f"Applied best parameters to {target_strategy.name} strategy: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error loading optimization study: {e}")
            return {}
    
    def _find_latest_study(self) -> Optional[str]:
        """
        Find the latest optimization study for the strategy.
        
        Returns:
            Optional[str]: The path to the latest study, or None if not found.
        """
        # Get all study files for this strategy
        study_files = [
            f for f in os.listdir(self.storage_dir)
            if f.startswith(f"{self.strategy_name}_opt_") and f.endswith(".pkl")
        ]
        
        if not study_files:
            return None
        
        # Sort by modification time (newest first)
        study_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.storage_dir, f)), reverse=True)
        
        # Return the newest
        return os.path.join(self.storage_dir, study_files[0])
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about past optimizations.
        
        Returns:
            Dict[str, Any]: Optimization statistics.
        """
        stats = {
            "strategy": self.strategy_name,
            "total_optimizations": 0,
            "latest_optimization": None,
            "best_value_history": [],
            "parameter_evolution": {}
        }
        
        # Get all study files for this strategy
        study_files = [
            f for f in os.listdir(self.storage_dir)
            if f.startswith(f"{self.strategy_name}_opt_") and f.endswith(".pkl")
        ]
        
        if not study_files:
            return stats
        
        # Sort by modification time (oldest first)
        study_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.storage_dir, f)))
        
        # Calculate statistics
        stats["total_optimizations"] = len(study_files)
        
        # Process each study
        for i, study_file in enumerate(study_files):
            try:
                study_path = os.path.join(self.storage_dir, study_file)
                study = joblib.load(study_path)
                
                # Get timestamp from filename or modification time
                try:
                    timestamp_str = study_file.split("_opt_")[1].split(".")[0]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except:
                    timestamp = datetime.fromtimestamp(os.path.getmtime(study_path))
                
                # Add to best value history
                stats["best_value_history"].append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "value": study.best_value
                })
                
                # Track parameter evolution
                for param_name, param_value in study.best_params.items():
                    if param_name not in stats["parameter_evolution"]:
                        stats["parameter_evolution"][param_name] = []
                    
                    stats["parameter_evolution"][param_name].append({
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "value": param_value
                    })
                
                # Update latest optimization info
                if i == len(study_files) - 1:
                    stats["latest_optimization"] = {
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "best_value": study.best_value,
                        "best_params": study.best_params,
                        "trials": len(study.trials),
                        "duration": None  # Cannot determine from the study object
                    }
            
            except Exception as e:
                logger.error(f"Error processing optimization study {study_file}: {e}")
        
        return stats
    
    def should_optimize(self, last_optimization_time: Optional[datetime] = None) -> bool:
        """
        Determine if optimization should be run based on the time since last optimization.
        
        Args:
            last_optimization_time (Optional[datetime], optional): The time of the last optimization.
                Defaults to None (determined from existing studies).
                
        Returns:
            bool: True if optimization should be run, False otherwise.
        """
        # If no last optimization time provided, find the latest study
        if last_optimization_time is None:
            latest_study_path = self._find_latest_study()
            
            if latest_study_path:
                last_optimization_time = datetime.fromtimestamp(os.path.getmtime(latest_study_path))
            else:
                # No previous optimization, so should optimize
                return True
        
        # Calculate time since last optimization
        time_since_last = datetime.now() - last_optimization_time
        
        # Check if enough time has passed
        return time_since_last.total_seconds() > OPTIMIZATION_INTERVAL * 24 * 3600  # Convert days to seconds

class WalkForwardOptimizer:
    """
    Implements Walk-Forward Optimization to avoid curve fitting.
    """
    def __init__(self, 
                 strategy_name: str,
                 in_sample_days: int = 90,
                 out_sample_days: int = 30,
                 timeframe: str = "1h",
                 n_windows: int = 3,
                 trials_per_window: int = 50):
        """
        Initialize the WalkForwardOptimizer.
        
        Args:
            strategy_name (str): The name of the strategy to optimize.
            in_sample_days (int, optional): Days for in-sample optimization. Defaults to 90.
            out_sample_days (int, optional): Days for out-of-sample testing. Defaults to 30.
            timeframe (str, optional): The timeframe for data. Defaults to "1h".
            n_windows (int, optional): Number of walk-forward windows. Defaults to 3.
            trials_per_window (int, optional): Trials per optimization window. Defaults to 50.
        """
        self.strategy_name = strategy_name
        self.in_sample_days = in_sample_days
        self.out_sample_days = out_sample_days
        self.timeframe = timeframe
        self.n_windows = n_windows
        self.trials_per_window = trials_per_window
        
        # Create strategy instance
        self.strategy = self._create_strategy_instance(strategy_name)
        
        # Storage directory for WFO results
        self.storage_dir = os.path.join(DATA_DIR, "walk_forward")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        logger.info(f"Initialized Walk-Forward Optimizer for {strategy_name}")
    
    def _create_strategy_instance(self, strategy_name: str) -> Any:
        """
        Create a strategy instance based on the strategy name.
        
        Args:
            strategy_name (str): The name of the strategy.
            
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
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def run_walk_forward_optimization(self) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Returns:
            Dict[str, Any]: The walk-forward optimization results.
        """
        logger.info("Starting walk-forward optimization")
        start_time = time.time()
        
        # Calculate date ranges for each window
        today = datetime.now()
        
        windows = []
        total_days = self.in_sample_days + self.out_sample_days
        
        for i in range(self.n_windows):
            window_end = today - timedelta(days=i * self.out_sample_days)
            window_start = window_end - timedelta(days=total_days)
            
            in_sample_start = window_start
            in_sample_end = window_start + timedelta(days=self.in_sample_days)
            out_sample_start = in_sample_end
            out_sample_end = window_end
            
            windows.append({
                "window": i + 1,
                "in_sample": {
                    "start": in_sample_start.strftime("%Y-%m-%d"),
                    "end": in_sample_end.strftime("%Y-%m-%d")
                },
                "out_sample": {
                    "start": out_sample_start.strftime("%Y-%m-%d"),
                    "end": out_sample_end.strftime("%Y-%m-%d")
                }
            })
        
        # Reverse windows to process oldest first
        windows.reverse()
        
        # Run optimization for each window
        results = []
        
        for window in windows:
            logger.info(f"Processing window {window['window']}")
            
            # Get in-sample data
            in_sample_df = get_data_fallback(
                symbol=SYMBOL,
                timeframe=self.timeframe,
                start_date=window["in_sample"]["start"],
                end_date=window["in_sample"]["end"]
            )
            
            # Get out-of-sample data
            out_sample_df = get_data_fallback(
                symbol=SYMBOL,
                timeframe=self.timeframe,
                start_date=window["out_sample"]["start"],
                end_date=window["out_sample"]["end"]
            )
            
            # Skip window if not enough data
            if len(in_sample_df) < 100 or len(out_sample_df) < 30:
                logger.warning(f"Not enough data for window {window['window']}, skipping")
                continue
            
            # Create optimizer for this window
            optimizer = StrategyOptimizer(
                strategy_name=self.strategy_name,
                study_name=f"{self.strategy_name}_wfo_{window['window']}",
                trials=self.trials_per_window,
                period_days=self.in_sample_days,
                timeframe=self.timeframe
            )
            
            # Run in-sample optimization
            in_sample_result = optimizer.optimize()
            best_params = in_sample_result["best_params"]
            
            # Update strategy with best parameters
            self.strategy.update_parameters(best_params)
            
            # Run out-of-sample backtest
            simulator = ExecutionSimulator(symbol=SYMBOL)
            out_sample_df = simulator.simulate_backtest(out_sample_df, self.strategy)
            
            # Get performance metrics
            performance = simulator.get_performance_summary()
            
            # Add to results
            window_result = {
                "window": window["window"],
                "in_sample": window["in_sample"],
                "out_sample": window["out_sample"],
                "best_params": best_params,
                "in_sample_value": in_sample_result["best_value"],
                "out_sample_performance": {
                    "total_trades": performance["total_trades"],
                    "win_rate": performance["win_rate"],
                    "profit_factor": performance["profit_factor"],
                    "total_return_pct": performance["total_return_pct"],
                    "max_drawdown_pct": performance["max_drawdown_pct"],
                    "sharpe_ratio": performance.get("sharpe_ratio", 0)
                }
            }
            
            results.append(window_result)
        
        # Calculate overall metrics
        total_trades = sum(r["out_sample_performance"]["total_trades"] for r in results)
        avg_win_rate = np.mean([r["out_sample_performance"]["win_rate"] for r in results])
        avg_profit_factor = np.mean([r["out_sample_performance"]["profit_factor"] for r in results])
        avg_return = np.mean([r["out_sample_performance"]["total_return_pct"] for r in results])
        avg_drawdown = np.mean([r["out_sample_performance"]["max_drawdown_pct"] for r in results])
        
        # Find the most robust parameters
        # We want parameters that work well across multiple windows
        param_consistency = self._analyze_parameter_consistency([r["best_params"] for r in results])
        
        # Calculate consistency score for each window's parameters
        for i, window_result in enumerate(results):
            consistency_score = self._calculate_parameter_consistency_score(
                window_result["best_params"],
                param_consistency
            )
            results[i]["consistency_score"] = consistency_score
        
        # Sort results by consistency score
        results.sort(key=lambda r: r["consistency_score"], reverse=True)
        
        # Choose the most consistent parameters
        robust_params = results[0]["best_params"] if results else {}
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Create final result
        wfo_result = {
            "strategy": self.strategy_name,
            "in_sample_days": self.in_sample_days,
            "out_sample_days": self.out_sample_days,
            "n_windows": self.n_windows,
            "trials_per_window": self.trials_per_window,
            "windows": results,
            "overall_metrics": {
                "total_trades": total_trades,
                "avg_win_rate": avg_win_rate,
                "avg_profit_factor": avg_profit_factor,
                "avg_return_pct": avg_return,
                "avg_drawdown_pct": avg_drawdown
            },
            "parameter_consistency": param_consistency,
            "robust_parameters": robust_params,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save result
        result_path = os.path.join(
            self.storage_dir, 
            f"{self.strategy_name}_wfo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(result_path, "w") as f:
            json.dump(wfo_result, f, indent=4)
        
        logger.info(f"Walk-forward optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {result_path}")
        
        return wfo_result
    
    def _analyze_parameter_consistency(self, param_sets: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze the consistency of parameters across optimization windows.
        
        Args:
            param_sets (List[Dict[str, Any]]): List of parameter sets from different windows.
            
        Returns:
            Dict[str, Dict[str, float]]: Parameter consistency analysis.
        """
        result = {}
        
        # Get all parameter names
        all_params = set()
        for params in param_sets:
            all_params.update(params.keys())
        
        # Analyze each parameter
        for param_name in all_params:
            values = [params.get(param_name) for params in param_sets if param_name in params]
            
            # Skip if not enough values
            if len(values) < 2:
                continue
            
            # Convert to numeric if possible
            try:
                numeric_values = [float(v) for v in values if v is not None]
                
                if numeric_values:
                    mean = np.mean(numeric_values)
                    std = np.std(numeric_values)
                    cv = std / mean if mean != 0 else float('inf')  # Coefficient of variation
                    
                    result[param_name] = {
                        "mean": mean,
                        "std": std,
                        "cv": cv,
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "range": max(numeric_values) - min(numeric_values),
                        "values": values
                    }
                else:
                    # Non-numeric parameter
                    value_counts = {}
                    for v in values:
                        if v in value_counts:
                            value_counts[v] += 1
                        else:
                            value_counts[v] = 1
                    
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    consistency = most_common[1] / len(values)
                    
                    result[param_name] = {
                        "most_common": most_common[0],
                        "consistency": consistency,
                        "value_counts": value_counts,
                        "values": values
                    }
            
            except:
                # Non-numeric parameter
                value_counts = {}
                for v in values:
                    if v in value_counts:
                        value_counts[v] += 1
                    else:
                        value_counts[v] = 1
                
                most_common = max(value_counts.items(), key=lambda x: x[1])
                consistency = most_common[1] / len(values)
                
                result[param_name] = {
                    "most_common": most_common[0],
                    "consistency": consistency,
                    "value_counts": value_counts,
                    "values": values
                }
        
        return result
    
    def _calculate_parameter_consistency_score(self, 
                                             params: Dict[str, Any],
                                             consistency_analysis: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate a consistency score for a parameter set.
        
        Args:
            params (Dict[str, Any]): The parameter set to evaluate.
            consistency_analysis (Dict[str, Dict[str, float]]): Parameter consistency analysis.
            
        Returns:
            float: The consistency score.
        """
        score = 0.0
        count = 0
        
        for param_name, param_value in params.items():
            if param_name in consistency_analysis:
                analysis = consistency_analysis[param_name]
                
                if "mean" in analysis:
                    # Numeric parameter
                    try:
                        param_value = float(param_value)
                        mean = analysis["mean"]
                        std = analysis["std"]
                        
                        # Calculate z-score
                        z_score = abs(param_value - mean) / std if std > 0 else 0
                        
                        # Convert to a score between 0 and 1 (lower z-score is better)
                        param_score = max(0, 1 - min(z_score / 2, 1))
                        
                        score += param_score
                        count += 1
                    except:
                        pass
                else:
                    # Categorical parameter
                    if param_value == analysis.get("most_common"):
                        score += 1
                    else:
                        score += 0
                    
                    count += 1
        
        # Calculate average score
        return score / count if count > 0 else 0
    
    def get_robust_parameters(self) -> Dict[str, Any]:
        """
        Get the most robust parameters from past walk-forward optimizations.
        
        Returns:
            Dict[str, Any]: The robust parameters.
        """
        # Find the latest WFO result file
        wfo_files = [
            f for f in os.listdir(self.storage_dir)
            if f.startswith(f"{self.strategy_name}_wfo_") and f.endswith(".json")
        ]
        
        if not wfo_files:
            logger.warning(f"No walk-forward optimization results found for {self.strategy_name}")
            return {}
        
        # Sort by modification time (newest first)
        wfo_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.storage_dir, f)), reverse=True)
        
        # Load the latest result
        try:
            with open(os.path.join(self.storage_dir, wfo_files[0]), "r") as f:
                wfo_result = json.load(f)
            
            robust_params = wfo_result.get("robust_parameters", {})
            
            # Apply to strategy if parameters found
            if robust_params:
                self.strategy.update_parameters(robust_params)
                logger.info(f"Applied robust parameters to {self.strategy_name} strategy: {robust_params}")
            
            return robust_params
            
        except Exception as e:
            logger.error(f"Error loading walk-forward optimization results: {e}")
            return {}