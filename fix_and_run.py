"""
Fix and Run Script
Validates the environment and runs the trading bot with proper configurations.
"""
import os
import sys
import subprocess
import time
from datetime import datetime, timedelta
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_and_run.log')
    ]
)

logger = logging.getLogger('fix_and_run')

def check_python_version():
    """Check if Python version is 3.8+."""
    logger.info("Checking Python version...")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python 3.8+ is required. You have Python {major}.{minor}")
        return False
    
    logger.info(f"Python {major}.{minor} detected ✓")
    return True

def check_dependencies():
    """Check and install required packages."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "plotly", "dash", "optuna",
        "scipy", "requests", "python-dotenv", "ta", "fpdf2", "discord-webhook",
        "pytest", "streamlit", "joblib", "scikit-learn", "tabulate", "coloredlogs"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"Package {package} is installed ✓")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        
        try:
            logger.info("Installing missing packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("All dependencies installed successfully ✓")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install missing packages: {e}")
            return False
    else:
        logger.info("All required packages are installed ✓")
    
    return True

def fix_parameter_file():
    """Fix the parameters.py file to use more sensitive parameters."""
    logger.info("Fixing parameters.py file...")
    
    params_path = "parameters.py"
    backup_path = "parameters.py.bak"
    
    # Check if the file exists in the config directory
    if not os.path.exists(params_path):
        # Check in the config module directory
        params_path = os.path.join("config", "parameters.py")
        backup_path = os.path.join("config", "parameters.py.bak")
        
        if not os.path.exists(params_path):
            logger.error(f"Could not find parameters.py")
            return False
    
    # Create a backup of the original file
    try:
        with open(params_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of parameters.py at {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False
    
    # Update the parameters
    try:
        with open("config/parameters.py", "w") as f:
            f.write('''"""
Strategy Parameters
Default parameters for each trading strategy.
These can be optimized by the auto-optimization framework.
"""

# Moving Average Crossover Strategy
MA_CROSSOVER_PARAMS = {
    "fast_ma_period": 8,          # Decreased from 10 for more signals
    "slow_ma_period": 21,         # Decreased from 50 for more signals
    "ma_type": "EMA",             # Options: "SMA", "EMA", "WMA"
    "signal_threshold": 0.0,      # Threshold for signal strength
    "exit_after_bars": 20,        # Exit after n bars if not stopped out
    "profit_target": 0.004,       # Take profit at 0.4% (40 pips for EUR/USD)
    "stop_loss": 0.002,           # Stop loss at 0.2% (20 pips for EUR/USD)
    "enable_trailing_stop": True, # Enable trailing stop
    "trailing_stop_activation": 0.001,  # Activate trailing stop after 0.1% profit
    "trailing_stop_distance": 0.0015,   # Trailing stop distance of 0.15%
}

# RSI Mean Reversion Strategy
RSI_MEAN_REVERSION_PARAMS = {
    "rsi_period": 7,              # Decreased from 14 for more signals
    "overbought_threshold": 70,   # Overbought threshold for RSI
    "oversold_threshold": 30,     # Oversold threshold for RSI
    "mean_period": 100,           # Decreased from 200 for more signals
    "counter_trend_factor": 1.0,  # Reduced from 1.5 to be less restrictive
    "exit_rsi_level": 50,         # Exit when RSI crosses this level
    "profit_target": 0.004,       # Take profit at 0.4% (40 pips for EUR/USD)
    "stop_loss": 0.002,           # Stop loss at 0.2% (20 pips for EUR/USD)
    "max_holding_period": 15,     # Maximum holding period in bars
    "confirmation_candles": 1,    # Reduced from 2 to generate more signals
}

# Hybrid Voting Strategy
HYBRID_VOTING_PARAMS = {
    # Component strategies and their weights
    "ma_crossover_weight": 1.0,
    "rsi_mean_reversion_weight": 1.0,
    
    # Additional parameters
    "voting_threshold": 0.3,      # Reduced from 0.6 to generate more signals
    "minimum_confirmation": 0.3,  # Reduced from 0.5 to generate more signals
    "position_sizing_factor": 1.0, # Scale position size based on vote strength
    
    # Risk management parameters
    "profit_target": 0.005,       # Take profit at 0.5% (50 pips for EUR/USD)
    "stop_loss": 0.002,           # Stop loss at 0.2% (20 pips for EUR/USD)
    "trailing_stop": True,        # Enable trailing stop
    "trailing_stop_activation": 0.002, # Activate trailing stop after 0.2% profit
    "trailing_stop_distance": 0.001,  # Trailing stop distance of 0.1%
    
    # Filter parameters
    "atr_filter_period": 14,      # ATR period for volatility filter
    "atr_filter_multiplier": 0.7, # Reduced from 1.0 to be less restrictive
    "volatility_filter": False,   # Disabled volatility filter to generate more signals
}

# Dictionary mapping strategy names to parameters
STRATEGY_PARAMS = {
    "ma_crossover": MA_CROSSOVER_PARAMS,
    "rsi_mean_reversion": RSI_MEAN_REVERSION_PARAMS,
    "hybrid_voting": HYBRID_VOTING_PARAMS,
}

# Optimization hyperparameters space
# These define the search space for the optimization framework

MA_CROSSOVER_HYPERPARAMS = {
    "fast_ma_period": {"type": "int", "low": 5, "high": 20, "step": 1},
    "slow_ma_period": {"type": "int", "low": 15, "high": 100, "step": 5},
    "ma_type": {"type": "categorical", "choices": ["SMA", "EMA", "WMA"]},
    "profit_target": {"type": "float", "low": 0.002, "high": 0.01, "step": 0.001},
    "stop_loss": {"type": "float", "low": 0.001, "high": 0.005, "step": 0.0005},
    "enable_trailing_stop": {"type": "categorical", "choices": [True, False]},
}

RSI_MEAN_REVERSION_HYPERPARAMS = {
    "rsi_period": {"type": "int", "low": 7, "high": 21, "step": 1},
    "overbought_threshold": {"type": "int", "low": 65, "high": 80, "step": 1},
    "oversold_threshold": {"type": "int", "low": 20, "high": 35, "step": 1},
    "mean_period": {"type": "int", "low": 50, "high": 200, "step": 25},
    "profit_target": {"type": "float", "low": 0.002, "high": 0.01, "step": 0.001},
    "stop_loss": {"type": "float", "low": 0.001, "high": 0.005, "step": 0.0005},
}

HYBRID_VOTING_HYPERPARAMS = {
    "ma_crossover_weight": {"type": "float", "low": 0.1, "high": 2.0, "step": 0.1},
    "rsi_mean_reversion_weight": {"type": "float", "low": 0.1, "high": 2.0, "step": 0.1},
    "voting_threshold": {"type": "float", "low": 0.3, "high": 0.7, "step": 0.05},
    "profit_target": {"type": "float", "low": 0.003, "high": 0.01, "step": 0.001},
    "stop_loss": {"type": "float", "low": 0.001, "high": 0.006, "step": 0.0005},
}

# Dictionary mapping strategy names to hyperparameter search spaces
STRATEGY_HYPERPARAMS = {
    "ma_crossover": MA_CROSSOVER_HYPERPARAMS,
    "rsi_mean_reversion": RSI_MEAN_REVERSION_HYPERPARAMS,
    "hybrid_voting": HYBRID_VOTING_HYPERPARAMS,
}''')
        logger.info("Updated parameters.py with more sensitive parameters ✓")
        return True
    except Exception as e:
        logger.error(f"Failed to update parameters.py: {e}")
        return False

def fix_simulator_file():
    """Fix the simulator.py file by patching the exit_position method and signal thresholds."""
    logger.info("Fixing simulator.py file...")
    
    simulator_path = "simulator.py"
    backup_path = "simulator.py.bak"
    
    # Check if the file exists in the execution directory
    if not os.path.exists(simulator_path):
        # Check in the execution module directory
        simulator_path = os.path.join("execution", "simulator.py")
        backup_path = os.path.join("execution", "simulator.py.bak")
        
        if not os.path.exists(simulator_path):
            logger.error(f"Could not find simulator.py")
            return False
    
    # Create a backup of the original file
    try:
        with open(simulator_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of simulator.py at {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False
    
    # Read the content of the file
    try:
        with open(simulator_path, 'r') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read simulator.py: {e}")
        return False
    
    # Find and replace the problematic capital calculation line
    if "self.capital += pnl + position_size * exit_price" in content:
        content = content.replace(
            "self.capital += pnl + position_size * exit_price",
            "self.capital += pnl  # Only add the profit/loss to capital, not the full position value"
        )
        logger.info("Fixed capital calculation in exit_position method ✓")
        
        # Also update the signal thresholds
        content = content.replace(
            "if df['signal'].iloc[i] > 0.5:",
            "if df['signal'].iloc[i] > 0.3:  # More sensitive threshold"
        )
        content = content.replace(
            "elif df['signal'].iloc[i] < -0.5:",
            "elif df['signal'].iloc[i] < -0.3:  # More sensitive threshold"
        )
        
        # Update exit signal thresholds too
        content = content.replace(
            "if df['signal'].iloc[i] < 0:",
            "if df['signal'].iloc[i] < -0.3:  # More sensitive threshold"
        )
        content = content.replace(
            "if df['signal'].iloc[i] > 0:",
            "if df['signal'].iloc[i] > 0.3:  # More sensitive threshold"
        )
        
        # Add safety checks for position sizing to enter_position
        if "# Calculate position size if not provided" in content:
            position_size_code = """
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
            position_size = 1000000  # 10 standard lots"""
            
            content = content.replace(
                """        # Calculate position size if not provided
        if position_size is None:
            position_size = calculate_position_size(
                capital=self.capital,
                risk_percentage=self.risk_per_trade,
                entry_price=entry_price,
                stop_loss=stop_loss,
                symbol=self.symbol
            )""", 
                position_size_code
            )
            logger.info("Added position size safety checks ✓")
        
        try:
            with open(simulator_path, 'w') as f:
                f.write(content)
            logger.info("Successfully updated simulator.py ✓")
            return True
        except Exception as e:
            logger.error(f"Failed to write fixed content: {e}")
            return False
    else:
        logger.info("The capital calculation in simulator.py appears to be already fixed")
        return True

def run_backtest():
    """Run a backtest with appropriate parameters."""
    logger.info("Running backtest...")
    
    # Calculate appropriate date range (6 months of data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    # Construct the command
    cmd = [
        sys.executable, "backtest.py",
        "--start-date", start_date,
        "--end-date", end_date,
        "--strategy", "hybrid_voting",
        "--report",
        "--dashboard"
    ]
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        logger.info("Backtest running, please be patient...")
        process.wait()
        
        if process.returncode == 0:
            logger.info("Backtest completed successfully ✓")
            return True
        else:
            logger.error(f"Backtest failed with return code {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        return False

def run_trading_bot():
    """Run the trading bot in live mode."""
    logger.info("Starting trading bot...")
    
    # Construct the command
    cmd = [sys.executable, "main.py"]
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait for a moment to make sure the bot starts properly
        time.sleep(10)
        
        if process.poll() is None:  # Process is still running
            logger.info("Trading bot started successfully ✓")
            logger.info("Press Ctrl+C in the bot window to stop the bot")
            return True
        else:
            logger.error(f"Trading bot exited prematurely with return code {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        return False

def check_and_create_directories():
    """Check and create required directories."""
    logger.info("Checking required directories...")
    
    required_dirs = [
        "data_files",
        "data_files/market_data_cache",
        "data_files/sentiment_cache",
        "data_files/dashboard",
        "logs",
        "reports"
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    logger.info("Directory structure looks good ✓")
    return True

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print(" EUR/USD FX Trading Bot - Quick Fix and Run ".center(60, "="))
    print("=" * 60 + "\n")
    
    # Check Python version
    if not check_python_version():
        print("\nFailed to meet requirements. Please use Python 3.8 or higher.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nFailed to install required dependencies.")
        sys.exit(1)
    
    # Check and create directories
    if not check_and_create_directories():
        print("\nFailed to create required directories.")
        sys.exit(1)
    
    # Fix simulator file
    if not fix_simulator_file():
        print("\nFailed to fix simulator.py file.")
        print("Please manually edit the exit_position method in simulator.py:")
        print('Replace: self.capital += pnl + position_size * exit_price')
        print('With:    self.capital += pnl  # Only add the profit/loss to capital')
        sys.exit(1)
    
    # Fix parameter file
    if not fix_parameter_file():
        print("\nFailed to fix parameters.py file.")
        print("The trading bot may not generate enough signals.")
    
    # Ask what the user wants to do
    print("\nWhat would you like to do?")
    print("1. Run backtest (recommended to test if everything works)")
    print("2. Start trading bot (live simulation)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        # Run backtest
        if run_backtest():
            print("\nBacktest completed successfully!")
        else:
            print("\nBacktest failed. Check the logs for details.")
    elif choice == "2":
        # Run trading bot
        if run_trading_bot():
            print("\nTrading bot started successfully!")
            print("The bot is running in a separate window.")
            print("You can close this window now.")
        else:
            print("\nTrading bot failed to start. Check the logs for details.")
    else:
        print("\nExiting...")
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()