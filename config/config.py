"""
Main Configuration
Contains all the configuration parameters for the trading bot.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data_files"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Trading parameters
SYMBOL = "EUR/USD"
TIMEFRAMES = ["1h", "4h", "1d"]  # Multiple timeframes for analysis
DEFAULT_TIMEFRAME = "1h"  # Default timeframe for trading

# Capital and risk management
INITIAL_CAPITAL = 10000.0  # Starting capital in USD
RISK_PER_TRADE = 0.02  # Risk 2% of capital per trade
MAX_OPEN_POSITIONS = 5  # Maximum number of open positions
POSITION_SIZE_TYPE = "atr"  # Options: "fixed", "percent", "atr"
ATR_RISK_FACTOR = 2.0  # For ATR-based position sizing

# Execution parameters
SPREAD = 0.00020  # Average EUR/USD spread in decimal (2 pips)
COMMISSION = 0.00007  # Commission in decimal (0.7 pips)
SLIPPAGE = 0.00010  # Slippage in decimal (1 pip)
EXECUTION_DELAY = 0.5  # Simulated execution delay in seconds

# Backtest parameters
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"

# Optimization parameters
OPTIMIZATION_PERIOD = 90  # Days for optimization window
OPTIMIZATION_INTERVAL = 30  # Run optimization every X days
OPTIMIZATION_TRIALS = 100  # Number of trials per optimization

# Dashboard configuration
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_RATE = 5  # Refresh rate in seconds
DASHBOARD_THEME = "dark"  # Options: "light", "dark"

# Reporting configuration
GENERATE_PDF_REPORTS = True
PDF_REPORT_FREQUENCY = "weekly"  # Options: "daily", "weekly", "monthly"

# Logging configuration
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
CONSOLE_LOGGING = True
FILE_LOGGING = True

# API configuration
API_REQUEST_TIMEOUT = 10  # Timeout for API requests in seconds
API_MAX_RETRIES = 3  # Maximum number of retries for API requests
API_RETRY_DELAY = 1  # Delay between retries in seconds

# Market hours (UTC)
FOREX_MARKET_HOURS = {
    "sunday_open": "22:00",
    "friday_close": "22:00",
    "daily_maintenance": ["00:00", "00:05"]  # Daily maintenance window
}

# Strategy defaults
DEFAULT_STRATEGY = "hybrid_voting"  # Options: "ma_crossover", "rsi_mean_reversion", "hybrid_voting"

# Alert configuration
SEND_DISCORD_ALERTS = True
ALERT_ON_TRADE = True
ALERT_ON_ERROR = True
ALERT_ON_OPTIMIZATION = True