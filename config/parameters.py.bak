"""
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
}