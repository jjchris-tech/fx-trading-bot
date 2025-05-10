"""
Logging Module
Provides logging functionality for the entire application.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import coloredlogs

from config.config import LOG_LEVEL, LOGS_DIR, CONSOLE_LOGGING, FILE_LOGGING

def setup_logger(name):
    """
    Set up and configure a logger instance.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level based on config
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers by removing any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Add console handler if enabled
    if CONSOLE_LOGGING:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add colored logs for console output
        coloredlogs.install(
            level=LOG_LEVEL,
            logger=logger,
            fmt=log_format,
            field_styles={
                'asctime': {'color': 'green'},
                'levelname': {'bold': True, 'color': 'black'},
                'name': {'color': 'blue'},
                'programname': {'color': 'cyan'},
            },
            level_styles={
                'debug': {'color': 'white'},
                'info': {'color': 'green'},
                'warning': {'color': 'yellow'},
                'error': {'color': 'red'},
                'critical': {'color': 'red', 'bold': True},
            },
        )
    
    # Add file handler if enabled
    if FILE_LOGGING:
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger for imports
default_logger = setup_logger("fx_trading_bot")