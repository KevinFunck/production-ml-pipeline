"""
Logging configuration for the ML Pipeline.
"""
import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from config.config import LOG_DIR, LOG_LEVEL, LOG_FORMAT

# Create logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with both console and file handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler with JSON format
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log")
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_formatter = jsonlogger.JsonFormatter()
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
