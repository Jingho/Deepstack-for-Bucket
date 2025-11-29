"""Logging configuration and utilities."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ..config.settings import Config, get_config


def setup_logging(config: Optional[Config] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration instance. If None, uses global config.
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = get_config()
    
    # Create logger
    logger = logging.getLogger("deepstack_bucket")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.log_to_file:
        log_file = config.log_dir / "deepstack_bucket.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.log_file_max_bytes,
            backupCount=config.log_file_backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name. If None, uses "deepstack_bucket".
        
    Returns:
        Logger instance
    """
    logger_name = name or "deepstack_bucket"
    return logging.getLogger(logger_name)


# Initialize logging on module import
_logger = setup_logging()

