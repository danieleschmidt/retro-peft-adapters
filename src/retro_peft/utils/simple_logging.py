"""
Simple logging utilities without complex dependencies.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class SimpleLogger:
    """Simple logger wrapper for basic functionality"""
    
    def __init__(self, name: str = "retro_peft", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        
        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(self._format_message(message, kwargs))
    
    def _format_message(self, message: str, context: Dict[str, Any]) -> str:
        """Format message with context"""
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"{message} | {context_str}"
        return message


# Global logger instance
_global_logger = None


def get_logger(name: str = "retro_peft") -> SimpleLogger:
    """Get logger instance"""
    return SimpleLogger(name)


def get_global_logger() -> SimpleLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SimpleLogger()
    return _global_logger


@contextmanager
def performance_timer(operation_name: str):
    """Simple performance timer context manager"""
    logger = get_global_logger()
    start_time = time.time()
    
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
        duration = (time.time() - start_time) * 1000
        logger.info(f"Completed {operation_name} in {duration:.2f}ms")
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Failed {operation_name} after {duration:.2f}ms: {e}")
        raise


def log_performance(operation: str, duration_ms: float):
    """Log performance metric"""
    logger = get_global_logger()
    logger.info(f"Performance: {operation} took {duration_ms:.2f}ms")