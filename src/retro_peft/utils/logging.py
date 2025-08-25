"""
Comprehensive logging system for retro-peft-adapters.

Provides structured logging with different levels, formatters,
and output destinations for development and production use.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add performance metrics if present
        if hasattr(record, "duration"):
            log_data["duration_ms"] = record.duration

        if hasattr(record, "memory_usage"):
            log_data["memory_mb"] = record.memory_usage

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}" f"{self.COLORS['RESET']}"
            )

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Build message
        message = f"[{timestamp}] {record.levelname} {record.name}: {record.getMessage()}"

        # Add context if available
        if hasattr(record, "context"):
            message += f" | Context: {record.context}"

        # Add duration if available
        if hasattr(record, "duration"):
            message += f" | Duration: {record.duration:.3f}ms"

        return message


class RetroPEFTLogger:
    """Enhanced logger with context and performance tracking"""

    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self._context = {}

    def set_context(self, **kwargs):
        """Set context fields for all subsequent log messages"""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear all context fields"""
        self._context.clear()

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context and extra fields"""
        # Remove 'level' from kwargs if present to avoid conflicts
        extra_fields = {k: v for k, v in {**self._context, **kwargs}.items() if k != 'level'}

        # Create log record with extra fields
        record = self.logger.makeRecord(self.name, level, "", 0, message, (), None)
        record.extra_fields = extra_fields

        # Add context and extra attributes to record
        for key, value in extra_fields.items():
            setattr(record, key, value)

        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, message, **kwargs)

    def log_performance(self, message: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        kwargs["duration"] = duration_ms
        self._log_with_context(logging.INFO, message, **kwargs)

    def log_memory_usage(self, message: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        kwargs["memory_usage"] = memory_mb
        self._log_with_context(logging.INFO, message, **kwargs)


def setup_logger(
    name: str = "retro_peft",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "colored",  # "colored", "json", "simple"
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
) -> RetroPEFTLogger:
    """
    Set up comprehensive logging for retro-peft-adapters.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Log format ("colored", "json", "simple")
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        enable_console: Whether to log to console

    Returns:
        Enhanced logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)

        if log_format == "colored":
            console_formatter = ColoredFormatter()
        elif log_format == "json":
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )

        # Always use JSON format for file logs
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Create enhanced logger wrapper
    retro_logger = RetroPEFTLogger(name, logger)

    # Log initial setup message
    retro_logger.info(
        "Logger initialized",
        config_level=level,
        log_file=log_file,
        log_format=log_format,
        enable_console=enable_console,
    )

    return retro_logger


def get_logger(name: str = "retro_peft") -> RetroPEFTLogger:
    """
    Get existing logger or create a default one.

    Args:
        name: Logger name

    Returns:
        Enhanced logger instance
    """
    # Check if logger already exists
    existing_logger = logging.getLogger(name)

    if existing_logger.handlers:
        # Return wrapped existing logger
        return RetroPEFTLogger(name, existing_logger)
    else:
        # Create new logger with default settings
        return setup_logger(name)


# Global logger instance
_global_logger = None


def get_global_logger() -> RetroPEFTLogger:
    """Get or create global logger instance"""
    global _global_logger

    if _global_logger is None:
        # Check environment variables for configuration
        log_level = os.getenv("RETRO_PEFT_LOG_LEVEL", "INFO")
        log_file = os.getenv("RETRO_PEFT_LOG_FILE")
        log_format = os.getenv("RETRO_PEFT_LOG_FORMAT", "colored")

        _global_logger = setup_logger(
            name="retro_peft", level=log_level, log_file=log_file, log_format=log_format
        )

    return _global_logger


# Convenience functions for quick logging
def log_info(message: str, **kwargs):
    """Quick info logging"""
    get_global_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Quick warning logging"""
    get_global_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Quick error logging"""
    get_global_logger().error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Quick debug logging"""
    get_global_logger().debug(message, **kwargs)


# Performance measurement utilities
import time
from contextlib import contextmanager


@contextmanager
def performance_timer(operation_name: str):
    """
    Context manager for measuring operation performance.
    
    Args:
        operation_name: Name of the operation being measured
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        get_global_logger().log_performance(f"Operation '{operation_name}' completed", duration)


def get_logger(name: str) -> RetroPEFTLogger:
    """
    Get logger instance for specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name=name)


def log_performance(operation: str, duration_ms: float, **kwargs):
    """Quick performance logging"""
    get_global_logger().log_performance(
        f"Performance: {operation}", duration_ms, operation=operation, **kwargs
    )


class LoggingContextManager:
    """Context manager for scoped logging with automatic cleanup"""

    def __init__(self, logger: RetroPEFTLogger, **context):
        self.logger = logger
        self.context = context
        self.original_context = None

    def __enter__(self):
        # Save original context
        self.original_context = self.logger._context.copy()

        # Set new context
        self.logger.set_context(**self.context)

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        self.logger._context = self.original_context

        # Log exception if occurred
        if exc_type is not None:
            self.logger.exception(f"Exception in logging context: {exc_type.__name__}: {exc_val}")


def with_logging_context(**context):
    """Decorator for adding logging context to functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_global_logger()
            with LoggingContextManager(logger, **context):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Performance timing decorator
def log_execution_time(operation_name: Optional[str] = None):
    """Decorator to log function execution time"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                log_performance(op_name, duration_ms)

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                get_global_logger().error(
                    f"Function {op_name} failed after {duration_ms:.3f}ms: {e}",
                    operation=op_name,
                    duration=duration_ms,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator
