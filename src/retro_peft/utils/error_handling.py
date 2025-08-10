"""
Comprehensive error handling and recovery utilities.
"""

import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union


class RetroAdapterError(Exception):
    """Base exception for retro adapter errors"""
    pass


class ConfigurationError(RetroAdapterError):
    """Raised when configuration is invalid"""
    pass


class RetrievalError(RetroAdapterError):
    """Raised when retrieval operations fail"""
    pass


class AdapterError(RetroAdapterError):
    """Raised when adapter operations fail"""
    pass


class ValidationError(RetroAdapterError):
    """Raised when input validation fails"""
    pass


class ErrorHandler:
    """
    Centralized error handling with logging and recovery strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = {}
    
    def register_recovery_strategy(
        self, 
        error_type: Type[Exception], 
        strategy: Callable[[Exception], Any]
    ):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(
        self, 
        error: Exception, 
        context: str = "", 
        reraise: bool = True,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            reraise: Whether to reraise the exception after handling
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        error_key = f"{type(error).__name__}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        self.logger.error(
            f"Error in {context}: {type(error).__name__}: {error}",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "count": self.error_counts[error_key],
                "traceback": traceback.format_exc()
            }
        )
        
        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery:
            recovery_result = self._attempt_recovery(error, context)
        
        # Reraise if requested and no successful recovery
        if reraise and recovery_result is None:
            raise error
        
        return recovery_result
    
    def _attempt_recovery(self, error: Exception, context: str) -> Optional[Any]:
        """Attempt to recover from an error"""
        error_type = type(error)
        
        # Try specific recovery strategy first
        if error_type in self.recovery_strategies:
            try:
                result = self.recovery_strategies[error_type](error)
                self.logger.info(f"Recovery successful for {error_type.__name__} in {context}")
                return result
            except Exception as recovery_error:
                self.logger.warning(
                    f"Recovery failed for {error_type.__name__}: {recovery_error}"
                )
        
        # Try general recovery strategies
        if isinstance(error, (FileNotFoundError, OSError)):
            return self._recover_file_error(error, context)
        elif isinstance(error, (ValueError, TypeError)):
            return self._recover_validation_error(error, context)
        elif isinstance(error, ImportError):
            return self._recover_import_error(error, context)
        
        return None
    
    def _recover_file_error(self, error: Exception, context: str) -> Optional[Any]:
        """Recover from file-related errors"""
        if "index" in context.lower():
            self.logger.info("Attempting to create empty index for file error recovery")
            return {"status": "empty_index", "documents": [], "embeddings": []}
        return None
    
    def _recover_validation_error(self, error: Exception, context: str) -> Optional[Any]:
        """Recover from validation errors"""
        if "parameter" in str(error).lower():
            self.logger.info("Using default parameters for validation error recovery")
            return {"rank": 16, "alpha": 32.0, "dropout": 0.1}
        return None
    
    def _recover_import_error(self, error: Exception, context: str) -> Optional[Any]:
        """Recover from import errors"""
        self.logger.info(f"Falling back to mock implementation for import error: {error}")
        return {"mock_mode": True, "reason": str(error)}
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error count statistics"""
        return self.error_counts.copy()
    
    def reset_statistics(self):
        """Reset error count statistics"""
        self.error_counts.clear()


def safe_execute(
    func: Callable,
    error_handler: Optional[ErrorHandler] = None,
    context: str = "",
    default_return: Any = None,
    max_retries: int = 0,
    retry_exceptions: tuple = (Exception,)
) -> Any:
    """
    Safely execute a function with error handling and retries.
    
    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Context for error logging
        default_return: Default return value on failure
        max_retries: Maximum number of retries
        retry_exceptions: Exceptions that should trigger retries
        
    Returns:
        Function result or default_return on failure
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if this exception type should trigger a retry
            if attempt < max_retries and isinstance(e, retry_exceptions):
                error_handler.logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} for {context}: {e}"
                )
                continue
            
            # Handle the error
            recovery_result = error_handler.handle_error(
                e, context=context, reraise=False, attempt_recovery=True
            )
            
            if recovery_result is not None:
                return recovery_result
            
            break
    
    # If we get here, all retries failed
    if last_exception:
        error_handler.logger.error(f"All retries exhausted for {context}: {last_exception}")
    
    return default_return


def resilient_operation(
    context: str = "",
    max_retries: int = 0,
    retry_exceptions: tuple = (Exception,),
    default_return: Any = None,
    error_handler: Optional[ErrorHandler] = None
):
    """
    Decorator for making operations resilient to failures.
    
    Args:
        context: Context for error logging
        max_retries: Maximum number of retries
        retry_exceptions: Exception types that should trigger retries
        default_return: Default return value on failure
        error_handler: Error handler instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_context = context or func.__name__
            
            def execute():
                return func(*args, **kwargs)
            
            return safe_execute(
                func=execute,
                error_handler=error_handler,
                context=operation_context,
                default_return=default_return,
                max_retries=max_retries,
                retry_exceptions=retry_exceptions
            )
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascade failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        
        import time
        return (time.time() - self.last_failure_time) >= self.timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        import time
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass