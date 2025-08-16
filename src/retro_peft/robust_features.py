"""
Generation 2: Robust Features Implementation

This module implements comprehensive error handling, validation, logging, 
monitoring, and security features for production readiness.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union


class AdapterError(Exception):
    """Base exception for all adapter-related errors"""
    pass


class ValidationError(AdapterError):
    """Raised when input validation fails"""
    pass


class RetrievalError(AdapterError):
    """Raised when retrieval operations fail"""
    pass


class ConfigurationError(AdapterError):
    """Raised when configuration is invalid"""
    pass


class ResourceError(AdapterError):
    """Raised when resources are unavailable or exhausted"""
    pass


class RobustValidator:
    """Comprehensive input validation and sanitization"""
    
    @staticmethod
    def validate_text_input(text: str, max_length: int = 10000, min_length: int = 1) -> str:
        """Validate and sanitize text input"""
        if not isinstance(text, str):
            raise ValidationError(f"Expected string, got {type(text)}")
        
        if len(text) < min_length:
            raise ValidationError(f"Text too short: {len(text)} < {min_length}")
        
        if len(text) > max_length:
            raise ValidationError(f"Text too long: {len(text)} > {max_length}")
        
        # Remove potentially harmful characters
        sanitized = text.replace('\x00', '').replace('\r\n', '\n')
        
        return sanitized
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary"""
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be dict, got {type(config)}")
        
        required_sections = ['adapters', 'retrieval', 'training', 'inference', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValidationError(f"Missing required config section: {section}")
        
        # Validate adapter config
        adapter_config = config['adapters']
        if 'lora' in adapter_config:
            lora_config = adapter_config['lora']
            if lora_config.get('r', 0) <= 0:
                raise ValidationError("LoRA rank must be positive")
            if lora_config.get('alpha', 0) <= 0:
                raise ValidationError("LoRA alpha must be positive")
        
        return config
    
    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate file path for security"""
        if not isinstance(path, str):
            raise ValidationError(f"Path must be string, got {type(path)}")
        
        # Check for path traversal attacks
        if '..' in path or path.startswith('/'):
            raise ValidationError(f"Potentially unsafe path: {path}")
        
        return path
    
    @staticmethod
    def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters"""
        validated = {}
        
        # Validate common parameters
        if 'max_length' in params:
            max_length = params['max_length']
            if not isinstance(max_length, int) or max_length <= 0 or max_length > 8192:
                raise ValidationError(f"Invalid max_length: {max_length}")
            validated['max_length'] = max_length
        
        if 'temperature' in params:
            temp = params['temperature']
            if not isinstance(temp, (int, float)) or temp <= 0 or temp > 2.0:
                raise ValidationError(f"Invalid temperature: {temp}")
            validated['temperature'] = float(temp)
        
        if 'k' in params:
            k = params['k']
            if not isinstance(k, int) or k <= 0 or k > 100:
                raise ValidationError(f"Invalid k value: {k}")
            validated['k'] = k
        
        return validated


class RobustErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.max_retries = 3
        self.backoff_factor = 1.5
    
    @contextmanager
    def error_context(self, operation: str):
        """Context manager for error handling"""
        start_time = time.time()
        try:
            self.logger.info(f"Starting operation: {operation}")
            yield
            duration = time.time() - start_time
            self.logger.info(f"Completed {operation} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            self._record_error(operation, e)
            self.logger.error(f"Failed {operation} after {duration:.3f}s: {e}")
            raise
    
    def _record_error(self, operation: str, error: Exception):
        """Record error for monitoring"""
        if operation not in self.error_counts:
            self.error_counts[operation] = {}
        
        error_type = type(error).__name__
        self.error_counts[operation][error_type] = (
            self.error_counts[operation].get(error_type, 0) + 1
        )
    
    def retry_operation(self, func: Callable, *args, **kwargs) -> Any:
        """Retry operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, TimeoutError, ResourceError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (self.backoff_factor ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed")
            except Exception as e:
                # Don't retry for non-recoverable errors
                self.logger.error(f"Non-recoverable error: {e}")
                raise
        
        raise last_exception
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors"""
        return {
            'error_counts': dict(self.error_counts),
            'total_operations': len(self.error_counts),
            'total_errors': sum(
                sum(counts.values()) for counts in self.error_counts.values()
            )
        }


def resilient_operation(max_retries: int = 3, backoff_factor: float = 1.5):
    """Decorator for making operations resilient to failures"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = RobustErrorHandler()
            
            def operation():
                return func(*args, **kwargs)
            
            return error_handler.retry_operation(operation)
        
        return wrapper
    
    return decorator


class HealthMonitor:
    """System health monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        self.operation_counts = {}
        self.performance_metrics = {}
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        if operation not in self.operation_counts:
            self.operation_counts[operation] = {'success': 0, 'failure': 0}
        
        status = 'success' if success else 'failure'
        self.operation_counts[operation][status] += 1
        
        # Record performance metrics
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append({
            'duration': duration,
            'timestamp': time.time(),
            'success': success
        })
        
        # Keep only last 1000 metrics per operation
        if len(self.performance_metrics[operation]) > 1000:
            self.performance_metrics[operation] = self.performance_metrics[operation][-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        uptime = time.time() - self.start_time
        
        total_operations = sum(
            counts['success'] + counts['failure'] 
            for counts in self.operation_counts.values()
        )
        
        total_failures = sum(
            counts['failure'] 
            for counts in self.operation_counts.values()
        )
        
        success_rate = (
            (total_operations - total_failures) / total_operations 
            if total_operations > 0 else 1.0
        )
        
        return {
            'status': 'healthy' if success_rate > 0.95 else 'degraded' if success_rate > 0.8 else 'unhealthy',
            'uptime_seconds': uptime,
            'total_operations': total_operations,
            'success_rate': success_rate,
            'operation_counts': dict(self.operation_counts)
        }
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations"""
        if operation and operation in self.performance_metrics:
            metrics = self.performance_metrics[operation]
            durations = [m['duration'] for m in metrics if m['success']]
            
            if durations:
                return {
                    'operation': operation,
                    'total_calls': len(metrics),
                    'successful_calls': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'p95_duration': sorted(durations)[int(len(durations) * 0.95)] if durations else 0
                }
        
        # Return summary for all operations
        summary = {}
        for op_name, metrics in self.performance_metrics.items():
            durations = [m['duration'] for m in metrics if m['success']]
            if durations:
                summary[op_name] = {
                    'total_calls': len(metrics),
                    'successful_calls': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'p95_duration': sorted(durations)[int(len(durations) * 0.95)]
                }
        
        return summary


# Global health monitor instance
_global_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _global_health_monitor
    
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    
    return _global_health_monitor


def monitored_operation(operation_name: Optional[str] = None):
    """Decorator for monitoring operation performance"""
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_health_monitor()
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                monitor.record_operation(op_name, duration, success)
        
        return wrapper
    
    return decorator


class SecurityManager:
    """Security features for input validation and protection"""
    
    def __init__(self):
        self.rate_limits = {}
        self.blocked_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',              # JavaScript injection
            r'data:.*base64',            # Data URLs
            r'\.\./',                    # Path traversal
        ]
    
    def validate_input(self, input_type: str, value: Any) -> Any:
        """Validate input based on type"""
        if input_type == "text":
            return RobustValidator.validate_text_input(str(value))
        elif input_type == "config":
            return RobustValidator.validate_config(value)
        elif input_type == "file_path":
            return RobustValidator.validate_file_path(str(value))
        elif input_type == "model_params":
            return RobustValidator.validate_model_params(value)
        else:
            return value
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests outside the window
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if current_time - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.rate_limits[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def scan_for_threats(self, text: str) -> List[str]:
        """Scan text for potential security threats"""
        import re
        
        threats = []
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"Blocked pattern detected: {pattern}")
        
        return threats


# Global security manager
_global_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    
    return _global_security_manager


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker"""
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise ResourceError("Circuit breaker is open")
            else:
                self.state = 'half-open'
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """Decorator for circuit breaker protection"""
    
    def decorator(func: Callable) -> Callable:
        circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Export all robust features
__all__ = [
    # Exceptions
    'AdapterError', 'ValidationError', 'RetrievalError', 'ConfigurationError', 'ResourceError',
    
    # Validation
    'RobustValidator',
    
    # Error handling
    'RobustErrorHandler', 'resilient_operation',
    
    # Monitoring
    'HealthMonitor', 'get_health_monitor', 'monitored_operation',
    
    # Security
    'SecurityManager', 'get_security_manager',
    
    # Circuit breaker
    'CircuitBreaker', 'with_circuit_breaker'
]