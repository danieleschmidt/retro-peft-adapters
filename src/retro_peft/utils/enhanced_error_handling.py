"""
Generation 2: Enhanced Error Handling and Recovery System

This module provides a comprehensive, production-grade error handling system
with advanced circuit breakers, retry mechanisms, graceful degradation,
and comprehensive error tracking and analysis.
"""

import asyncio
import logging
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Type, Union

try:
    import psutil
except ImportError:
    psutil = None


# Enhanced Exception Hierarchy
class RetroAdapterError(Exception):
    """Base exception for retro adapter errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now()
        self.error_id = str(uuid.uuid4())


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


class TimeoutError(RetroAdapterError):
    """Raised when operations timeout"""
    pass


class ResourceExhaustionError(RetroAdapterError):
    """Raised when system resources are exhausted"""
    pass


class NetworkError(RetroAdapterError):
    """Raised when network operations fail"""
    pass


class CompatibilityError(RetroAdapterError):
    """Raised when compatibility issues are detected"""
    pass


class DataIntegrityError(RetroAdapterError):
    """Raised when data integrity checks fail"""
    pass


class CircuitBreakerError(RetroAdapterError):
    """Raised when circuit breaker is open"""
    pass


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_id: str = field(default_factory=lambda: f"err_{int(time.time()*1000)}")
    correlation_id: Optional[str] = None


@dataclass
class ErrorRecord:
    """Detailed error record for tracking and analysis"""
    error: Exception
    context: ErrorContext
    severity: ErrorSeverity
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None
    retry_count: int = 0
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 1
    resolution_notes: Optional[str] = None


class RecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def __init__(self, name: str, max_attempts: int = 3, priority: int = 0):
        self.name = name
        self.max_attempts = max_attempts
        self.priority = priority  # Higher priority strategies are tried first
        
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can recover from the error"""
        raise NotImplementedError
        
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class GracefulDegradationStrategy(RecoveryStrategy):
    """Strategy for graceful degradation when primary functionality fails"""
    
    def __init__(self, fallback_func: Callable, name: str = "graceful_degradation",
                 applicable_errors: Optional[Set[Type[Exception]]] = None):
        super().__init__(name, priority=1)
        self.fallback_func = fallback_func
        self.applicable_errors = applicable_errors or {Exception}
        
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy applies to the error"""
        return any(isinstance(error, err_type) for err_type in self.applicable_errors)
        
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Execute fallback function"""
        return self.fallback_func()


class RetryWithBackoffStrategy(RecoveryStrategy):
    """Strategy for retrying with exponential backoff"""
    
    def __init__(self, retryable_errors: Set[Type[Exception]], 
                 initial_delay: float = 1.0, max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0, jitter: bool = True, max_attempts: int = 3):
        super().__init__("retry_with_backoff", max_attempts, priority=2)
        self.retryable_errors = retryable_errors
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if error type is retryable"""
        return any(isinstance(error, err_type) for err_type in self.retryable_errors)
        
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Not directly used - handled by ErrorHandler retry logic"""
        raise NotImplementedError("Use ErrorHandler.retry_with_strategy instead")
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        import random
        
        delay = min(self.initial_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
            
        return delay


class CacheInvalidationStrategy(RecoveryStrategy):
    """Strategy for invalidating caches on errors"""
    
    def __init__(self, cache_invalidator: Callable, 
                 applicable_errors: Optional[Set[Type[Exception]]] = None):
        super().__init__("cache_invalidation", priority=3)
        self.cache_invalidator = cache_invalidator
        self.applicable_errors = applicable_errors or {DataIntegrityError}
        
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return any(isinstance(error, err_type) for err_type in self.applicable_errors)
        
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Invalidate cache and retry"""
        self.cache_invalidator()
        return {"cache_invalidated": True, "retry_recommended": True}


class ResourceCleanupStrategy(RecoveryStrategy):
    """Strategy for cleaning up resources on resource exhaustion"""
    
    def __init__(self, cleanup_func: Callable):
        super().__init__("resource_cleanup", priority=4)
        self.cleanup_func = cleanup_func
        
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return isinstance(error, ResourceExhaustionError)
        
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Clean up resources and allow retry"""
        self.cleanup_func()
        return {"resources_cleaned": True, "retry_recommended": True}


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with comprehensive monitoring and adaptive thresholds"""
    
    def __init__(self, name: str, failure_threshold: int = 5, timeout: int = 60,
                 expected_exception: Type[Exception] = Exception,
                 recovery_threshold: int = 3, monitoring_window: int = 300,
                 adaptive_threshold: bool = False):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.recovery_threshold = recovery_threshold
        self.monitoring_window = monitoring_window
        self.adaptive_threshold = adaptive_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.state_change_time = time.time()
        
        # Enhanced monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.recent_failures = []
        self.recent_successes = []
        
        # Adaptive threshold tracking
        self.baseline_failure_rate = 0.05  # 5% baseline failure rate
        self.adaptive_multiplier = 2.0
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker with enhanced monitoring"""
        with self._lock:
            self.total_requests += 1
            
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.state_change_time = time.time()
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    self.logger.debug(f"Circuit breaker {self.name} is OPEN, rejecting request")
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Last failure: {self.last_failure_time}",
                        context={"circuit_breaker": self.name, "state": self.state}
                    )
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._on_success(execution_time)
            return result
        
        except self.expected_exception as e:
            execution_time = time.time() - start_time
            self._on_failure(execution_time, e)
            raise e
        except Exception as e:
            # Unexpected exception - don't count towards circuit breaker
            execution_time = time.time() - start_time
            self.logger.warning(f"Unexpected exception in circuit breaker {self.name}: {e}")
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        
        return (time.time() - self.last_failure_time) >= self.timeout
    
    def _on_success(self, execution_time: float):
        """Handle successful execution"""
        with self._lock:
            current_time = time.time()
            
            self.total_successes += 1
            self.success_count += 1
            self.last_success_time = current_time
            
            # Track recent successes
            self.recent_successes.append({
                'timestamp': current_time,
                'execution_time': execution_time
            })
            
            # Clean old successes
            cutoff_time = current_time - self.monitoring_window
            self.recent_successes = [
                s for s in self.recent_successes if s['timestamp'] > cutoff_time
            ]
            
            if self.state == "HALF_OPEN":
                if self.success_count >= self.recovery_threshold:
                    self.state = "CLOSED"
                    self.state_change_time = current_time
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info(
                        f"Circuit breaker {self.name} transitioned to CLOSED after "
                        f"{self.recovery_threshold} successful attempts"
                    )
            elif self.state == "CLOSED":
                # Reset failure count on success in closed state
                self.failure_count = max(0, self.failure_count - 1)
                
            # Update adaptive threshold if enabled
            if self.adaptive_threshold:
                self._update_adaptive_threshold()
    
    def _on_failure(self, execution_time: float, exception: Exception):
        """Handle failed execution"""
        with self._lock:
            current_time = time.time()
            
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = current_time
            self.success_count = 0  # Reset success count on failure
            
            # Track recent failures for analysis
            self.recent_failures.append({
                'timestamp': current_time,
                'exception': type(exception).__name__,
                'execution_time': execution_time,
                'message': str(exception)
            })
            
            # Clean old failures outside monitoring window
            cutoff_time = current_time - self.monitoring_window
            self.recent_failures = [
                f for f in self.recent_failures if f['timestamp'] > cutoff_time
            ]
            
            # Determine current failure threshold (adaptive or fixed)
            current_threshold = self._get_current_threshold()
            
            # State transitions
            if self.state == "CLOSED" and self.failure_count >= current_threshold:
                self.state = "OPEN"
                self.state_change_time = current_time
                self.logger.warning(
                    f"Circuit breaker {self.name} opened due to {self.failure_count} failures. "
                    f"Threshold: {current_threshold}"
                )
            elif self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.state_change_time = current_time
                self.logger.warning(f"Circuit breaker {self.name} re-opened due to failure in HALF_OPEN state")
    
    def _get_current_threshold(self) -> int:
        """Get current failure threshold (adaptive or fixed)"""
        if not self.adaptive_threshold:
            return self.failure_threshold
        
        # Calculate adaptive threshold based on recent failure rate
        recent_failure_rate = len(self.recent_failures) / max(1, len(self.recent_failures) + len(self.recent_successes))
        
        if recent_failure_rate > self.baseline_failure_rate * self.adaptive_multiplier:
            # Lower threshold during high failure periods
            return max(1, int(self.failure_threshold * 0.7))
        else:
            return self.failure_threshold
    
    def _update_adaptive_threshold(self):
        """Update adaptive threshold based on historical data"""
        if len(self.recent_failures) + len(self.recent_successes) < 10:
            return  # Need more data
        
        total_recent = len(self.recent_failures) + len(self.recent_successes)
        recent_failure_rate = len(self.recent_failures) / total_recent
        
        # Update baseline if we have enough stable data
        if total_recent > 100 and abs(recent_failure_rate - self.baseline_failure_rate) < 0.01:
            # Stable period, update baseline
            self.baseline_failure_rate = recent_failure_rate * 0.1 + self.baseline_failure_rate * 0.9
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics"""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.state_change_time
            
            return {
                'name': self.name,
                'state': self.state,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_requests': self.total_requests,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'failure_rate': self.total_failures / max(1, self.total_requests),
                'success_rate': self.total_successes / max(1, self.total_requests),
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'state_uptime': uptime,
                'recent_failures_count': len(self.recent_failures),
                'recent_successes_count': len(self.recent_successes),
                'failure_threshold': self.failure_threshold,
                'current_threshold': self._get_current_threshold(),
                'recovery_threshold': self.recovery_threshold,
                'timeout': self.timeout,
                'adaptive_threshold_enabled': self.adaptive_threshold,
                'baseline_failure_rate': self.baseline_failure_rate
            }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.state = "CLOSED"
            self.state_change_time = time.time()
            self.recent_failures.clear()
            self.recent_successes.clear()
            self.logger.info(f"Circuit breaker {self.name} manually reset")
    
    def force_open(self):
        """Manually open the circuit breaker"""
        with self._lock:
            self.state = "OPEN"
            self.state_change_time = time.time()
            self.last_failure_time = time.time()
            self.logger.warning(f"Circuit breaker {self.name} manually opened")


class ProductionErrorHandler:
    """Production-grade error handler with comprehensive recovery and monitoring"""
    
    def __init__(self, logger: Optional[logging.Logger] = None,
                 enable_telemetry: bool = True, max_error_records: int = 10000,
                 enable_adaptive_recovery: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.error_records: Dict[str, ErrorRecord] = {}
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.enable_telemetry = enable_telemetry
        self.max_error_records = max_error_records
        self.enable_adaptive_recovery = enable_adaptive_recovery
        
        self._lock = threading.RLock()
        
        # Performance metrics
        self.total_errors_handled = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_strategy_stats = {}
        
        # Setup default recovery strategies
        self._setup_default_strategies()
        
    def _setup_default_strategies(self):
        """Setup default recovery strategies"""
        # Network error retry strategy
        network_retry = RetryWithBackoffStrategy(
            retryable_errors={NetworkError, ConnectionError, TimeoutError},
            initial_delay=0.5,
            max_delay=30.0,
            max_attempts=3
        )
        self.register_recovery_strategy(network_retry)
        
        # Resource exhaustion cleanup strategy
        if psutil:
            def cleanup_memory():
                import gc
                gc.collect()
                return True
                
            resource_cleanup = ResourceCleanupStrategy(cleanup_memory)
            self.register_recovery_strategy(resource_cleanup)
        
        # General graceful degradation
        general_fallback = GracefulDegradationStrategy(
            fallback_func=lambda: {"status": "degraded", "mode": "fallback"},
            applicable_errors={Exception}
        )
        self.register_recovery_strategy(general_fallback)
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a recovery strategy"""
        with self._lock:
            self.recovery_strategies.append(strategy)
            # Sort by priority (higher first)
            self.recovery_strategies.sort(key=lambda s: s.priority, reverse=True)
            self.recovery_strategy_stats[strategy.name] = {
                'attempts': 0, 'successes': 0, 'failures': 0
            }
            self.logger.info(f"Registered recovery strategy: {strategy.name} (priority: {strategy.priority})")
    
    def register_circuit_breaker(self, name: str, **kwargs) -> EnhancedCircuitBreaker:
        """Register an enhanced circuit breaker"""
        with self._lock:
            breaker = EnhancedCircuitBreaker(name=name, **kwargs)
            self.circuit_breakers[name] = breaker
            self.logger.info(f"Registered enhanced circuit breaker: {name}")
            return breaker
    
    def handle_error(self, error: Exception, context: ErrorContext,
                    reraise: bool = True, attempt_recovery: bool = True) -> Optional[Any]:
        """Handle an error with comprehensive tracking and recovery"""
        with self._lock:
            self.total_errors_handled += 1
            
            # Create or update error record
            error_key = self._create_error_key(error, context)
            
            if error_key in self.error_records:
                record = self.error_records[error_key]
                record.occurrence_count += 1
                record.last_occurrence = datetime.now()
            else:
                # Determine error severity
                severity = self._determine_error_severity(error, context)
                
                record = ErrorRecord(
                    error=error,
                    context=context,
                    severity=severity
                )
                self.error_records[error_key] = record
                
                # Cleanup old records if needed
                if len(self.error_records) > self.max_error_records:
                    self._cleanup_old_error_records()
        
        # Enhanced logging with context
        self._log_error(error, record)
        
        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery:
            recovery_result = self._attempt_comprehensive_recovery(error, context, record)
            
        # Update recovery metrics
        with self._lock:
            if recovery_result is not None:
                self.successful_recoveries += 1
                record.recovery_successful = True
            elif attempt_recovery:
                self.failed_recoveries += 1
                record.recovery_attempted = True
        
        # Send telemetry if enabled
        if self.enable_telemetry:
            self._send_error_telemetry(error, record, recovery_result)
        
        # Reraise if requested and no successful recovery
        if reraise and recovery_result is None:
            raise error
        
        return recovery_result
    
    def _attempt_comprehensive_recovery(self, error: Exception, context: ErrorContext, 
                                      record: ErrorRecord) -> Optional[Any]:
        """Attempt comprehensive recovery using all registered strategies"""
        
        # Try recovery strategies in priority order
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error, context):
                try:
                    self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
                    
                    # Track strategy usage
                    with self._lock:
                        self.recovery_strategy_stats[strategy.name]['attempts'] += 1
                    
                    # Special handling for retry strategies
                    if isinstance(strategy, RetryWithBackoffStrategy):
                        result = self._execute_retry_strategy(error, context, strategy, record)
                    else:
                        result = strategy.attempt_recovery(error, context)
                    
                    if result is not None:
                        self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                        
                        # Update records
                        record.recovery_strategy = strategy.name
                        with self._lock:
                            self.recovery_strategy_stats[strategy.name]['successes'] += 1
                        
                        return result
                        
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy {strategy.name} failed: {recovery_error}"
                    )
                    with self._lock:
                        self.recovery_strategy_stats[strategy.name]['failures'] += 1
        
        return None
    
    def _execute_retry_strategy(self, error: Exception, context: ErrorContext,
                              strategy: RetryWithBackoffStrategy, record: ErrorRecord) -> Optional[Any]:
        """Execute retry strategy - placeholder for external retry logic"""
        # This indicates that retry should be handled by the caller
        record.retry_count += 1
        return None
    
    def _create_error_key(self, error: Exception, context: ErrorContext) -> str:
        """Create a unique key for error tracking"""
        return f"{type(error).__name__}:{context.operation}:{context.component}"
    
    def _determine_error_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine the severity of an error"""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, ResourceExhaustionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (NetworkError, TimeoutError, ConnectionError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ValidationError, ConfigurationError)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _log_error(self, error: Exception, record: ErrorRecord):
        """Enhanced error logging with full context"""
        log_data = {
            "error_id": record.context.error_id,
            "correlation_id": record.context.correlation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": record.context.operation,
            "component": record.context.component,
            "severity": record.severity.value,
            "occurrence_count": record.occurrence_count,
            "request_id": record.context.request_id,
            "user_id": record.context.user_id,
            "metadata": record.context.metadata,
            "traceback": traceback.format_exc(),
            "recovery_attempted": record.recovery_attempted,
            "recovery_successful": record.recovery_successful
        }
        
        # Log based on severity
        if record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR in {record.context.operation}", extra=log_data)
        elif record.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR in {record.context.operation}", extra=log_data)
        elif record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR in {record.context.operation}", extra=log_data)
        else:
            self.logger.info(f"LOW SEVERITY ERROR in {record.context.operation}", extra=log_data)
    
    def _send_error_telemetry(self, error: Exception, record: ErrorRecord, recovery_result: Any):
        """Send error telemetry data"""
        # Placeholder for telemetry integration
        telemetry_data = {
            "error_id": record.context.error_id,
            "error_type": type(error).__name__,
            "severity": record.severity.value,
            "recovery_successful": recovery_result is not None,
            "component": record.context.component,
            "operation": record.context.operation
        }
        
        # In a real implementation, this would send to monitoring systems
        self.logger.debug(f"Error telemetry: {telemetry_data}")
    
    def _cleanup_old_error_records(self):
        """Remove oldest error records to maintain size limit"""
        if len(self.error_records) <= self.max_error_records:
            return
            
        # Sort by first occurrence and remove oldest 25%
        sorted_records = sorted(
            self.error_records.items(),
            key=lambda x: x[1].first_occurrence
        )
        
        records_to_remove = len(sorted_records) - int(self.max_error_records * 0.75)
        for i in range(records_to_remove):
            error_key = sorted_records[i][0]
            del self.error_records[error_key]
            
        self.logger.info(f"Cleaned up {records_to_remove} old error records")
    
    @contextmanager
    def error_boundary(self, operation: str, component: str,
                      request_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      correlation_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Context manager for error boundaries with comprehensive tracking"""
        context = ErrorContext(
            operation=operation,
            component=component,
            request_id=request_id,
            user_id=user_id,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        start_time = time.time()
        try:
            self.logger.debug(f"Entering error boundary: {operation} in {component}")
            yield context
            duration = time.time() - start_time
            self.logger.debug(f"Exiting error boundary: {operation} completed in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            context.metadata["duration_seconds"] = duration
            self.handle_error(e, context, reraise=True)
    
    def execute_with_circuit_breaker(self, name: str, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if name not in self.circuit_breakers:
            self.register_circuit_breaker(name)
            
        breaker = self.circuit_breakers[name]
        return breaker.call(func, *args, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics"""
        with self._lock:
            stats = {
                "error_handling": {
                    "total_errors_handled": self.total_errors_handled,
                    "successful_recoveries": self.successful_recoveries,
                    "failed_recoveries": self.failed_recoveries,
                    "recovery_success_rate": (
                        self.successful_recoveries / max(1, self.successful_recoveries + self.failed_recoveries)
                    ),
                    "unique_error_types": len(self.error_records)
                },
                "recovery_strategies": dict(self.recovery_strategy_stats),
                "circuit_breakers": {
                    name: breaker.get_metrics()
                    for name, breaker in self.circuit_breakers.items()
                },
                "error_breakdown": self._get_error_breakdown(),
                "system_health": self._get_system_health()
            }
            
            return stats
    
    def _get_error_breakdown(self) -> Dict[str, Any]:
        """Get detailed error breakdown"""
        severity_counts = {severity.value: 0 for severity in ErrorSeverity}
        component_counts = {}
        operation_counts = {}
        
        for record in self.error_records.values():
            severity_counts[record.severity.value] += record.occurrence_count
            
            component = record.context.component
            component_counts[component] = component_counts.get(component, 0) + record.occurrence_count
            
            operation = record.context.operation
            operation_counts[operation] = operation_counts.get(operation, 0) + record.occurrence_count
        
        # Most frequent errors
        frequent_errors = sorted(
            self.error_records.values(),
            key=lambda x: x.occurrence_count,
            reverse=True
        )[:10]
        
        return {
            "by_severity": severity_counts,
            "by_component": component_counts,
            "by_operation": operation_counts,
            "most_frequent": [
                {
                    "error_type": type(record.error).__name__,
                    "operation": record.context.operation,
                    "component": record.context.component,
                    "count": record.occurrence_count,
                    "severity": record.severity.value
                }
                for record in frequent_errors
            ]
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        if not psutil:
            return {"status": "monitoring_unavailable"}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "available_memory_mb": memory.available / 1024 / 1024,
                "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "degraded"
            }
        except Exception as e:
            return {"status": "health_check_failed", "error": str(e)}


# Enhanced decorators
def production_resilient(
    operation: str = "",
    component: str = "unknown",
    max_retries: int = 3,
    retry_exceptions: tuple = (Exception,),
    default_return: Any = None,
    enable_circuit_breaker: bool = True,
    circuit_breaker_name: Optional[str] = None,
    timeout_seconds: Optional[float] = None
):
    """
    Production-grade resilient decorator with comprehensive error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ProductionErrorHandler()
            operation_name = operation or func.__name__
            cb_name = circuit_breaker_name or f"{component}:{operation_name}"
            
            # Extract context from kwargs
            request_id = kwargs.pop('_request_id', None)
            user_id = kwargs.pop('_user_id', None)
            correlation_id = kwargs.pop('_correlation_id', None)
            
            # Create error context
            context = ErrorContext(
                operation=operation_name,
                component=component,
                request_id=request_id,
                user_id=user_id,
                correlation_id=correlation_id
            )
            
            # Timeout wrapper if specified
            def execute_func():
                if timeout_seconds:
                    return _execute_with_timeout(func, timeout_seconds, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            # Retry logic with backoff
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    if enable_circuit_breaker:
                        return error_handler.execute_with_circuit_breaker(
                            cb_name, execute_func
                        )
                    else:
                        return execute_func()
                        
                except Exception as e:
                    last_exception = e
                    context.metadata["attempt"] = attempt + 1
                    
                    # Check if should retry
                    if attempt < max_retries and isinstance(e, retry_exceptions):
                        # Calculate backoff delay
                        delay = min(2 ** attempt, 60)  # Max 60 seconds
                        error_handler.logger.warning(
                            f"Retry attempt {attempt + 1}/{max_retries} for {operation_name} "
                            f"after {delay}s delay: {e}"
                        )
                        time.sleep(delay)
                        continue
                    
                    # Handle error
                    recovery_result = error_handler.handle_error(
                        e, context, reraise=False, attempt_recovery=True
                    )
                    
                    if recovery_result is not None:
                        return recovery_result
                    
                    break
            
            # All retries failed
            if last_exception:
                error_handler.logger.error(
                    f"All retries exhausted for {operation_name}: {last_exception}"
                )
            
            return default_return
        
        return wrapper
    return decorator


def _execute_with_timeout(func: Callable, timeout_seconds: float, *args, **kwargs):
    """Execute function with timeout"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel timeout
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> ProductionErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ProductionErrorHandler()
    
    return _global_error_handler


# Async versions
async def async_production_resilient(
    func: Callable[..., Awaitable[Any]],
    operation: str = "",
    component: str = "unknown",
    max_retries: int = 3,
    retry_exceptions: tuple = (Exception,),
    default_return: Any = None,
    initial_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """Async version of production resilient execution"""
    import random
    
    error_handler = get_global_error_handler()
    operation_name = operation or func.__name__
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            # Check if should retry
            if attempt < max_retries and isinstance(e, retry_exceptions):
                delay = min(initial_delay * (2 ** attempt), max_delay)
                delay = delay * (0.5 + random.random() * 0.5)  # Add jitter
                
                error_handler.logger.warning(
                    f"Async retry attempt {attempt + 1}/{max_retries} for {operation_name} "
                    f"after {delay:.1f}s delay: {e}"
                )
                
                await asyncio.sleep(delay)
                continue
            
            # Handle error
            context = ErrorContext(operation=operation_name, component=component)
            recovery_result = error_handler.handle_error(
                e, context, reraise=False, attempt_recovery=True
            )
            
            if recovery_result is not None:
                return recovery_result
            
            break
    
    return default_return


# Export all classes and functions
__all__ = [
    # Exceptions
    'RetroAdapterError', 'ConfigurationError', 'RetrievalError', 'AdapterError',
    'ValidationError', 'TimeoutError', 'ResourceExhaustionError', 'NetworkError',
    'CompatibilityError', 'DataIntegrityError', 'CircuitBreakerError',
    
    # Core classes
    'ErrorSeverity', 'ErrorContext', 'ErrorRecord',
    'RecoveryStrategy', 'GracefulDegradationStrategy', 'RetryWithBackoffStrategy',
    'CacheInvalidationStrategy', 'ResourceCleanupStrategy',
    'EnhancedCircuitBreaker', 'ProductionErrorHandler',
    
    # Decorators
    'production_resilient',
    
    # Functions
    'get_global_error_handler', 'async_production_resilient'
]