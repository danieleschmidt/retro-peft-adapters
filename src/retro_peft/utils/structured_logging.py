"""
Generation 2: Production-Grade Structured Logging System

This module provides comprehensive structured logging with correlation IDs, distributed tracing,
performance monitoring, and centralized log aggregation capabilities for maximum observability.
"""

import contextvars
import json
import logging
import logging.config
import logging.handlers
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union, Callable
from pathlib import Path
import traceback

try:
    import psutil
except ImportError:
    psutil = None

# Context variables for distributed tracing
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='')
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')
operation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('operation_id', default='')
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('session_id', default='')


@dataclass
class LogContext:
    """Structured context information for logging"""
    request_id: str = ''
    user_id: str = ''
    correlation_id: str = ''
    operation_id: str = ''
    component: str = ''
    session_id: str = ''
    trace_id: str = ''
    span_id: str = ''
    parent_span_id: str = ''
    tags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    @classmethod
    def from_context_vars(cls) -> 'LogContext':
        """Create LogContext from current context variables"""
        return cls(
            request_id=request_id_var.get(''),
            user_id=user_id_var.get(''),
            correlation_id=correlation_id_var.get(''),
            operation_id=operation_id_var.get(''),
            session_id=session_id_var.get('')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty values"""
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class TraceSpan:
    """Distributed tracing span information"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = 'active'  # active, completed, failed
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = 'completed'):
        """Finish the span"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = 'info', **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def add_error(self, error: Exception):
        """Add an error to the span"""
        error_entry = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_entry)
        self.status = 'failed'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return asdict(self)


class DistributedTracer:
    """Production-grade distributed tracing implementation"""
    
    def __init__(self, service_name: str = "retro-peft-adapters"):
        self.service_name = service_name
        self._spans: Dict[str, TraceSpan] = {}
        self._active_spans: Dict[str, str] = {}  # thread_id -> span_id
        self._trace_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Span export callbacks
        self._span_exporters: List[Callable[[TraceSpan], None]] = []
        
    def add_span_exporter(self, exporter: Callable[[TraceSpan], None]):
        """Add a span exporter callback"""
        self._span_exporters.append(exporter)
        
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None,
                  tags: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new tracing span"""
        thread_id = str(threading.get_ident())
        
        with self._lock:
            # Generate IDs
            span_id = str(uuid.uuid4())
            
            # Get or create trace ID
            if parent_span_id and parent_span_id in self._spans:
                trace_id = self._spans[parent_span_id].trace_id
            else:
                # Check for existing active span in this thread
                active_span_id = self._active_spans.get(thread_id)
                if active_span_id and active_span_id in self._spans:
                    trace_id = self._spans[active_span_id].trace_id
                    parent_span_id = active_span_id
                else:
                    trace_id = str(uuid.uuid4())
            
            # Create span
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time(),
                tags=tags or {}
            )
            
            # Add service information
            span.add_tag('service.name', self.service_name)
            span.add_tag('service.version', os.getenv('SERVICE_VERSION', 'unknown'))
            
            self._spans[span_id] = span
            self._active_spans[thread_id] = span_id
            
            # Update context variables
            correlation_id_var.set(trace_id)
            operation_id_var.set(span_id)
            
            return span
    
    def finish_span(self, span_id: str, status: str = 'completed'):
        """Finish a span"""
        with self._lock:
            if span_id in self._spans:
                span = self._spans[span_id]
                span.finish(status)
                
                # Export span
                for exporter in self._span_exporters:
                    try:
                        exporter(span)
                    except Exception as e:
                        self.logger.warning(f"Failed to export span: {e}")
                
                # Remove from active spans
                thread_id = str(threading.get_ident())
                if self._active_spans.get(thread_id) == span_id:
                    del self._active_spans[thread_id]
    
    def get_active_span(self) -> Optional[TraceSpan]:
        """Get the current active span for this thread"""
        thread_id = str(threading.get_ident())
        span_id = self._active_spans.get(thread_id)
        return self._spans.get(span_id) if span_id else None
    
    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Get a span by ID"""
        return self._spans.get(span_id)
    
    def get_trace_spans(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        return [span for span in self._spans.values() if span.trace_id == trace_id]
    
    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """Get trace as hierarchical tree structure"""
        spans = self.get_trace_spans(trace_id)
        if not spans:
            return {}
        
        # Build span lookup
        span_dict = {span.span_id: span for span in spans}
        
        # Find root spans (no parent)
        root_spans = [span for span in spans if not span.parent_span_id]
        
        def build_tree(span: TraceSpan) -> Dict[str, Any]:
            children = [
                build_tree(child_span)
                for child_span in spans
                if child_span.parent_span_id == span.span_id
            ]
            
            return {
                'span': span.to_dict(),
                'children': children
            }
        
        return {
            'trace_id': trace_id,
            'service': self.service_name,
            'spans': [build_tree(root_span) for root_span in root_spans],
            'total_spans': len(spans),
            'duration_ms': max(
                (span.duration_ms or 0) for span in spans
            ) if spans else 0
        }
    
    @contextmanager
    def trace(self, operation_name: str, **tags):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, tags=tags)
        try:
            yield span
            self.finish_span(span.span_id, 'completed')
        except Exception as e:
            span.add_error(e)
            self.finish_span(span.span_id, 'failed')
            raise


# Global tracer instance
_global_tracer = DistributedTracer()


def get_global_tracer() -> DistributedTracer:
    """Get global distributed tracer"""
    return _global_tracer


class StructuredJSONFormatter(logging.Formatter):
    """Enhanced structured JSON formatter with distributed tracing"""
    
    def __init__(self, include_trace: bool = True, include_system_info: bool = True):
        super().__init__()
        self.include_trace = include_trace
        self.include_system_info = include_system_info
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Base log structure
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.get_ident(),
            'process': os.getpid()
        }
        
        # Add context variables if available
        if self.include_trace:
            context = LogContext.from_context_vars()
            context_dict = context.to_dict()
            if context_dict:
                log_data['context'] = context_dict
        
        # Add current span information
        active_span = get_global_tracer().get_active_span()
        if active_span:
            log_data['span'] = {
                'span_id': active_span.span_id,
                'trace_id': active_span.trace_id,
                'parent_span_id': active_span.parent_span_id,
                'operation': active_span.operation_name
            }
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add performance metrics
        if hasattr(record, 'duration_ms'):
            log_data['performance'] = {'duration_ms': record.duration_ms}
        
        if hasattr(record, 'memory_mb'):
            log_data['performance'] = log_data.get('performance', {})
            log_data['performance']['memory_mb'] = record.memory_mb
        
        # Add system information
        if self.include_system_info and psutil:
            try:
                process = psutil.Process()
                log_data['system'] = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'memory_rss_mb': process.memory_info().rss / 1024 / 1024
                }
            except Exception:
                pass  # Ignore system info collection errors
        
        # Add custom tags if present
        if hasattr(record, 'tags'):
            log_data['tags'] = record.tags
        
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class ProductionStructuredLogger:
    """Production-grade structured logger with comprehensive features"""
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self._context = {}
        self.tracer = get_global_tracer()
        
    def set_context(self, **kwargs):
        """Set persistent context for all log messages"""
        self._context.update(kwargs)
        
        # Update context variables
        if 'request_id' in kwargs:
            request_id_var.set(kwargs['request_id'])
        if 'user_id' in kwargs:
            user_id_var.set(kwargs['user_id'])
        if 'correlation_id' in kwargs:
            correlation_id_var.set(kwargs['correlation_id'])
        if 'session_id' in kwargs:
            session_id_var.set(kwargs['session_id'])
    
    def clear_context(self):
        """Clear all context"""
        self._context.clear()
        
        # Clear context variables
        request_id_var.set('')
        user_id_var.set('')
        correlation_id_var.set('')
        operation_id_var.set('')
        session_id_var.set('')
    
    def _log_with_context(self, level: int, message: str, 
                         tags: Optional[Dict[str, Any]] = None,
                         duration_ms: Optional[float] = None,
                         memory_mb: Optional[float] = None,
                         **kwargs):
        """Log message with context and structured fields"""
        extra_fields = {**self._context, **kwargs}
        
        # Create enhanced log record
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        
        # Add structured fields
        record.extra_fields = extra_fields
        if tags:
            record.tags = tags
        if duration_ms is not None:
            record.duration_ms = duration_ms
        if memory_mb is not None:
            record.memory_mb = memory_mb
        
        # Add to active span if available
        active_span = self.tracer.get_active_span()
        if active_span:
            active_span.add_log(message, level=logging.getLevelName(level).lower(), **extra_fields)
        
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
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def performance(self, message: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        self._log_with_context(
            logging.INFO, f"PERFORMANCE: {message}", 
            duration_ms=duration_ms, **kwargs
        )
    
    def audit(self, event: str, **kwargs):
        """Log audit event"""
        self._log_with_context(
            logging.INFO, f"AUDIT: {event}",
            tags={'audit': True, 'event': event},
            **kwargs
        )
    
    def security(self, event: str, severity: str = 'medium', **kwargs):
        """Log security event"""
        level = logging.WARNING if severity in ('high', 'critical') else logging.INFO
        self._log_with_context(
            level, f"SECURITY: {event}",
            tags={'security': True, 'severity': severity, 'event': event},
            **kwargs
        )
    
    @contextmanager
    def operation(self, operation_name: str, **tags):
        """Context manager for logging operations with tracing"""
        start_time = time.time()
        
        with self.tracer.trace(operation_name, **tags) as span:
            try:
                self.info(f"Starting operation: {operation_name}", operation=operation_name, **tags)
                yield span
                
                duration_ms = (time.time() - start_time) * 1000
                self.performance(f"Completed operation: {operation_name}", duration_ms, 
                               operation=operation_name, **tags)
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.error(f"Failed operation: {operation_name}", 
                          operation=operation_name, duration_ms=duration_ms,
                          error=str(e), **tags)
                raise


def setup_structured_logging(
    name: str = "retro_peft",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_tracing: bool = True,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
    include_system_info: bool = True
) -> ProductionStructuredLogger:
    """
    Setup production-grade structured logging
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        enable_console: Whether to enable console logging
        enable_tracing: Whether to enable distributed tracing
        max_file_size: Maximum log file size
        backup_count: Number of backup files
        include_system_info: Whether to include system metrics
        
    Returns:
        Production structured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    json_formatter = StructuredJSONFormatter(
        include_trace=enable_tracing,
        include_system_info=include_system_info
    )
    
    # Console handler with JSON formatting
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    # Create structured logger wrapper
    structured_logger = ProductionStructuredLogger(name, logger)
    
    # Setup span exporters if tracing is enabled
    if enable_tracing:
        tracer = get_global_tracer()
        
        # Log span completion
        def log_span_completion(span: TraceSpan):
            if span.status == 'completed':
                structured_logger.debug(
                    f"Span completed: {span.operation_name}",
                    span_id=span.span_id,
                    trace_id=span.trace_id,
                    duration_ms=span.duration_ms
                )
        
        tracer.add_span_exporter(log_span_completion)
    
    # Log initialization
    structured_logger.info(
        "Structured logging initialized",
        level=level,
        log_file=log_file,
        tracing_enabled=enable_tracing,
        system_info_enabled=include_system_info
    )
    
    return structured_logger


# Global structured logger
_global_structured_logger = None


def get_structured_logger() -> ProductionStructuredLogger:
    """Get global structured logger instance"""
    global _global_structured_logger
    
    if _global_structured_logger is None:
        # Get configuration from environment
        log_level = os.getenv("RETRO_PEFT_LOG_LEVEL", "INFO")
        log_file = os.getenv("RETRO_PEFT_LOG_FILE")
        enable_tracing = os.getenv("RETRO_PEFT_ENABLE_TRACING", "true").lower() == "true"
        
        _global_structured_logger = setup_structured_logging(
            level=log_level,
            log_file=log_file,
            enable_tracing=enable_tracing
        )
    
    return _global_structured_logger


# Decorators for structured logging
def trace_operation(operation_name: Optional[str] = None, **default_tags):
    """Decorator for tracing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_structured_logger()
            
            # Extract context from kwargs
            tags = {**default_tags}
            for key in list(kwargs.keys()):
                if key.startswith('_tag_'):
                    tag_name = key[5:]  # Remove '_tag_' prefix
                    tags[tag_name] = kwargs.pop(key)
            
            with logger.operation(op_name, **tags):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_performance(operation_name: Optional[str] = None, log_memory: bool = False):
    """Decorator for performance logging"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import gc
            
            logger = get_structured_logger()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Memory tracking
            if log_memory and psutil:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                log_kwargs = {'operation': op_name}
                
                if log_memory and psutil:
                    memory_after = process.memory_info().rss / 1024 / 1024
                    memory_delta = memory_after - memory_before
                    log_kwargs['memory_delta_mb'] = memory_delta
                    log_kwargs['memory_after_mb'] = memory_after
                
                logger.performance(f"Operation completed: {op_name}", duration_ms, **log_kwargs)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Operation failed: {op_name}", 
                           operation=op_name, duration_ms=duration_ms, error=str(e))
                raise
        
        return wrapper
    return decorator


@contextmanager
def logging_context(**context):
    """Context manager for setting logging context"""
    logger = get_structured_logger()
    original_context = logger._context.copy()
    
    try:
        logger.set_context(**context)
        yield logger
    finally:
        logger._context = original_context
        
        # Reset context variables if they were changed
        for key in context.keys():
            if key == 'request_id':
                request_id_var.set(original_context.get('request_id', ''))
            elif key == 'user_id':
                user_id_var.set(original_context.get('user_id', ''))
            elif key == 'correlation_id':
                correlation_id_var.set(original_context.get('correlation_id', ''))
            elif key == 'session_id':
                session_id_var.set(original_context.get('session_id', ''))


# Export all functionality
__all__ = [
    'LogContext', 'TraceSpan', 'DistributedTracer', 'StructuredJSONFormatter',
    'ProductionStructuredLogger', 'setup_structured_logging', 'get_structured_logger',
    'get_global_tracer', 'trace_operation', 'log_performance', 'logging_context',
    'request_id_var', 'user_id_var', 'correlation_id_var', 'operation_id_var', 'session_id_var'
]