"""
Utility modules for retro-peft-adapters.

Provides validation, error handling, monitoring, and other utility functions.
"""

# Import only modules that exist and work without heavy dependencies
try:
    from .validation import InputValidator, ValidationError
    _VALIDATION_AVAILABLE = True
except ImportError:
    _VALIDATION_AVAILABLE = False

try:
    from .error_handling import ErrorHandler, AdapterError, resilient_operation
    _ERROR_HANDLING_AVAILABLE = True
except ImportError:
    _ERROR_HANDLING_AVAILABLE = False

try:
    from .health_monitoring import HealthMonitor, MetricsCollector, get_health_monitor
    _HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    _HEALTH_MONITORING_AVAILABLE = False

# Basic exports
__all__ = []

if _VALIDATION_AVAILABLE:
    __all__.extend(["InputValidator", "ValidationError"])

if _ERROR_HANDLING_AVAILABLE:
    __all__.extend(["ErrorHandler", "AdapterError", "resilient_operation"])

if _HEALTH_MONITORING_AVAILABLE:
    __all__.extend(["HealthMonitor", "MetricsCollector", "get_health_monitor"])
