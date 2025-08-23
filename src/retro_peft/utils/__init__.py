"""
Utility modules for retro-peft-adapters.

Provides validation, error handling, monitoring, and other utility functions.
"""

# Provide fallback implementations first
class FallbackInputValidator:
    @staticmethod
    def validate_model_name(name):
        return str(name) if name else "default"
    
    @staticmethod
    def validate_adapter_config(config):
        return config if isinstance(config, dict) else {}
    
    @staticmethod
    def validate_text_content(text, max_length=100000):
        return str(text)[:max_length]

class FallbackValidationError(Exception):
    pass

class FallbackErrorHandler:
    def __init__(self, logger=None):
        self.logger = logger

class FallbackAdapterError(Exception):
    pass

def fallback_resilient_operation(**kwargs):
    def decorator(func):
        return func
    return decorator

# Try to import real implementations, use fallbacks if failed
try:
    from .validation import InputValidator, ValidationError
except ImportError:
    InputValidator = FallbackInputValidator
    ValidationError = FallbackValidationError

try:
    from .error_handling import ErrorHandler, AdapterError, resilient_operation
except ImportError:
    ErrorHandler = FallbackErrorHandler
    AdapterError = FallbackAdapterError
    resilient_operation = fallback_resilient_operation

# Always export the main classes (real or fallback)
__all__ = [
    "InputValidator", 
    "ValidationError",
    "ErrorHandler", 
    "AdapterError", 
    "resilient_operation"
]
