"""
Security utilities and input validation for retro-peft-adapters.

Provides security measures, input sanitization, and validation
to protect against malicious inputs and ensure safe operation.
"""

import re
import hashlib
import hmac
import secrets
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import json

from .logging import get_global_logger


class InputValidator:
    """
    Comprehensive input validation and sanitization.
    """
    
    # Common patterns for validation
    PATTERNS = {
        'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.,!?]+$'),
        'model_name': re.compile(r'^[a-zA-Z0-9\-_./]+$'),
        'file_path': re.compile(r'^[a-zA-Z0-9\-_./\\:]+$'),
        'url': re.compile(r'^https?://[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]+$'),
        'config_key': re.compile(r'^[a-zA-Z0-9_]+$'),
        'prompt_text': re.compile(r'^[\w\s\-.,!?;:()\[\]{}"\'/\\@#$%^&*+=|`~<>]*$', re.UNICODE)
    }
    
    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'exec\s*\(', re.IGNORECASE),
        re.compile(r'import\s+os', re.IGNORECASE),
        re.compile(r'__import__', re.IGNORECASE),
        re.compile(r'subprocess', re.IGNORECASE),
        re.compile(r'system\s*\(', re.IGNORECASE)
    ]
    
    # Maximum lengths for different input types
    MAX_LENGTHS = {
        'prompt': 10000,
        'model_name': 200,  
        'file_path': 500,
        'config_value': 1000,
        'general_string': 5000
    }
    
    def __init__(self):
        self.logger = get_global_logger()
    
    def validate_prompt(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Validate and sanitize prompt text.
        
        Args:
            prompt: Input prompt text
            max_length: Optional max length override
            
        Returns:
            Sanitized prompt text
            
        Raises:
            ValueError: If prompt is invalid or dangerous
        """
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        
        # Check length
        max_len = max_length or self.MAX_LENGTHS['prompt']
        if len(prompt) > max_len:
            raise ValueError(f"Prompt too long: {len(prompt)} > {max_len}")
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(prompt):
                self.logger.warning(f"Dangerous pattern detected in prompt: {pattern.pattern}")
                raise ValueError("Prompt contains potentially dangerous content")
        
        # Basic sanitization
        sanitized = prompt.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized
    
    def validate_model_name(self, model_name: str) -> str:
        """
        Validate model name/path.
        
        Args:
            model_name: Model name or path
            
        Returns:
            Validated model name
            
        Raises:
            ValueError: If model name is invalid
        """
        if not isinstance(model_name, str):
            raise ValueError("Model name must be a string")
        
        if len(model_name) > self.MAX_LENGTHS['model_name']:
            raise ValueError(f"Model name too long: {len(model_name)}")
        
        # Check for path traversal attempts
        if '..' in model_name or model_name.startswith('/'):
            if not model_name.startswith('/') or not Path(model_name).exists():
                raise ValueError("Invalid model path: potential path traversal")
        
        # Validate pattern
        if not self.PATTERNS['model_name'].match(model_name):
            raise ValueError("Model name contains invalid characters")
        
        return model_name.strip()
    
    def validate_file_path(self, file_path: str, must_exist: bool = False) -> str:
        """
        Validate file path for safety.
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            
        Returns:
            Validated file path
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        
        if len(file_path) > self.MAX_LENGTHS['file_path']:
            raise ValueError(f"File path too long: {len(file_path)}")
        
        # Convert to Path object for validation
        path = Path(file_path)
        
        # Check for path traversal
        try:
            resolved = path.resolve()
            if '..' in str(path) and not str(resolved).startswith(os.getcwd()):
                raise ValueError("Path traversal detected")
        except Exception:
            raise ValueError("Invalid file path")
        
        # Check existence if required
        if must_exist and not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        return str(path)
    
    def validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        validated_config = {}
        
        for key, value in config.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(f"Configuration key must be string: {key}")
            
            if not self.PATTERNS['config_key'].match(key):
                raise ValueError(f"Invalid configuration key: {key}")
            
            # Validate value based on type
            if isinstance(value, str):
                if len(value) > self.MAX_LENGTHS['config_value']:
                    raise ValueError(f"Configuration value too long for key {key}")
                
                # Check for dangerous patterns
                for pattern in self.DANGEROUS_PATTERNS:
                    if pattern.search(value):
                        raise ValueError(f"Dangerous pattern in config value for key {key}")
                
                validated_config[key] = value.strip()
            
            elif isinstance(value, (int, float, bool)):
                validated_config[key] = value
            
            elif isinstance(value, (list, tuple)):
                # Recursively validate list items
                validated_list = []
                for item in value:
                    if isinstance(item, str):
                        validated_list.append(self.validate_prompt(item, max_length=500))
                    elif isinstance(item, (int, float, bool)):
                        validated_list.append(item)
                    else:
                        raise ValueError(f"Unsupported list item type in config key {key}")
                validated_config[key] = validated_list
            
            elif isinstance(value, dict):
                validated_config[key] = self.validate_config_dict(value)
            
            else:
                raise ValueError(f"Unsupported configuration value type for key {key}: {type(value)}")
        
        return validated_config
    
    def sanitize_for_logging(self, data: Any) -> Any:
        """
        Sanitize data for safe logging (remove sensitive information).
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data safe for logging
        """
        if isinstance(data, str):
            # Mask potential tokens/keys
            if len(data) > 20 and any(char in data for char in 'abcdef0123456789'):
                # Potential hash/token
                return data[:8] + "***" + data[-4:]
            return data
        
        elif isinstance(data, dict):
            sanitized = {}
            sensitive_keys = {'token', 'key', 'password', 'secret', 'api_key'}
            
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self.sanitize_for_logging(value)
            return sanitized
        
        elif isinstance(data, (list, tuple)):
            return [self.sanitize_for_logging(item) for item in data]
        
        else:
            return data


class SecurityManager:
    """
    Central security manager for retro-peft-adapters.
    """
    
    def __init__(self):
        self.validator = InputValidator()
        self.logger = get_global_logger()
        
        # Rate limiting storage
        self._rate_limits: Dict[str, List[float]] = {}
        
        # Security configuration
        self.config = {
            'enable_input_validation': True,
            'enable_rate_limiting': True,
            'max_requests_per_minute': 60,
            'enable_audit_logging': True,
            'block_suspicious_patterns': True
        }
    
    def validate_input(self, input_type: str, value: Any, **kwargs) -> Any:
        """
        Validate input based on type.
        
        Args:
            input_type: Type of input ('prompt', 'model_name', 'file_path', 'config')
            value: Value to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If input is invalid
        """
        if not self.config['enable_input_validation']:
            return value
        
        try:
            if input_type == 'prompt':
                return self.validator.validate_prompt(value, **kwargs)
            elif input_type == 'model_name':
                return self.validator.validate_model_name(value)
            elif input_type == 'file_path':
                return self.validator.validate_file_path(value, **kwargs)
            elif input_type == 'config':
                return self.validator.validate_config_dict(value)
            else:
                raise ValueError(f"Unknown input type: {input_type}")
        
        except ValueError as e:
            self.logger.warning(f"Input validation failed for {input_type}: {e}")
            if self.config['enable_audit_logging']:
                self._audit_log('input_validation_failed', {
                    'input_type': input_type,
                    'error': str(e),
                    'value_preview': str(value)[:100] if isinstance(value, str) else str(type(value))
                })
            raise
    
    def check_rate_limit(self, identifier: str, max_requests: Optional[int] = None) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier for rate limiting (e.g., user ID, IP)
            max_requests: Override default max requests per minute
            
        Returns:
            True if within limits, False if rate limited
        """
        if not self.config['enable_rate_limiting']:
            return True
        
        max_req = max_requests or self.config['max_requests_per_minute']
        current_time = time.time()
        
        # Initialize or get existing request times
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
        
        request_times = self._rate_limits[identifier]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if within limits
        if len(request_times) >= max_req:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            if self.config['enable_audit_logging']:
                self._audit_log('rate_limit_exceeded', {
                    'identifier': identifier,
                    'requests_in_minute': len(request_times),
                    'max_allowed': max_req
                })
            return False
        
        # Add current request
        request_times.append(current_time)
        return True
    
    def scan_for_threats(self, text: str) -> Dict[str, Any]:
        """
        Scan text for potential security threats.
        
        Args:
            text: Text to scan
            
        Returns:
            Threat analysis results
        """
        threats = {
            'detected': False,
            'threats': [],
            'risk_level': 'low',
            'details': {}
        }
        
        if not self.config['block_suspicious_patterns']:
            return threats
        
        # Check for dangerous patterns
        for pattern in self.validator.DANGEROUS_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                threats['detected'] = True
                threats['threats'].append({
                    'type': 'dangerous_pattern',
                    'pattern': pattern.pattern,
                    'matches': matches[:5]  # Limit to first 5 matches
                })
        
        # Check for suspicious keywords
        suspicious_keywords = [
            'exec', 'eval', 'import os', 'subprocess', 'system',
            '__import__', 'open(', 'file(', 'input(', 'raw_input(',
            'compile', 'reload', 'globals', 'locals', 'vars'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in suspicious_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                threats['detected'] = True
        
        if found_keywords:
            threats['threats'].append({
                'type': 'suspicious_keywords',
                'keywords': found_keywords
            })
        
        # Determine risk level
        if threats['detected']:
            if len(threats['threats']) > 2 or any('exec' in str(t) for t in threats['threats']):
                threats['risk_level'] = 'high'
            else:
                threats['risk_level'] = 'medium'
        
        return threats
    
    def secure_prompt_processing(self, prompt: str, user_id: Optional[str] = None) -> str:
        """
        Securely process a prompt with all security checks.
        
        Args:
            prompt: Input prompt
            user_id: Optional user identifier for rate limiting
            
        Returns:
            Validated and sanitized prompt
            
        Raises:
            SecurityError: If security checks fail
        """
        # Rate limiting
        identifier = user_id or 'anonymous'
        if not self.check_rate_limit(identifier):
            raise SecurityError("Rate limit exceeded")
        
        # Threat scanning
        threat_analysis = self.scan_for_threats(prompt)
        if threat_analysis['detected'] and threat_analysis['risk_level'] == 'high':
            self.logger.error(f"High-risk threat detected in prompt: {threat_analysis}")
            if self.config['enable_audit_logging']:
                self._audit_log('high_risk_threat_blocked', {
                    'user_id': user_id,
                    'threat_analysis': threat_analysis,
                    'prompt_preview': prompt[:100]
                })
            raise SecurityError("High-risk content detected")
        
        # Input validation
        validated_prompt = self.validate_input('prompt', prompt)
        
        return validated_prompt
    
    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Log security events for auditing"""
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': self.validator.sanitize_for_logging(details)
        }
        
        self.logger.info(f"Security audit: {event_type}", audit_event=audit_entry)
    
    def generate_api_key(self, length: int = 32) -> str:
        """
        Generate a secure API key.
        
        Args:
            length: Key length in bytes
            
        Returns:
            Hex-encoded API key
        """
        return secrets.token_hex(length)
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash sensitive data with salt.
        
        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for key derivation
        from hashlib import pbkdf2_hmac
        hash_bytes = pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hash_bytes.hex(), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """
        Verify data against hash.
        
        Args:
            data: Original data
            hash_value: Hash to verify against
            salt: Salt used in hashing
            
        Returns:
            True if hash matches
        """
        computed_hash, _ = self.hash_data(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


# Global security manager instance
_global_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    
    return _global_security_manager


def secure_function(input_validation: Optional[Dict[str, str]] = None):
    """
    Decorator to add security validation to functions.
    
    Args:
        input_validation: Dictionary mapping parameter names to validation types
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if input_validation:
                security_manager = get_security_manager()
                
                # Get function signature for parameter mapping
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Validate specified parameters
                for param_name, validation_type in input_validation.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if value is not None:
                            validated_value = security_manager.validate_input(
                                validation_type, value
                            )
                            bound_args.arguments[param_name] = validated_value
                
                # Call function with validated arguments
                return func(*bound_args.args, **bound_args.kwargs)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions
def validate_prompt(prompt: str) -> str:
    """Quick prompt validation"""
    return get_security_manager().validate_input('prompt', prompt)


def validate_model_name(model_name: str) -> str:
    """Quick model name validation"""
    return get_security_manager().validate_input('model_name', model_name)


def check_rate_limit(identifier: str) -> bool:
    """Quick rate limit check"""
    return get_security_manager().check_rate_limit(identifier)