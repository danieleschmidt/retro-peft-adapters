"""
Input validation and data sanitization utilities.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputValidator:
    """
    Comprehensive input validation for retro-peft components.
    
    Provides safe validation of user inputs to prevent injection attacks
    and ensure data integrity.
    """
    
    # Safe filename pattern (no path traversal)
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    
    # Safe text pattern (no control characters except newline/tab)
    SAFE_TEXT_PATTERN = re.compile(r'^[\w\s\.\,\!\?\-\(\)\'\"]+$', re.MULTILINE)
    
    @staticmethod
    def validate_model_name(name: str) -> str:
        """
        Validate model name for safety.
        
        Args:
            name: Model name to validate
            
        Returns:
            Cleaned model name
            
        Raises:
            ValidationError: If name is invalid
        """
        if not isinstance(name, str):
            raise ValidationError(f"Model name must be string, got {type(name)}")
        
        if not name or len(name.strip()) == 0:
            raise ValidationError("Model name cannot be empty")
        
        if len(name) > 100:
            raise ValidationError("Model name too long (max 100 chars)")
        
        # Remove potentially dangerous characters
        cleaned = re.sub(r'[<>"|*?]', '', name.strip())
        
        if not cleaned:
            raise ValidationError("Model name contains only invalid characters")
        
        return cleaned
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Validate file path for safety.
        
        Args:
            path: File path to validate
            must_exist: Whether file must already exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(f"Path must be string or Path, got {type(path)}")
        
        # Check for path traversal attempts
        try:
            resolved = path.resolve()
            if ".." in str(path):
                raise ValidationError("Path traversal detected in path")
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid path: {e}")
        
        # Check if file must exist
        if must_exist and not resolved.exists():
            raise ValidationError(f"Required path does not exist: {resolved}")
        
        return resolved
    
    @staticmethod
    def validate_text_content(text: str, max_length: int = 100000) -> str:
        """
        Validate and sanitize text content.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"Text must be string, got {type(text)}")
        
        if len(text) > max_length:
            raise ValidationError(f"Text too long: {len(text)} > {max_length}")
        
        # Remove null bytes and other control characters (except \n, \t, \r)
        cleaned = text.replace('\x00', '').replace('\x08', '').replace('\x0c', '')
        
        # Check for suspicious patterns
        if re.search(r'<script|javascript:|data:|vbscript:', cleaned, re.IGNORECASE):
            raise ValidationError("Potentially malicious content detected")
        
        return cleaned
    
    @staticmethod
    def validate_numeric_parameter(
        value: Union[int, float], 
        param_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        must_be_int: bool = False
    ) -> Union[int, float]:
        """
        Validate numeric parameters.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            must_be_int: Whether value must be integer
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{param_name} must be numeric, got {type(value)}")
        
        if must_be_int and not isinstance(value, int):
            if float(value).is_integer():
                value = int(value)
            else:
                raise ValidationError(f"{param_name} must be integer, got {value}")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def validate_adapter_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate adapter configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be dictionary, got {type(config)}")
        
        validated = {}
        
        # Validate rank parameter
        if 'rank' in config:
            validated['rank'] = InputValidator.validate_numeric_parameter(
                config['rank'], 'rank', min_val=1, max_val=512, must_be_int=True
            )
        
        # Validate alpha parameter
        if 'alpha' in config:
            validated['alpha'] = InputValidator.validate_numeric_parameter(
                config['alpha'], 'alpha', min_val=0.1, max_val=1000.0
            )
        
        # Validate dropout
        if 'dropout' in config:
            validated['dropout'] = InputValidator.validate_numeric_parameter(
                config['dropout'], 'dropout', min_val=0.0, max_val=1.0
            )
        
        # Validate target_modules
        if 'target_modules' in config:
            modules = config['target_modules']
            if not isinstance(modules, list):
                raise ValidationError("target_modules must be list")
            
            validated_modules = []
            for module in modules:
                if not isinstance(module, str):
                    raise ValidationError("target_modules must contain strings")
                if len(module) > 50:
                    raise ValidationError("Module name too long")
                validated_modules.append(module)
            
            validated['target_modules'] = validated_modules
        
        # Copy other safe parameters
        safe_params = ['embedding_dim', 'retrieval_layers', 'fusion_method']
        for param in safe_params:
            if param in config:
                validated[param] = config[param]
        
        return validated
    
    @staticmethod
    def validate_document_list(documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate list of documents for indexing.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            List of validated document dictionaries
            
        Raises:
            ValidationError: If documents are invalid
        """
        if not isinstance(documents, list):
            raise ValidationError(f"Documents must be list, got {type(documents)}")
        
        if len(documents) == 0:
            raise ValidationError("Document list cannot be empty")
        
        if len(documents) > 100000:
            raise ValidationError("Too many documents (max 100,000)")
        
        validated = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                # Convert string to document format
                text = InputValidator.validate_text_content(doc, max_length=50000)
                validated.append({
                    "text": text,
                    "metadata": {"doc_id": i}
                })
            
            elif isinstance(doc, dict):
                # Validate dictionary format
                if 'text' not in doc:
                    raise ValidationError(f"Document {i} missing 'text' field")
                
                text = InputValidator.validate_text_content(
                    doc['text'], max_length=50000
                )
                
                metadata = doc.get('metadata', {})
                if not isinstance(metadata, dict):
                    raise ValidationError(f"Document {i} metadata must be dict")
                
                # Sanitize metadata
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = value
                
                clean_metadata['doc_id'] = i
                
                validated.append({
                    "text": text,
                    "metadata": clean_metadata
                })
            
            else:
                raise ValidationError(f"Document {i} must be string or dict")
        
        return validated


class ConfigValidator:
    """Validate system configuration files"""
    
    @staticmethod
    def validate_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate retrieval system configuration"""
        required_fields = ['embedding_dim', 'chunk_size']
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required config field: {field}")
        
        validated = {}
        
        # Validate embedding dimension
        validated['embedding_dim'] = InputValidator.validate_numeric_parameter(
            config['embedding_dim'], 'embedding_dim', 
            min_val=64, max_val=4096, must_be_int=True
        )
        
        # Validate chunk size
        validated['chunk_size'] = InputValidator.validate_numeric_parameter(
            config['chunk_size'], 'chunk_size',
            min_val=10, max_val=2048, must_be_int=True
        )
        
        # Optional parameters
        if 'overlap' in config:
            validated['overlap'] = InputValidator.validate_numeric_parameter(
                config['overlap'], 'overlap',
                min_val=0, max_val=config.get('chunk_size', 512), must_be_int=True
            )
        
        return validated