"""
Configuration management system for retro-peft-adapters.

Provides flexible configuration loading, validation, and management
with support for multiple sources and environments.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, field, fields
from enum import Enum
import copy

from .logging import get_global_logger
from .security import get_security_manager


class ConfigSource(Enum):
    """Configuration source types"""
    FILE = "file"
    ENV = "environment"
    DEFAULT = "default"
    OVERRIDE = "override"


@dataclass
class AdapterConfig:
    """Configuration for adapter settings"""
    type: str = "RetroLoRA"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    retrieval_dim: int = 768
    fusion_method: str = "cross_attention"
    retrieval_weight: float = 0.3


@dataclass 
class RetrievalConfig:
    """Configuration for retrieval settings"""
    backend: str = "faiss"
    encoder: str = "sentence-transformers/all-mpnet-base-v2"
    index_path: Optional[str] = None
    chunk_size: int = 512
    overlap: int = 50
    max_retrieved_docs: int = 5
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Configuration for training settings"""
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./checkpoints"
    logging_steps: int = 100


@dataclass
class InferenceConfig:
    """Configuration for inference settings"""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    cache_size: int = 1000
    enable_caching: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings"""
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_audit_logging: bool = True
    block_suspicious_patterns: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging settings"""
    level: str = "INFO"
    format: str = "colored"
    log_file: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for monitoring settings"""
    enable_metrics: bool = True
    enable_health_checks: bool = True
    metrics_collection_interval: float = 60.0
    max_metric_points: int = 10000
    retention_hours: int = 24


@dataclass
class Config:
    """Main configuration class"""
    
    # Core settings
    model_name: str = "microsoft/DialoGPT-small"
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    # Component configurations
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    _source: ConfigSource = ConfigSource.DEFAULT
    _file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        
        for field_info in fields(self):
            if field_info.name.startswith('_'):
                continue
                
            value = getattr(self, field_info.name)
            
            if hasattr(value, 'to_dict'):
                result[field_info.name] = value.to_dict()
            elif hasattr(value, '__dict__'):
                # Handle dataclass instances
                result[field_info.name] = {
                    k: v for k, v in value.__dict__.items() 
                    if not k.startswith('_')
                }
            else:
                result[field_info.name] = value
        
        return result
    
    def update(self, other: Union[Dict[str, Any], 'Config']):
        """Update configuration with values from another config or dict"""
        if isinstance(other, dict):
            self._update_from_dict(other)
        elif isinstance(other, Config):
            self._update_from_config(other)
        else:
            raise ValueError("Can only update from dict or Config instance")
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update from dictionary"""
        for key, value in data.items():
            if key.startswith('_'):
                continue
                
            if hasattr(self, key):
                current_value = getattr(self, key)
                
                if isinstance(current_value, (AdapterConfig, RetrievalConfig, TrainingConfig, 
                                            InferenceConfig, SecurityConfig, LoggingConfig, 
                                            MonitoringConfig)):
                    # Update nested config
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if hasattr(current_value, nested_key):
                                setattr(current_value, nested_key, nested_value)
                elif isinstance(value, dict) and key == 'custom':
                    # Update custom dict
                    self.custom.update(value)
                else:
                    setattr(self, key, value)
    
    def _update_from_config(self, other: 'Config'):
        """Update from another Config instance"""
        self._update_from_dict(other.to_dict())
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'adapter.rank')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        parts = path.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'adapter.rank')
            value: Value to set
        """
        parts = path.split('.')
        current = self
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            else:
                raise ValueError(f"Cannot set path {path}: {part} not found")
        
        final_key = parts[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
        else:
            raise ValueError(f"Cannot set path {path}: {final_key} not found")


class ConfigManager:
    """
    Configuration manager with support for multiple sources and validation.
    """
    
    def __init__(self):
        self.logger = get_global_logger()
        self.security_manager = get_security_manager()
        
        # Default config search paths
        self.search_paths = [
            Path.cwd() / "retro_peft_config.yaml",
            Path.cwd() / "retro_peft_config.json", 
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.json",
            Path.home() / ".retro_peft" / "config.yaml",
            Path.home() / ".config" / "retro_peft" / "config.yaml",
        ]
        
        # Environment variable prefix
        self.env_prefix = "RETRO_PEFT_"
    
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        load_from_env: bool = True,
        validate: bool = True
    ) -> Config:
        """
        Load configuration from multiple sources.
        
        Args:
            config_path: Optional path to config file
            config_dict: Optional config dictionary
            load_from_env: Whether to load from environment variables
            validate: Whether to validate configuration
            
        Returns:
            Loaded configuration
        """
        # Start with default config
        config = Config()
        
        # Load from file
        if config_path:
            file_config = self._load_from_file(config_path)
            config.update(file_config)
            config._file_path = str(config_path)
            config._source = ConfigSource.FILE
        else:
            # Search for config files
            for search_path in self.search_paths:
                if search_path.exists():
                    self.logger.info(f"Found config file: {search_path}")
                    file_config = self._load_from_file(search_path)
                    config.update(file_config)
                    config._file_path = str(search_path)
                    config._source = ConfigSource.FILE
                    break
        
        # Load from dictionary
        if config_dict:
            config.update(config_dict)
        
        # Load from environment variables
        if load_from_env:
            env_config = self._load_from_env()
            if env_config:
                config.update(env_config)
        
        # Validate configuration
        if validate:
            self._validate_config(config)
        
        self.logger.info(f"Configuration loaded from {config._source.value}")
        return config
    
    def _load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        self.logger.info(f"Loading config from file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            # Validate loaded data
            if not isinstance(config_data, dict):
                raise ValueError("Config file must contain a dictionary")
            
            # Security validation
            validated_config = self.security_manager.validate_input('config', config_data)
            
            return validated_config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {e}")
            raise
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Mapping of environment variables to config paths
        env_mappings = {
            'MODEL_NAME': 'model_name',
            'DEVICE': 'device',
            'CACHE_DIR': 'cache_dir',
            'ADAPTER_TYPE': 'adapter.type',
            'ADAPTER_RANK': 'adapter.rank',
            'ADAPTER_ALPHA': 'adapter.alpha',
            'RETRIEVAL_BACKEND': 'retrieval.backend',
            'RETRIEVAL_ENCODER': 'retrieval.encoder',
            'RETRIEVAL_INDEX_PATH': 'retrieval.index_path',
            'TRAINING_EPOCHS': 'training.num_epochs',
            'TRAINING_BATCH_SIZE': 'training.batch_size',
            'TRAINING_LR': 'training.learning_rate',
            'INFERENCE_MAX_LENGTH': 'inference.max_length',
            'INFERENCE_TEMPERATURE': 'inference.temperature',
            'LOG_LEVEL': 'logging.level',
            'LOG_FILE': 'logging.log_file',
            'SECURITY_ENABLE_VALIDATION': 'security.enable_input_validation',
            'SECURITY_RATE_LIMIT': 'security.max_requests_per_minute'
        }
        
        for env_var, config_path in env_mappings.items():
            full_env_var = self.env_prefix + env_var
            value = os.getenv(full_env_var)
            
            if value is not None:
                # Type conversion
                converted_value = self._convert_env_value(value)
                
                # Set in nested config structure
                self._set_nested_value(env_config, config_path, converted_value)
                
                self.logger.debug(f"Loaded from env: {full_env_var} -> {config_path} = {converted_value}")
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        parts = path.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _validate_config(self, config: Config):
        """Validate configuration values"""
        errors = []
        
        # Validate model name
        try:
            self.security_manager.validate_input('model_name', config.model_name)
        except ValueError as e:
            errors.append(f"Invalid model_name: {e}")
        
        # Validate adapter configuration
        if config.adapter.rank <= 0:
            errors.append("adapter.rank must be positive")
        
        if config.adapter.alpha <= 0:
            errors.append("adapter.alpha must be positive")
        
        if not config.adapter.target_modules:
            errors.append("adapter.target_modules cannot be empty")
        
        # Validate retrieval configuration
        if config.retrieval.chunk_size <= 0:
            errors.append("retrieval.chunk_size must be positive")
        
        if config.retrieval.max_retrieved_docs <= 0:
            errors.append("retrieval.max_retrieved_docs must be positive")
        
        # Validate training configuration
        if config.training.num_epochs <= 0:
            errors.append("training.num_epochs must be positive")
        
        if config.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        
        if config.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")
        
        # Validate inference configuration
        if config.inference.max_length <= 0:
            errors.append("inference.max_length must be positive")
        
        if not 0 < config.inference.temperature <= 2:
            errors.append("inference.temperature must be between 0 and 2")
        
        # Validate file paths
        if config.retrieval.index_path:
            try:
                self.security_manager.validate_input('file_path', config.retrieval.index_path)
            except ValueError as e:
                errors.append(f"Invalid retrieval.index_path: {e}")
        
        if config.logging.log_file:
            try:
                self.security_manager.validate_input('file_path', config.logging.log_file)
            except ValueError as e:
                errors.append(f"Invalid logging.log_file: {e}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def save_config(self, config: Config, file_path: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Path to save to
            format: File format ('yaml' or 'json')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        try:
            with open(file_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def create_template_config(self, file_path: Union[str, Path], format: str = 'yaml'):
        """
        Create a template configuration file with all options and comments.
        
        Args:
            file_path: Path to create template at
            format: File format ('yaml' or 'json')
        """
        template_config = Config()  # Default config with all options
        
        if format.lower() == 'yaml':
            # Add comments for YAML template
            config_dict = template_config.to_dict()
            self._add_yaml_comments(config_dict)
        
        self.save_config(template_config, file_path, format)
        self.logger.info(f"Template configuration created: {file_path}")
    
    def _add_yaml_comments(self, config_dict: Dict[str, Any]):
        """Add helpful comments to YAML config (placeholder for future enhancement)"""
        # This would add inline comments explaining each configuration option
        # For now, the structure itself serves as documentation
        pass


# Global config manager instance
_global_config_manager = None
_global_config = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return _global_config_manager


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Config:
    """
    Load configuration using global config manager.
    
    Args:
        config_path: Optional path to config file
        config_dict: Optional config dictionary
        **kwargs: Additional arguments for load_config
        
    Returns:
        Loaded configuration
    """
    global _global_config
    
    config_manager = get_config_manager()
    _global_config = config_manager.load_config(
        config_path=config_path,
        config_dict=config_dict,
        **kwargs
    )
    
    return _global_config


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Global configuration (loads default if not already loaded)
    """
    global _global_config
    
    if _global_config is None:
        _global_config = load_config()
    
    return _global_config


def update_config(updates: Union[Dict[str, Any], Config]):
    """
    Update global configuration.
    
    Args:
        updates: Updates to apply
    """
    global _global_config
    
    config = get_config()
    config.update(updates)


def save_config(file_path: Union[str, Path], format: str = 'yaml'):
    """
    Save global configuration to file.
    
    Args:
        file_path: Path to save to
        format: File format
    """
    config = get_config()
    config_manager = get_config_manager()
    config_manager.save_config(config, file_path, format)