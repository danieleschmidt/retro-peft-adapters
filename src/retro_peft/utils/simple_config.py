"""
Simple configuration management without heavy dependencies.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary"""
    return {
        "adapters": {
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "adalora": {
                "initial_r": 64,
                "target_r": 8,
                "beta1": 0.85,
                "beta2": 0.85
            },
            "ia3": {
                "target_modules": ["k_proj", "v_proj", "down_proj"]
            }
        },
        "retrieval": {
            "embedding_dim": 768,
            "chunk_size": 512,
            "overlap": 50,
            "max_docs": 5,
            "backend": "mock"
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "output_dir": "./checkpoints"
        },
        "inference": {
            "max_length": 200,
            "temperature": 0.7,
            "do_sample": True
        },
        "logging": {
            "level": "INFO",
            "format": "simple"
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure"""
    required_sections = ["adapters", "retrieval", "training", "inference", "logging"]
    
    for section in required_sections:
        if section not in config:
            print(f"Warning: Missing config section: {section}")
            return False
    
    # Basic validation
    if config["adapters"]["lora"]["r"] <= 0:
        return False
    
    if config["retrieval"]["chunk_size"] <= 0:
        return False
    
    return True


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or defaults"""
    config = get_default_config()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge user config with defaults
            for section, values in user_config.items():
                if section in config:
                    if isinstance(values, dict) and isinstance(config[section], dict):
                        config[section].update(values)
                    else:
                        config[section] = values
                else:
                    config[section] = values
        
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        raise