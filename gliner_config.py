#!/usr/bin/env python3
"""
GLiNER Configuration Management

Handles loading, validation, and management of JSON configuration files
for the GLiNER pipeline, replacing command-line arguments with structured
configuration files.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

class GLiNERConfig:
    """Configuration manager for GLiNER pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path is None:
            # Use default configuration
            self.config_path = self._get_default_config_path()
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Handle configuration inheritance
            if 'extends' in self.config:
                self._load_parent_config()
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            logger.info(f"Loaded configuration from: {self.config_path}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, "config", "gliner_config_default.json")
        
        if not os.path.exists(default_path):
            # Create config directory and default file if they don't exist
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            self._create_default_config(default_path)
        
        return default_path
    
    def _create_default_config(self, config_path: str):
        """Create default configuration file."""
        default_config = {
            "project": {
                "name": "Legal GLiNER Fine-tuning",
                "description": "Configuration for legal document NER training"
            },
            "data": {
                "train_folder": "input/original/annotated-json/train",
                "val_folder": "input/original/annotated-json/dev",
                "test_folder": "input/original/annotated-json/test",
                "max_tokens_per_chunk": 400,
                "chunk_overlap": 50,
                "train_split": 1.0
            },
            "model": {
                "name": "urchade/gliner_small-v2.1",
                "entity_types": ["Event", "Event_who", "Event_when", "Event_what"]
            },
            "training": {
                "num_epochs": 10,
                "batch_size": 8,
                "learning_rate": 5e-6,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "save_steps": 1000,
                "eval_steps": 1000,
                "logging_steps": 100
            },
            "evaluation": {
                "threshold": 0.5,
                "overlap_threshold": 0.5
            },
            "output": {
                "base_dir": "output/gliner_pipeline",
                "model_dir": "models",
                "predictions_dir": "predictions",
                "evaluation_dir": "evaluation"
            },
            "logging": {
                "level": "INFO",
                "save_logs": True
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    def _load_parent_config(self):
        """Load parent configuration for inheritance."""
        parent_path = self.config.get('extends')
        if not parent_path:
            return
        
        # Make path relative to current config directory
        if not os.path.isabs(parent_path):
            config_dir = os.path.dirname(self.config_path)
            parent_path = os.path.join(config_dir, parent_path)
        
        if not os.path.exists(parent_path):
            raise FileNotFoundError(f"Parent configuration file not found: {parent_path}")
        
        try:
            with open(parent_path, 'r', encoding='utf-8') as f:
                parent_config = json.load(f)
            
            # Merge parent config with current config (current overrides parent)
            merged_config = self._deep_merge(parent_config, self.config)
            self.config = merged_config
            
        except Exception as e:
            raise RuntimeError(f"Error loading parent configuration {parent_path}: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'GLINER_TRAIN_FOLDER': ('data', 'train_folder'),
            'GLINER_TEST_FOLDER': ('data', 'test_folder'),
            'GLINER_MODEL_NAME': ('model', 'name'),
            'GLINER_BATCH_SIZE': ('training', 'batch_size'),
            'GLINER_NUM_EPOCHS': ('training', 'num_epochs'),
            'GLINER_LEARNING_RATE': ('training', 'learning_rate'),
            'GLINER_THRESHOLD': ('evaluation', 'threshold'),
            'GLINER_OUTPUT_DIR': ('output', 'base_dir'),
            'GLINER_MAX_TOKENS': ('data', 'max_tokens_per_chunk'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if section not in self.config:
                    self.config[section] = {}
                
                # Type conversion
                if key in ['batch_size', 'num_epochs', 'max_tokens_per_chunk']:
                    self.config[section][key] = int(env_value)
                elif key in ['learning_rate', 'threshold', 'weight_decay', 'warmup_ratio']:
                    self.config[section][key] = float(env_value)
                elif key in ['save_logs']:
                    self.config[section][key] = env_value.lower() in ['true', '1', 'yes']
                else:
                    self.config[section][key] = env_value
                
                logger.info(f"Override from environment: {env_var} = {env_value}")
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = ['data', 'model', 'training', 'evaluation', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate required fields
        required_fields = {
            'data': ['train_folder', 'val_folder', 'test_folder'],
            'model': ['name', 'entity_types'],
            'training': ['num_epochs', 'batch_size', 'learning_rate'],
            'evaluation': ['threshold'],
            'output': ['base_dir']
        }
        
        for section, fields in required_fields.items():
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}")
        
        # Validate value ranges
        self._validate_ranges()
        
        # Validate paths
        self._validate_paths()
    
    def _validate_ranges(self):
        """Validate numerical parameter ranges."""
        validations = [
            ('training', 'batch_size', lambda x: 1 <= x <= 128, "Batch size must be between 1 and 128"),
            ('training', 'num_epochs', lambda x: 1 <= x <= 100, "Number of epochs must be between 1 and 100"),
            ('training', 'learning_rate', lambda x: 1e-8 <= x <= 1e-1, "Learning rate must be between 1e-8 and 1e-1"),
            ('evaluation', 'threshold', lambda x: 0.0 <= x <= 1.0, "Threshold must be between 0.0 and 1.0"),
            ('data', 'max_tokens_per_chunk', lambda x: 50 <= x <= 2000, "Max tokens per chunk must be between 50 and 2000"),
            ('data', 'chunk_overlap', lambda x: 0 <= x <= 200, "Chunk overlap must be between 0 and 200"),
        ]
        
        for section, field, validator, message in validations:
            if section in self.config and field in self.config[section]:
                value = self.config[section][field]
                if not validator(value):
                    raise ValueError(f"{message}. Got: {value}")
    
    def _validate_paths(self):
        """Validate that required paths exist."""
        train_folder = self.get('data.train_folder')
        val_folder = self.get('data.val_folder')
        test_folder = self.get('data.test_folder')
        
        if not os.path.exists(train_folder):
            logger.warning(f"Training folder not found: {train_folder}")
        
        if not os.path.exists(val_folder):
            logger.warning(f"Validation folder not found: {val_folder}")
        
        if not os.path.exists(test_folder):
            logger.warning(f"Test folder not found: {test_folder}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            'train_file': None,  # Set by pipeline
            'val_file': None,    # Set by pipeline
            'model_name': self.get('model.name'),
            'output_dir': None,  # Set by pipeline
            'learning_rate': self.get('training.learning_rate'),
            'batch_size': self.get('training.batch_size'),
            'num_epochs': self.get('training.num_epochs'),
            'weight_decay': self.get('training.weight_decay', 0.01),
            'warmup_ratio': self.get('training.warmup_ratio', 0.1),
            'save_steps': self.get('training.save_steps', 1000),
            'eval_steps': self.get('training.eval_steps', 1000),
            'logging_steps': self.get('training.logging_steps', 100)
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation-specific configuration."""
        return {
            'model_path': None,    # Set by pipeline
            'test_folder': self.get('data.test_folder'),
            'output_folder': None, # Set by pipeline
            'threshold': self.get('evaluation.threshold'),
            'chunk_overlap_tokens': self.get('data.chunk_overlap', 50)
        }
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return {
            'train_folder': self.get('data.train_folder'),
            'val_folder': self.get('data.val_folder'),
            'test_folder': self.get('data.test_folder'),
            'max_tokens_per_chunk': self.get('data.max_tokens_per_chunk'),
            'chunk_overlap': self.get('data.chunk_overlap'),
            'train_split': self.get('data.train_split', 1.0)
        }
    
    def save(self, output_path: str):
        """Save current configuration to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2, ensure_ascii=False)

def load_config(config_path: Optional[str] = None) -> GLiNERConfig:
    """
    Load GLiNER configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        GLiNERConfig instance
    """
    return GLiNERConfig(config_path)

def create_config_template(output_path: str):
    """Create a documented configuration template."""
    template = {
        "_description": "GLiNER Fine-tuning Configuration Template",
        "_usage": "Copy this file and modify values as needed. Remove comments (keys starting with _) before use.",
        
        "project": {
            "_description": "Project metadata",
            "name": "Legal GLiNER Fine-tuning",
            "description": "Configuration for legal document NER training"
        },
        
        "extends": {
            "_description": "Optional: inherit from another config file",
            "_example": "config/base_config.json",
            "_note": "Remove this section if not using inheritance"
        },
        
        "data": {
            "_description": "Data processing configuration",
            "train_folder": "input/original/annotated-json/train",
            "test_folder": "input/original/annotated-json/test",
            "max_tokens_per_chunk": {
                "_description": "Maximum tokens per text chunk (50-2000)",
                "_value": 400
            },
            "chunk_overlap": {
                "_description": "Overlap tokens between chunks (0-200)",
                "_value": 50
            },
            "train_split": {
                "_description": "Training/validation split ratio (0.0-1.0)",
                "_value": 0.9
            }
        },
        
        "model": {
            "_description": "Model configuration",
            "name": {
                "_description": "Pre-trained GLiNER model name",
                "_options": ["urchade/gliner_small-v2.1", "urchade/gliner_medium-v2.1", "urchade/gliner_large-v2.1"],
                "_value": "urchade/gliner_small-v2.1"
            },
            "entity_types": {
                "_description": "Entity types to extract",
                "_value": ["Event", "Event_who", "Event_when", "Event_what"]
            }
        },
        
        "training": {
            "_description": "Training hyperparameters",
            "num_epochs": {
                "_description": "Number of training epochs (1-100)",
                "_value": 10
            },
            "batch_size": {
                "_description": "Training batch size (1-128). Reduce if out of memory.",
                "_value": 8
            },
            "learning_rate": {
                "_description": "Learning rate (1e-8 to 1e-1)",
                "_value": 5e-6
            },
            "weight_decay": {
                "_description": "Weight decay for regularization",
                "_value": 0.01
            },
            "warmup_ratio": {
                "_description": "Warmup ratio for learning rate schedule",
                "_value": 0.1
            },
            "save_steps": {
                "_description": "Steps between model saves",
                "_value": 1000
            },
            "eval_steps": {
                "_description": "Steps between evaluations",
                "_value": 1000
            },
            "logging_steps": {
                "_description": "Steps between logging",
                "_value": 100
            }
        },
        
        "evaluation": {
            "_description": "Evaluation configuration",
            "threshold": {
                "_description": "Confidence threshold for predictions (0.0-1.0)",
                "_value": 0.5
            },
            "overlap_threshold": {
                "_description": "Overlap threshold for evaluation metrics",
                "_value": 0.5
            }
        },
        
        "output": {
            "_description": "Output directory configuration",
            "base_dir": "output/gliner_pipeline",
            "model_dir": "models",
            "predictions_dir": "predictions",
            "evaluation_dir": "evaluation"
        },
        
        "logging": {
            "_description": "Logging configuration",
            "level": {
                "_description": "Logging level",
                "_options": ["DEBUG", "INFO", "WARNING", "ERROR"],
                "_value": "INFO"
            },
            "save_logs": {
                "_description": "Whether to save logs to file",
                "_value": True
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully:")
    print(config)