"""
Utilities for loading and managing prompt configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptConfig:
    """Class to handle prompt configurations"""
    
    def __init__(self, config_name: str, prompts_dir: str = "input/prompts"):
        self.config_name = config_name
        self.prompts_dir = Path(prompts_dir)
        self.config_path = self.prompts_dir / f"{config_name}.json"
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load prompt configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['event_definitions', 'instruction']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in prompt configuration")
            
            logger.info(f"Successfully loaded prompt configuration: {self.config_name}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompt configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompt configuration: {e}")
    
    @property
    def event_definitions(self) -> str:
        """Get event definitions from the configuration"""
        return self.config_data['event_definitions']
    
    @property
    def instruction(self) -> str:
        """Get instruction from the configuration"""
        return self.config_data['instruction']
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary"""
        return self.config_data.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the prompt configuration"""
        return {
            'config_name': self.config_name,
            'config_path': str(self.config_path),
            'event_definitions': self.event_definitions,
            'instruction': self.instruction
        }

def list_available_prompts(prompts_dir: str = "input/prompts") -> List[str]:
    """List all available prompt configuration files"""
    prompts_path = Path(prompts_dir)
    
    if not prompts_path.exists():
        logger.warning(f"Prompts directory not found: {prompts_path}")
        return []
    
    json_files = list(prompts_path.glob("*.json"))
    prompt_names = [f.stem for f in json_files]
    
    logger.info(f"Found {len(prompt_names)} prompt configurations: {prompt_names}")
    return prompt_names

def load_prompt_config(config_name: str, prompts_dir: str = "input/prompts") -> PromptConfig:
    """Load a specific prompt configuration by name"""
    return PromptConfig(config_name, prompts_dir)

def validate_prompt_config(config_path: str) -> bool:
    """Validate a prompt configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_fields = ['event_definitions', 'instruction']
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field '{field}' in {config_path}")
                return False
        
        # Check if values are non-empty strings
        for field in required_fields:
            if not isinstance(config[field], str) or not config[field].strip():
                logger.error(f"Field '{field}' must be a non-empty string in {config_path}")
                return False
        
        logger.info(f"Prompt configuration is valid: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating prompt configuration {config_path}: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # List available prompts
    available_prompts = list_available_prompts()
    print(f"Available prompt configurations: {available_prompts}")
    
    # Load a specific prompt
    if available_prompts:
        config = load_prompt_config(available_prompts[0])
        print(f"Loaded configuration: {config.config_name}")
        print(f"Event definitions: {config.event_definitions[:100]}...")
        print(f"Instruction: {config.instruction[:100]}...")
