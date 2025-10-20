"""
Configuration management for the energy consumption experiment.
Loads and validates configuration from YAML files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the energy consumption experiment."""
    
    # Experiment metadata
    name: str
    version: str
    description: str
    
    # Experimental design
    models: List[str]
    tasks: List[str]
    repetitions: int
    randomization_method: str
    
    # Measurement settings
    gpu_sampling_rate: int
    thermal_management: Dict[str, Any]
    
    # Execution settings
    checkpoint_dir: Path
    warmup_enabled: bool
    warmup_prompts: int
    cooldown_enabled: bool
    
    # Analysis settings
    primary_metrics: List[str]
    diagnostic_metrics: List[str]
    
    @classmethod
    def from_files(cls, 
                   experiment_config_path: str,
                   models_config_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML files.
        
        Args:
            experiment_config_path: Path to experiment configuration YAML
            models_config_path: Path to models configuration YAML
            
        Returns:
            ExperimentConfig instance
        """
        logger.info("Loading experiment configuration")
        
        # Load experiment config
        with open(experiment_config_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        # Load models config
        with open(models_config_path, 'r') as f:
            models_config = yaml.safe_load(f)
        
        # Extract model names
        model_names = list(models_config['models'].keys())
        
        # Create configuration object
        config = cls(
            name=exp_config['experiment']['name'],
            version=exp_config['experiment']['version'],
            description=exp_config['experiment']['description'],
            
            models=model_names,
            tasks=['factual_simple', 'summarization_short', 'reasoning_arithmetic', 'code_simple'],
            repetitions=exp_config['design']['repetitions'],
            randomization_method=exp_config['design']['randomization']['method'],
            
            gpu_sampling_rate=exp_config['measurement']['energy']['gpu_sampling_rate_hz'],
            thermal_management=exp_config['thermal_management'],
            
            checkpoint_dir=Path(exp_config['data']['checkpointing']['checkpoint_dir']),
            warmup_enabled=exp_config['model_execution']['warmup']['enabled'],
            warmup_prompts=exp_config['model_execution']['warmup']['additional_warmup_prompts'],
            cooldown_enabled=exp_config['thermal_management']['enabled'],
            
            primary_metrics=exp_config['measurement']['primary_metrics'],
            diagnostic_metrics=exp_config['measurement']['diagnostic_metrics']
        )
        
        logger.info(f"Configuration loaded: {len(config.models)} models, "
                   f"{len(config.tasks)} tasks, {config.repetitions} repetitions")
        
        return config
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating experiment configuration")
        
        # Check required fields
        if not self.name or not self.version:
            raise ValueError("Experiment name and version are required")
        
        if not self.models:
            raise ValueError("At least one model must be specified")
        
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        
        if self.repetitions <= 0:
            raise ValueError("Repetitions must be positive")
        
        # Check measurement settings
        if self.gpu_sampling_rate <= 0:
            raise ValueError("GPU sampling rate must be positive")
        
        # Check thermal management
        if self.cooldown_enabled:
            thermal = self.thermal_management
            if thermal['min_cooldown_seconds'] <= 0:
                raise ValueError("Minimum cooldown must be positive")
            if thermal['max_cooldown_seconds'] <= thermal['min_cooldown_seconds']:
                raise ValueError("Maximum cooldown must be greater than minimum")
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Configuration validation passed")
        return True
    
    def get_total_runs(self) -> int:
        """Calculate total number of experimental runs."""
        return len(self.models) * len(self.tasks) * self.repetitions
    
    def get_estimated_time_hours(self) -> float:
        """Estimate total experiment time in hours."""
        # Rough estimation based on configuration
        runs_per_hour = 20  # Conservative estimate
        total_runs = self.get_total_runs()
        
        # Add overhead for model loading, warmup, cooldown
        model_loading_hours = len(self.models) * 0.25  # 15 min per model
        warmup_hours = len(self.models) * 0.1 if self.warmup_enabled else 0
        cooldown_hours = total_runs * (3/60) if self.cooldown_enabled else 0  # 3 min per run
        
        base_time = total_runs / runs_per_hour
        total_time = base_time + model_loading_hours + warmup_hours + cooldown_hours
        
        return total_time
    
    def to_dict(self) -> Dict[str, Any]:
        #Convert configuration to dictionary for serialization.
        return {
            'experiment': {
                'name': self.name,
                'version': self.version,
                'description': self.description
            },
            'design': {
                'models': self.models,
                'tasks': self.tasks,
                'repetitions': self.repetitions,
                'total_runs': self.get_total_runs(),
                'randomization_method': self.randomization_method
            },
            'measurement': {
                'gpu_sampling_rate': self.gpu_sampling_rate,
                'primary_metrics': self.primary_metrics,
                'diagnostic_metrics': self.diagnostic_metrics
            },
            'execution': {
                'checkpoint_dir': str(self.checkpoint_dir),
                'warmup_enabled': self.warmup_enabled,
                'warmup_prompts': self.warmup_prompts,
                'cooldown_enabled': self.cooldown_enabled
            },
            'estimates': {
                'total_runs': self.get_total_runs(),
                'estimated_time_hours': self.get_estimated_time_hours()
            }
        }


class ConfigLoader:
    #Utility class for loading various configuration files.
    
    @staticmethod
    def load_models_config(config_path: str) -> Dict[str, Any]:
        #Load models configuration from YAML file.
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_experiment_config(config_path: str) -> Dict[str, Any]:
        #Load experiment configuration from YAML file.
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def validate_config_files(experiment_path: str, models_path: str) -> bool:
     
        # Validate that configuration files exist and are readable.

        exp_path = Path(experiment_path)
        mod_path = Path(models_path)
        
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
        
        if not mod_path.exists():
            raise FileNotFoundError(f"Models config not found: {models_path}")
        
        # Try to load and parse
        try:
            with open(exp_path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid experiment config YAML: {e}")
        
        try:
            with open(mod_path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid models config YAML: {e}")
        
        logger.info("Configuration files validated successfully")
        return True