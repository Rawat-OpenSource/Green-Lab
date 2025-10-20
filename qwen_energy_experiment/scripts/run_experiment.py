#!/usr/bin/env python3
"""
Main script for running the Qwen energy consumption experiment.
Provides command-line interface with configuration and resumability options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.experiment.controller import ExperimentController
from src.experiment.config import ExperimentConfig, ConfigLoader
from src.measurement.carbon_calculator import CarbonIntensityConfig


def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging for the experiment."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/experiment.log")
        ]
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen LLM energy consumption experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment with default configuration
  python scripts/run_experiment.py
  
  # Resume interrupted experiment
  python scripts/run_experiment.py --resume
  
  # Use custom carbon intensity
  python scripts/run_experiment.py --carbon-config configs/carbon_intensity/custom.json
  
  # Run with specific models only
  python scripts/run_experiment.py --models qwen2_5-7b qwen2_5-72b
  
  # Dry run to validate configuration
  python scripts/run_experiment.py --dry-run
        """
    )
    
    # Configuration files
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML"
    )
    
    parser.add_argument(
        "--models-config", 
        default="configs/models.yaml",
        help="Path to models configuration YAML"
    )
    
    parser.add_argument(
        "--carbon-config",
        help="Path to carbon intensity JSON config (uses default if not specified)"
    )
    
    # Execution options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Validate configuration without running experiment"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (subset of configured models)"
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (subset of configured tasks)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def validate_environment() -> bool:
    """
    Validate that the environment is ready for the experiment.
    
    Returns:
        True if environment is valid
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check GPU availability
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available - GPU required for experiment")
            return False
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 20:
            logger.warning("GPU memory < 20GB - may not be sufficient for larger models")
        
        # Check dependencies
        import vllm
        import pynvml
        import pyRAPL
        
        logger.info("All dependencies available")
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Qwen Energy Consumption Experiment")
    logger.info("=" * 50)
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Validate configuration files
        ConfigLoader.validate_config_files(args.experiment_config, args.models_config)
        
        # Load configurations
        logger.info("Loading configurations...")
        
        config = ExperimentConfig.from_files(args.experiment_config, args.models_config)
        models_config = ConfigLoader.load_models_config(args.models_config)
        
        # Load carbon intensity configuration
        carbon_config = None
        if args.carbon_config:
            carbon_config = CarbonIntensityConfig.from_config_file(args.carbon_config)
            logger.info(f"Using carbon intensity: {carbon_config.intensity_gco2e_per_kwh} gCO2e/kWh from {carbon_config.source}")
        else:
            logger.info("Using default carbon intensity (Netherlands average)")
        
        # Override models/tasks if specified
        if args.models:
            config.models = args.models
            logger.info(f"Using specified models: {args.models}")
        
        if args.tasks:
            config.tasks = args.tasks
            logger.info(f"Using specified tasks: {args.tasks}")
        
        # Validate configuration
        config.validate()
        
        # Print experiment summary
        logger.info("Experiment Configuration:")
        logger.info(f"  Models: {len(config.models)} ({', '.join(config.models)})")
        logger.info(f"  Tasks: {len(config.tasks)} ({', '.join(config.tasks)})")
        logger.info(f"  Repetitions: {config.repetitions}")
        logger.info(f"  Total runs: {config.get_total_runs()}")
        logger.info(f"  Estimated time: {config.get_estimated_time_hours():.1f} hours")
        
        # Dry run mode
        if args.dry_run:
            logger.info("Dry run completed successfully - configuration is valid")
            return
        
        # Create experiment controller
        logger.info("Initializing experiment controller...")
        controller = ExperimentController(config, models_config, carbon_config)
        
        # Run experiment
        logger.info("Starting experiment execution...")
        results = controller.run_experiment(resume=args.resume)
        
        # Print results summary
        logger.info("Experiment Results:")
        logger.info(f"  Session ID: {results['session_id']}")
        logger.info(f"  Runs completed: {results['runs_completed']}/{results['total_runs_planned']}")
        logger.info(f"  Completion rate: {results['completion_rate']:.1%}")
        logger.info(f"  Results saved to: {results['checkpoint_directory']}")
        
        if results['completion_rate'] == 1.0:
            logger.info("Experiment completed successfully!")
        else:
            logger.warning(f"Experiment partially completed ({results['completion_rate']:.1%})")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()