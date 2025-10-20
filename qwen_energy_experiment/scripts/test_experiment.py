#!/usr/bin/env python3
"""
Quick test script for validating the Qwen energy experiment setup.
Uses minimal configuration for fast validation runs.

Implementation of Paper Section 4.4.7 (Time Budget and Feasibility) scaled-down testing.
Provides validation of Paper Section 5 (Experiment Execution) components.
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


def setup_test_logging():
    """Setup logging for test runs."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)


def run_minimal_test():
    """
    Run minimal test with smallest possible configuration:
    - 1 small model (qwen2_5-7b) (Paper Table 2: Selected LLM Subjects)
    - 1 task type (factual) (Paper Table 3: Selected Inference Subjects)  
    - 1 repetition (Paper Section 4.4.4: Repetitions reduced)
    Total: ~2-3 minutes
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Running MINIMAL test (1 model, 1 task, 1 rep)")
    logger.info("=" * 50)
    
    try:
        # Load base config and override for minimal test
        config = ExperimentConfig.from_files(
            "configs/experiment_test.yaml", 
            "configs/models.yaml"
        )
        models_config = ConfigLoader.load_models_config("configs/models.yaml")
        
        # Override for absolute minimal test
        config.models = ["qwen2_5-7b"]  # Smallest viable model
        config.tasks = ["factual"]      # basic task
        config.repetitions = 1          # Single run only
        
        logger.info(f"Test configuration: {config.models[0]} × {config.tasks[0]} × {config.repetitions} rep")
        logger.info(f"Estimated time: ~3 minutes")
        
        # Create controller and run
        controller = ExperimentController(config, models_config, None)
        results = controller.run_experiment(resume=False)
        
        logger.info("MINIMAL TEST COMPLETED SUCCESSFULLY!")
        logger.info(f"Results: {results['runs_completed']}/{results['total_runs_planned']} runs")
        
        return True
        
    except Exception as e:
        logger.error(f"Minimal test failed: {e}")
        return False


def run_quick_test():
    """
    Run quick test with test configuration:
    - 2 models (Paper Table 2: Subset)
    - 2 tasks (Paper Table 3: Subset) 
    - 2 repetitions (Paper Section 4.4.4: Reduced from 20)
    Total: ~15-20 minutes
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Running QUICK test (2 models, 2 tasks, 2 reps)")
    logger.info("=" * 50)
    
    try:
        # Use test configuration
        config = ExperimentConfig.from_files(
            "configs/experiment_test.yaml",
            "configs/models.yaml"
        )
        models_config = ConfigLoader.load_models_config("configs/models.yaml")
        
        # Override for quick test
        config.models = ["qwen2_5-7b", "qwen2_5-14b"]
        config.tasks = ["factual", "summarization"] 
        config.repetitions = 2
        
        logger.info(f"Test configuration: {len(config.models)} models × {len(config.tasks)} tasks × {config.repetitions} reps")
        logger.info(f"Total runs: {config.get_total_runs()}")
        logger.info(f"Estimated time: ~20 minutes")
        
        # Create controller and run
        controller = ExperimentController(config, models_config, None)
        results = controller.run_experiment(resume=False)
        
        logger.info("QUICK TEST COMPLETED SUCCESSFULLY!")
        logger.info(f"Results: {results['runs_completed']}/{results['total_runs_planned']} runs")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test Qwen energy experiment")
    parser.add_argument(
        "--test-type",
        choices=["minimal", "quick", "dry-run"],
        default="minimal",
        help="Type of test to run"
    )
    
    args = parser.parse_args()
    setup_test_logging()
    
    logger = logging.getLogger(__name__)
    
    if args.test_type == "dry-run":
        logger.info("Running dry-run validation (Paper Section 4.4.2: Platform validation)...")
        # Import and run main script's validation
        from scripts.run_experiment import validate_environment
        from src.experiment.config import ConfigLoader
        
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
            
        ConfigLoader.validate_config_files("configs/experiment_test.yaml", "configs/models.yaml")
        logger.info("Dry-run validation successful!")
        return
    
    elif args.test_type == "minimal":
        success = run_minimal_test()
    elif args.test_type == "quick":
        success = run_quick_test()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()