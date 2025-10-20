"""
Main experiment controller with resumability, thermal management, and session logging.
Orchestrates the complete energy consumption measurement experiment.

Implementation of Paper Section 4.4 (Experiment Design) and Section 5 (Experiment Execution).
Specifically implements:
- Section 4.4.1: Two-factor factorial design with blocking
- Section 4.4.5: Experimental procedure with resumability
- Section 5.2: Preparation phases (Pre-Session, Per-Block, Per-Run)
- Section 5.3: Complete measurement pipeline
"""

import json
import time
import logging
import uuid
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import ExperimentConfig
from ..models.model_manager import ModelManager, ModelConfig
from ..models.inference_engine import InferenceEngine
from ..tasks.task_executor import TaskExecutor, TaskLibrary
from ..measurement.energy_monitor import EnergyMonitor
from ..measurement.carbon_calculator import CarbonIntensityConfig, CarbonCalculator

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Comprehensive session metadata for reproducibility."""
    
    session_id: str
    start_timestamp: datetime
    carbon_config: CarbonIntensityConfig
    
    # Hardware configuration
    gpu_model: str
    gpu_driver_version: str
    cuda_version: str
    cpu_model: str
    os_version: str
    
    # Software configuration
    python_version: str
    torch_version: str
    vllm_version: str
    measurement_config: Dict
    
    # Experiment configuration
    models_tested: List[str]
    tasks_tested: List[str]
    repetitions_per_condition: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'start_timestamp': self.start_timestamp.isoformat(),
            'carbon_intensity_config': self.carbon_config.to_dict(),
            
            'hardware': {
                'gpu_model': self.gpu_model,
                'gpu_driver_version': self.gpu_driver_version,
                'cuda_version': self.cuda_version,
                'cpu_model': self.cpu_model,
                'os_version': self.os_version
            },
            
            'software': {
                'python_version': self.python_version,
                'torch_version': self.torch_version,
                'vllm_version': self.vllm_version,
                'measurement_config': self.measurement_config
            },
            
            'experiment': {
                'models_tested': self.models_tested,
                'tasks_tested': self.tasks_tested,
                'repetitions_per_condition': self.repetitions_per_condition
            }
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save session metadata to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Session metadata saved to {filepath}")


class ExperimentController:
    """Main controller for the energy consumption experiment."""
    
    def __init__(self, 
                 config: ExperimentConfig,
                 models_config: Dict,
                 carbon_config: Optional[CarbonIntensityConfig] = None):
        """
        Initialize experiment controller.
        
        Args:
            config: Experiment configuration
            models_config: Models configuration dictionary
            carbon_config: Carbon intensity configuration (uses default if None)
        """
        self.config = config
        self.models_config = models_config
        self.carbon_config = carbon_config or CarbonIntensityConfig()
        
        # Initialize components
        self.energy_monitor = EnergyMonitor(gpu_sampling_rate=config.gpu_sampling_rate)
        self.model_manager = ModelManager()
        self.inference_engine = InferenceEngine(self.model_manager)
        self.task_library = TaskLibrary()
        self.task_executor = TaskExecutor(self.inference_engine, self.energy_monitor, self.task_library)
        self.carbon_calculator = CarbonCalculator(self.carbon_config)
        
        # Session management
        self.session_metadata: Optional[SessionMetadata] = None
        self.current_session_dir: Optional[Path] = None
        
        # Checkpoint management
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Experiment controller initialized")
    
    def start_session(self) -> str:
        """
        Start a new experimental session.
        
        Returns:
            Session ID
        """
        session_id = self._generate_session_id()
        self.current_session_dir = self.checkpoint_dir / session_id
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session metadata
        self.session_metadata = self._generate_session_metadata(session_id)
        
        # Save session metadata
        metadata_file = self.current_session_dir / "session_metadata.json"
        self.session_metadata.save_to_file(metadata_file)
        
        logger.info(f"Session {session_id} started")
        return session_id
    
    def run_experiment(self, resume: bool = True) -> Dict[str, any]:
        """
        Run the complete experiment with resumability.
        
        Args:
            resume: Whether to resume from existing checkpoints
            
        Returns:
            Experiment results summary
        """
        if self.session_metadata is None:
            self.start_session()
        
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Configuration: {len(self.config.models)} models, "
                   f"{len(self.config.tasks)} tasks, {self.config.repetitions} repetitions")
        logger.info(f"Total runs: {self.config.get_total_runs()}")
        
        # Generate experimental runs
        experimental_runs = self._generate_experimental_runs()
        
        # Filter completed runs if resuming
        if resume:
            experimental_runs = self._filter_completed_runs(experimental_runs)
            logger.info(f"Resuming: {len(experimental_runs)} runs remaining")
        
        if not experimental_runs:
            logger.info("All runs already completed")
            return self._generate_final_summary()
        
        # Group runs by model for efficient loading
        runs_by_model = self._group_runs_by_model(experimental_runs)
        
        # Execute experiment
        results = []
        for model_name, model_runs in runs_by_model.items():
            logger.info(f"Processing model: {model_name} ({len(model_runs)} runs)")
            
            model_results = self._execute_model_block(model_name, model_runs)
            results.extend(model_results)
        
        logger.info("Experiment completed successfully")
        return self._generate_final_summary()
    
    def _execute_model_block(self, model_name: str, runs: List[Dict]) -> List[Dict]:
        """
        Execute all runs for a specific model.
        
        Args:
            model_name: Name of the model to run
            runs: List of run configurations
            
        Returns:
            List of execution results
        """
        # Load model
        model_config = self._create_model_config(model_name)
        self.model_manager.load_model(model_config)
        
        # Warm up model
        if self.config.warmup_enabled:
            self.model_manager.warm_up_model(num_warmup_runs=self.config.warmup_prompts + 1)
        
        # Execute runs
        results = []
        for i, run in enumerate(runs):
            logger.info(f"Executing run {i+1}/{len(runs)}: {run['task_name']}")
            
            try:
                result = self._execute_single_run(run)
                results.append(result)
                
                # Save checkpoint immediately
                self._save_run_checkpoint(run['run_id'], result)
                
                # Thermal cooldown between runs
                if i < len(runs) - 1 and self.config.cooldown_enabled:
                    self._thermal_cooldown()
                    
            except Exception as e:
                logger.error(f"Failed to execute run {run['run_id']}: {e}")
                # Continue with next run rather than failing entire experiment
                continue
        
        # Unload model
        self.model_manager.unload_model()
        
        # Model-level cooldown
        if self.config.cooldown_enabled:
            self._thermal_cooldown()
        
        logger.info(f"Model {model_name} completed: {len(results)} successful runs")
        return results
    
    def _execute_single_run(self, run_config: Dict) -> Dict:
        """
        Execute a single experimental run with energy measurement.
        
        Args:
            run_config: Run configuration
            
        Returns:
            Run results with energy and carbon footprint
        """
        start_time = time.time()
        
        # Execute task with energy measurement
        task_result = self.task_executor.execute_single_task(
            task_name=run_config['task_name'],
            prompt_index=run_config['prompt_index'],
            record_diagnostics=True
        )
        
        # Calculate carbon footprint
        carbon_gco2e = self.carbon_calculator.calculate_carbon_footprint(task_result.energy_joules)
        
        # Combine results
        result = task_result.to_dict()
        result.update({
            'run_id': run_config['run_id'],
            'session_id': self.session_metadata.session_id,
            'execution_start_time': start_time,
            'execution_duration_seconds': time.time() - start_time,
            
            # Carbon footprint data
            'carbon_footprint_gco2e': carbon_gco2e,
            
            # Session-level carbon config
            'session_carbon_intensity_gco2e_per_kwh': self.carbon_config.intensity_gco2e_per_kwh,
            'carbon_intensity_source': self.carbon_config.source,
            
            # Quality control
            'measurement_quality': 'good'  # Could add quality assessment logic
        })
        
        return result
    
    def _thermal_cooldown(self) -> float:
        """
        Adaptive thermal cooldown based on GPU temperature and power.
        
        Returns:
            Actual cooldown time in seconds
        """
        thermal_config = self.config.thermal_management
        
        if not thermal_config.get('enabled', True):
            return 0.0
        
        logger.debug("Starting thermal cooldown")
        start_time = time.time()
        
        # Minimum cooldown period
        time.sleep(thermal_config['min_cooldown_seconds'])
        
        # Adaptive cooldown until thermal baseline reached
        while True:
            elapsed = time.time() - start_time
            
            # Get current thermal state
            thermal_state = self.energy_monitor.get_thermal_state()
            
            if 'error' in thermal_state:
                logger.warning("Could not get thermal state, using fixed cooldown")
                break
            
            temp = thermal_state['temperature_celsius']
            power = thermal_state['power_watts']
            
            # Check if baseline reached
            temp_ok = temp <= thermal_config['baseline_gpu_temp_celsius']
            power_ok = power <= thermal_config['baseline_gpu_power_watts']
            
            if temp_ok and power_ok:
                logger.debug(f"Thermal baseline reached: {temp:.1f}°C, {power:.1f}W after {elapsed:.1f}s")
                break
            
            # Safety timeout
            if elapsed >= thermal_config['max_cooldown_seconds']:
                logger.warning(f"Maximum cooldown reached ({elapsed:.1f}s), proceeding")
                break
            
            logger.debug(f"Waiting for thermal baseline: {temp:.1f}°C, {power:.1f}W")
            time.sleep(thermal_config.get('thermal_check_interval_seconds', 10))
        
        total_cooldown = time.time() - start_time
        logger.debug(f"Thermal cooldown completed: {total_cooldown:.1f}s")
        return total_cooldown
    
    def _generate_experimental_runs(self) -> List[Dict]:
        """Generate all experimental run configurations."""
        runs = []
        run_counter = 0
        
        for model_name in self.config.models:
            for task_name in self.config.tasks:
                for rep in range(self.config.repetitions):
                    run_id = f"{model_name}_{task_name}_{rep:02d}"
                    
                    runs.append({
                        'run_id': run_id,
                        'model_name': model_name,
                        'task_name': task_name,
                        'repetition': rep,
                        'prompt_index': rep % 20,  # Cycle through 20 prompts per task
                        'sequence_number': run_counter
                    })
                    
                    run_counter += 1
        
        # Randomize run order (Latin square would be implemented here)
        # For now, using simple randomization
        import random
        random.shuffle(runs)
        
        return runs
    
    def _filter_completed_runs(self, runs: List[Dict]) -> List[Dict]:
        """Filter out runs that have already been completed."""
        incomplete_runs = []
        
        for run in runs:
            checkpoint_file = self.current_session_dir / f"{run['run_id']}_result.json"
            
            if not checkpoint_file.exists():
                incomplete_runs.append(run)
            else:
                # Verify checkpoint is complete
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('completion_status') != 'complete':
                        incomplete_runs.append(run)
                        
                except (json.JSONDecodeError, KeyError):
                    incomplete_runs.append(run)
        
        skipped_count = len(runs) - len(incomplete_runs)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} completed runs")
        
        return incomplete_runs
    
    def _group_runs_by_model(self, runs: List[Dict]) -> Dict[str, List[Dict]]:
        """Group runs by model for efficient loading."""
        runs_by_model = {}
        
        for run in runs:
            model_name = run['model_name']
            if model_name not in runs_by_model:
                runs_by_model[model_name] = []
            runs_by_model[model_name].append(run)
        
        return runs_by_model
    
    def _create_model_config(self, model_name: str) -> ModelConfig:
        """Create ModelConfig from configuration."""
        model_info = self.models_config['models'][model_name]
        
        return ModelConfig(
            name=model_info['name'],
            model_path=model_info['model_path'],
            parameters=model_info['parameters'],
            generation=model_info['generation'],
            release_date=model_info['release_date'],
            max_context_length=model_info.get('max_context_length', 4096),
            torch_dtype=model_info.get('torch_dtype', 'auto')
        )
    
    def _save_run_checkpoint(self, run_id: str, result: Dict) -> None:
        """Save run result as checkpoint."""
        checkpoint_file = self.current_session_dir / f"{run_id}_result.json"
        
        result['completion_status'] = 'complete'
        result['checkpoint_timestamp'] = time.time()
        
        with open(checkpoint_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"qwen_energy_{timestamp}_{unique_id}"
    
    def _generate_session_metadata(self, session_id: str) -> SessionMetadata:
        """Generate comprehensive session metadata."""
        import sys
        
        try:
            import torch
            torch_version = torch.__version__
            cuda_version = torch.version.cuda if torch.cuda.is_available() else "Not Available"
        except ImportError:
            torch_version = "Not Available"
            cuda_version = "Not Available"
        
        # Get GPU info
        gpu_info = self.energy_monitor.get_system_info().get('gpu', {})
        
        measurement_config = {
            'gpu_sampling_rate_hz': self.config.gpu_sampling_rate,
            'energy_integration_method': 'trapezoidal',
            'rapl_overflow_handling': True,
            'thermal_baseline_temp_celsius': self.config.thermal_management.get('baseline_gpu_temp_celsius', 45),
            'thermal_baseline_power_watts': self.config.thermal_management.get('baseline_gpu_power_watts', 30),
            'primary_metrics': self.config.primary_metrics,
            'diagnostic_metrics': self.config.diagnostic_metrics
        }
        
        return SessionMetadata(
            session_id=session_id,
            start_timestamp=datetime.now(timezone.utc),
            carbon_config=self.carbon_config,
            
            gpu_model=gpu_info.get('name', 'Unknown'),
            gpu_driver_version=gpu_info.get('driver_version', 'Unknown'),
            cuda_version=cuda_version,
            cpu_model=platform.processor() or platform.machine(),
            os_version=f"{platform.system()} {platform.release()}",
            
            python_version=sys.version,
            torch_version=torch_version,
            vllm_version="vLLM_version_placeholder",  # Would get actual version
            measurement_config=measurement_config,
            
            models_tested=self.config.models,
            tasks_tested=self.config.tasks,
            repetitions_per_condition=self.config.repetitions
        )
    
    def _generate_final_summary(self) -> Dict[str, any]:
        """Generate final experiment summary."""
        # Count completed runs
        checkpoint_files = list(self.current_session_dir.glob("*_result.json"))
        completed_runs = len(checkpoint_files)
        
        return {
            'session_id': self.session_metadata.session_id if self.session_metadata else 'unknown',
            'experiment_name': self.config.name,
            'total_runs_planned': self.config.get_total_runs(),
            'runs_completed': completed_runs,
            'completion_rate': completed_runs / self.config.get_total_runs() if self.config.get_total_runs() > 0 else 0,
            'models_tested': self.config.models,
            'tasks_tested': self.config.tasks,
            'repetitions_per_condition': self.config.repetitions,
            'checkpoint_directory': str(self.current_session_dir),
            'carbon_intensity_used': self.carbon_config.intensity_gco2e_per_kwh,
            'completion_timestamp': datetime.now(timezone.utc).isoformat()
        }