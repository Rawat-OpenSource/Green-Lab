"""
Task executor for running inference tasks with energy measurement.
Coordinates between task definitions, inference engine, and energy monitoring.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .task_definitions import TaskDefinition, TaskLibrary
from ..models.inference_engine import InferenceEngine, InferenceResult
from ..measurement.energy_monitor import EnergyMonitor

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """Results from executing a task with energy measurement."""
    
    # Task information
    task_name: str
    task_category: str
    task_complexity: str
    prompt: str
    
    # Inference results
    inference_result: InferenceResult
    
    # Energy measurements
    energy_joules: float
    gpu_energy_joules: float
    cpu_energy_joules: float
    dram_energy_joules: float
    
    # Execution metadata
    execution_timestamp: float
    model_name: str
    
    # Diagnostic metrics (quality control only)
    gpu_temperature_start: Optional[float] = None
    gpu_temperature_end: Optional[float] = None
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        result = {
            # Task info
            'task_name': self.task_name,
            'task_category': self.task_category,
            'task_complexity': self.task_complexity,
            'prompt': self.prompt,
            
            # Primary metrics (KPIs)
            'response': self.inference_result.response,
            'input_tokens': self.inference_result.input_tokens,
            'output_tokens': self.inference_result.output_tokens,
            'total_tokens': self.inference_result.total_tokens,
            'latency_ms': self.inference_result.latency_ms,
            'throughput_output_tokens_per_sec': self.inference_result.throughput_output_tokens_per_sec,
            'throughput_total_tokens_per_sec': self.inference_result.throughput_total_tokens_per_sec,
            
            # Energy metrics (primary)
            'total_energy_joules': self.energy_joules,
            'gpu_energy_joules': self.gpu_energy_joules,
            'cpu_energy_joules': self.cpu_energy_joules,
            'dram_energy_joules': self.dram_energy_joules,
            
            # Derived efficiency metrics
            'energy_per_output_token': self.energy_joules / self.inference_result.output_tokens if self.inference_result.output_tokens > 0 else float('inf'),
            'energy_per_total_token': self.energy_joules / self.inference_result.total_tokens if self.inference_result.total_tokens > 0 else float('inf'),
            'output_tokens_per_joule': self.inference_result.output_tokens / self.energy_joules if self.energy_joules > 0 else 0,
            'total_tokens_per_joule': self.inference_result.total_tokens / self.energy_joules if self.energy_joules > 0 else 0,
            
            # Metadata
            'execution_timestamp': self.execution_timestamp,
            'model_name': self.model_name,
            'finish_reason': self.inference_result.finish_reason,
            
            # Diagnostic metrics (not primary KPIs)
            'gpu_temperature_start_celsius': self.gpu_temperature_start,
            'gpu_temperature_end_celsius': self.gpu_temperature_end,
        }
        
        return result


class TaskExecutor:
    """Executes tasks with comprehensive measurement and monitoring."""
    
    def __init__(self, 
                 inference_engine: InferenceEngine,
                 energy_monitor: EnergyMonitor,
                 task_library: Optional[TaskLibrary] = None):
        """
        Initialize task executor.
        
        Args:
            inference_engine: Engine for running inference
            energy_monitor: Monitor for energy measurements
            task_library: Library of task definitions (creates default if None)
        """
        self.inference_engine = inference_engine
        self.energy_monitor = energy_monitor
        self.task_library = task_library or TaskLibrary()
        
        logger.info("Task executor initialized")
    
    def execute_single_task(self, 
                           task_name: str,
                           prompt_index: int = 0,
                           record_diagnostics: bool = True) -> TaskExecutionResult:
        """
        Execute a single task with energy measurement.
        
        Args:
            task_name: Name of the task to execute
            prompt_index: Index of prompt to use from task definition
            record_diagnostics: Whether to record diagnostic metrics
            
        Returns:
            TaskExecutionResult with comprehensive measurements
        """
        # Get task definition
        task_def = self.task_library.get_task(task_name)
        
        if prompt_index >= len(task_def.prompts):
            raise ValueError(f"Prompt index {prompt_index} out of range for task {task_name}")
        
        prompt = task_def.prompts[prompt_index]
        model_name = self.inference_engine.model_manager.current_config.name
        
        logger.debug(f"Executing task {task_name}, prompt {prompt_index}, model {model_name}")
        
        # Record diagnostic data if requested
        gpu_temp_start = None
        gpu_temp_end = None
        
        if record_diagnostics:
            thermal_state = self.energy_monitor.get_thermal_state()
            gpu_temp_start = thermal_state.get('temperature_celsius')
        
        # Start energy measurement
        self.energy_monitor.start_measurement()
        
        try:
            # Run inference
            inference_result = self.inference_engine.run_inference(
                prompt=prompt,
                max_tokens=task_def.max_tokens,
                temperature=task_def.temperature,
                top_p=task_def.top_p,
                stop_sequences=task_def.stop_sequences
            )
            
            # Stop energy measurement
            energy_data = self.energy_monitor.stop_measurement()
            
            # Record end diagnostics
            if record_diagnostics:
                thermal_state = self.energy_monitor.get_thermal_state()
                gpu_temp_end = thermal_state.get('temperature_celsius')
            
            # Create result
            result = TaskExecutionResult(
                task_name=task_name,
                task_category=task_def.category.value,
                task_complexity=task_def.complexity,
                prompt=prompt,
                inference_result=inference_result,
                energy_joules=energy_data['total_joules'],
                gpu_energy_joules=energy_data['gpu_joules'],
                cpu_energy_joules=energy_data['cpu_joules'],
                dram_energy_joules=energy_data['dram_joules'],
                execution_timestamp=time.time(),
                model_name=model_name,
                gpu_temperature_start=gpu_temp_start,
                gpu_temperature_end=gpu_temp_end
            )
            
            logger.debug(f"Task {task_name} completed: {inference_result.output_tokens} tokens, "
                        f"{energy_data['total_joules']:.3f} J")
            
            return result
            
        except Exception as e:
            # Make sure to stop energy monitoring even if inference fails
            try:
                self.energy_monitor.stop_measurement()
            except:
                pass
            
            logger.error(f"Failed to execute task {task_name}: {e}")
            raise
    
    def execute_task_batch(self, 
                          task_name: str,
                          num_repetitions: int = 20,
                          record_diagnostics: bool = True) -> List[TaskExecutionResult]:
        """
        Execute multiple repetitions of a task.
        
        Args:
            task_name: Name of the task to execute
            num_repetitions: Number of repetitions to run
            record_diagnostics: Whether to record diagnostic metrics
            
        Returns:
            List of TaskExecutionResults
        """
        task_def = self.task_library.get_task(task_name)
        
        if num_repetitions > len(task_def.prompts):
            logger.warning(f"Requested {num_repetitions} repetitions but task {task_name} "
                          f"only has {len(task_def.prompts)} prompts. Using all available.")
            num_repetitions = len(task_def.prompts)
        
        logger.info(f"Executing {num_repetitions} repetitions of task {task_name}")
        
        results = []
        for i in range(num_repetitions):
            logger.debug(f"Task {task_name} repetition {i+1}/{num_repetitions}")
            
            result = self.execute_single_task(
                task_name=task_name,
                prompt_index=i,
                record_diagnostics=record_diagnostics
            )
            
            results.append(result)
        
        logger.info(f"Completed {len(results)} repetitions of task {task_name}")
        return results
    
    def execute_all_tasks(self, 
                         num_repetitions: int = 20,
                         record_diagnostics: bool = True) -> Dict[str, List[TaskExecutionResult]]:
        """
        Execute all tasks in the task library.
        
        Args:
            num_repetitions: Number of repetitions per task
            record_diagnostics: Whether to record diagnostic metrics
            
        Returns:
            Dictionary mapping task names to their results
        """
        all_results = {}
        all_tasks = self.task_library.get_all_tasks()
        
        logger.info(f"Executing all {len(all_tasks)} tasks with {num_repetitions} repetitions each")
        
        for task_def in all_tasks:
            task_name = task_def.name
            logger.info(f"Starting task {task_name}")
            
            results = self.execute_task_batch(
                task_name=task_name,
                num_repetitions=num_repetitions,
                record_diagnostics=record_diagnostics
            )
            
            all_results[task_name] = results
            
            # Log summary for this task
            total_energy = sum(r.energy_joules for r in results)
            avg_latency = sum(r.inference_result.latency_ms for r in results) / len(results)
            avg_tokens = sum(r.inference_result.output_tokens for r in results) / len(results)
            
            logger.info(f"Task {task_name} completed: {len(results)} runs, "
                       f"{total_energy:.2f}J total, {avg_latency:.1f}ms avg latency, "
                       f"{avg_tokens:.1f} avg output tokens")
        
        # Log overall summary
        total_runs = sum(len(results) for results in all_results.values())
        total_energy = sum(r.energy_joules for results in all_results.values() for r in results)
        
        logger.info(f"All tasks completed: {total_runs} total runs, {total_energy:.2f}J total energy")
        
        return all_results
    
    def validate_task_quality(self, results: List[TaskExecutionResult]) -> Dict[str, any]:
        """
        Validate quality of task execution results.
        
        Args:
            results: List of task execution results
            
        Returns:
            Quality validation report
        """
        if not results:
            return {'error': 'No results to validate'}
        
        # Extract inference results for analysis
        inference_results = [r.inference_result for r in results]
        
        # Use inference engine's validation
        quality_report = self.inference_engine.validate_inference_quality(inference_results)
        
        # Add task-specific metrics
        energy_values = [r.energy_joules for r in results]
        latency_values = [r.inference_result.latency_ms for r in results]
        
        quality_report.update({
            'energy_statistics': {
                'mean_joules': sum(energy_values) / len(energy_values),
                'min_joules': min(energy_values),
                'max_joules': max(energy_values),
                'std_dev': self._calculate_std_dev(energy_values)
            },
            'latency_statistics': {
                'mean_ms': sum(latency_values) / len(latency_values),
                'min_ms': min(latency_values),
                'max_ms': max(latency_values),
                'std_dev': self._calculate_std_dev(latency_values)
            }
        })
        
        return quality_report
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_execution_summary(self, results: List[TaskExecutionResult]) -> Dict[str, any]:
        """
        Generate execution summary statistics.
        
        Args:
            results: List of execution results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {'error': 'No results provided'}
        
        # Primary metrics
        total_energy = sum(r.energy_joules for r in results)
        total_latency = sum(r.inference_result.latency_ms for r in results)
        total_output_tokens = sum(r.inference_result.output_tokens for r in results)
        total_input_tokens = sum(r.inference_result.input_tokens for r in results)
        
        # Efficiency metrics
        avg_energy_per_token = total_energy / total_output_tokens if total_output_tokens > 0 else 0
        avg_tokens_per_joule = total_output_tokens / total_energy if total_energy > 0 else 0
        avg_throughput = total_output_tokens / (total_latency / 1000) if total_latency > 0 else 0
        
        return {
            'num_executions': len(results),
            'total_energy_joules': total_energy,
            'total_latency_ms': total_latency,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'average_energy_per_output_token': avg_energy_per_token,
            'average_output_tokens_per_joule': avg_tokens_per_joule,
            'average_throughput_tokens_per_sec': avg_throughput,
            'model_name': results[0].model_name,
            'tasks_executed': list(set(r.task_name for r in results))
        }