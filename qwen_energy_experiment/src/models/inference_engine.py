"""
Inference engine for running LLM inference with performance tracking.
Integrates with vLLM for optimized inference and precise timing measurements.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from vllm import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Results from a single inference run."""
    
    # Input/output
    prompt: str
    response: str
    
    # Token counts
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Timing metrics
    latency_ms: float
    time_to_first_token_ms: Optional[float]
    generation_time_ms: float
    
    # Throughput metrics (both variants)
    throughput_output_tokens_per_sec: float
    throughput_total_tokens_per_sec: float
    
    # Quality metrics
    finish_reason: str
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            'prompt': self.prompt,
            'response': self.response,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'latency_ms': self.latency_ms,
            'time_to_first_token_ms': self.time_to_first_token_ms,
            'generation_time_ms': self.generation_time_ms,
            'throughput_output_tokens_per_sec': self.throughput_output_tokens_per_sec,
            'throughput_total_tokens_per_sec': self.throughput_total_tokens_per_sec,
            'finish_reason': self.finish_reason
        }


class InferenceEngine:
    """Handles LLM inference with precise performance measurements."""
    
    def __init__(self, model_manager):
        """
        Initialize inference engine.
        
        Args:
            model_manager: ModelManager instance with loaded model
        """
        self.model_manager = model_manager
        logger.info("Inference engine initialized")
    
    def run_inference(self, 
                     prompt: str,
                     max_tokens: int = 100,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     stop_sequences: Optional[List[str]] = None) -> InferenceResult:
        """
        Run inference on a single prompt with precise timing.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            
        Returns:
            InferenceResult with timing and token metrics
        """
        if self.model_manager.current_model is None:
            raise RuntimeError("No model loaded in ModelManager")
        
        # Count input tokens
        input_tokens = self.model_manager.count_tokens(prompt)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_sequences
        )
        
        # Run inference with precise timing
        start_time = time.perf_counter()
        
        outputs = self.model_manager.current_model.generate(
            [prompt], 
            sampling_params,
            use_tqdm=False  # Disable progress bar for clean logging
        )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract results
        output = outputs[0]
        response = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        
        # Calculate timing metrics
        # Note: vLLM doesn't expose time-to-first-token directly, so we estimate
        time_to_first_token_ms = None  # Would need custom vLLM callback
        generation_time_ms = latency_ms  # Approximation for total generation time
        
        # Calculate token metrics
        total_tokens = input_tokens + output_tokens
        
        # Calculate throughput metrics (both variants as specified in plan)
        latency_seconds = latency_ms / 1000.0
        throughput_output_tokens_per_sec = output_tokens / latency_seconds if latency_seconds > 0 else 0
        throughput_total_tokens_per_sec = total_tokens / latency_seconds if latency_seconds > 0 else 0
        
        result = InferenceResult(
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            generation_time_ms=generation_time_ms,
            throughput_output_tokens_per_sec=throughput_output_tokens_per_sec,
            throughput_total_tokens_per_sec=throughput_total_tokens_per_sec,
            finish_reason=finish_reason
        )
        
        logger.debug(f"Inference completed: {input_tokens} -> {output_tokens} tokens, "
                    f"{latency_ms:.1f}ms, {throughput_output_tokens_per_sec:.1f} tok/s")
        
        return result
    
    def run_batch_inference(self, 
                           prompts: List[str],
                           max_tokens: int = 100,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           stop_sequences: Optional[List[str]] = None) -> List[InferenceResult]:
        """
        Run inference on multiple prompts.
        
        Note: For energy measurement accuracy, we process prompts individually
        rather than batching to maintain per-query measurements.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum output tokens per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            
        Returns:
            List of InferenceResults
        """
        if not prompts:
            return []
        
        logger.info(f"Running batch inference on {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
            
            result = self.run_inference(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
            
            results.append(result)
        
        # Calculate batch statistics
        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        
        logger.info(f"Batch inference completed: {len(results)} queries, "
                   f"{total_latency:.1f}ms total, {total_output_tokens} output tokens")
        
        return results
    
    def calculate_efficiency_metrics(self, results: List[InferenceResult]) -> Dict[str, float]:
        """
        Calculate efficiency metrics from inference results.
        
        Args:
            results: List of inference results
            
        Returns:
            Dictionary with efficiency statistics
        """
        if not results:
            return {}
        
        # Aggregate metrics
        total_latency_ms = sum(r.latency_ms for r in results)
        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        
        # Calculate averages
        avg_latency_ms = total_latency_ms / len(results)
        avg_output_tokens = total_output_tokens / len(results)
        avg_total_tokens = total_tokens / len(results)
        
        # Calculate throughput metrics
        total_latency_seconds = total_latency_ms / 1000.0
        avg_throughput_output = total_output_tokens / total_latency_seconds if total_latency_seconds > 0 else 0
        avg_throughput_total = total_tokens / total_latency_seconds if total_latency_seconds > 0 else 0
        
        return {
            'num_queries': len(results),
            'total_latency_ms': total_latency_ms,
            'average_latency_ms': avg_latency_ms,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'average_output_tokens': avg_output_tokens,
            'average_total_tokens': avg_total_tokens,
            'average_throughput_output_tokens_per_sec': avg_throughput_output,
            'average_throughput_total_tokens_per_sec': avg_throughput_total
        }
    
    def validate_inference_quality(self, results: List[InferenceResult]) -> Dict[str, any]:
        """
        Validate inference quality and detect issues.
        
        Args:
            results: List of inference results
            
        Returns:
            Validation report
        """
        if not results:
            return {'error': 'No results to validate'}
        
        # Analyze finish reasons
        finish_reasons = [r.finish_reason for r in results]
        finish_reason_counts = {reason: finish_reasons.count(reason) for reason in set(finish_reasons)}
        
        # Check for common issues
        truncated_responses = sum(1 for r in results if r.finish_reason == 'length')
        empty_responses = sum(1 for r in results if not r.response.strip())
        very_short_responses = sum(1 for r in results if r.output_tokens < 3)
        
        # Quality metrics
        avg_response_length = sum(len(r.response) for r in results) / len(results)
        avg_output_tokens = sum(r.output_tokens for r in results) / len(results)
        
        quality_report = {
            'total_queries': len(results),
            'finish_reason_distribution': finish_reason_counts,
            'quality_issues': {
                'truncated_responses': truncated_responses,
                'empty_responses': empty_responses,
                'very_short_responses': very_short_responses
            },
            'average_response_length_chars': avg_response_length,
            'average_output_tokens': avg_output_tokens,
            'quality_score': 1.0 - (truncated_responses + empty_responses) / len(results)
        }
        
        # Log quality issues if found
        if truncated_responses > 0:
            logger.warning(f"Found {truncated_responses} truncated responses")
        if empty_responses > 0:
            logger.warning(f"Found {empty_responses} empty responses")
        
        return quality_report