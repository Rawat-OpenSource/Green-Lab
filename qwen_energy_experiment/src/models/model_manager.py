"""
Model management system for loading, caching, and managing Qwen models.
Handles memory management, model warm-up, and VRAM optimization.
"""

import logging
import time
import gc
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

import torch
import psutil
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a Qwen model."""
    
    name: str
    model_path: str
    parameters: int
    generation: str
    release_date: str
    
    # Model settings
    max_context_length: int = 4096
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.parameters <= 0:
            raise ValueError("Model parameters must be positive")
        if not self.model_path:
            raise ValueError("Model path cannot be empty")


class ModelManager:
    """Manages model loading, unloading, and memory optimization."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for caching models (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.current_model = None
        self.current_tokenizer = None
        self.current_config: Optional[ModelConfig] = None
        self.model_cache: Dict[str, any] = {}
        
        # Memory tracking
        self._initial_memory = self._get_memory_usage()
        
        logger.info(f"Model manager initialized. Cache dir: {self.cache_dir}")
    
    def load_model(self, model_config: ModelConfig, use_cache: bool = True) -> None:
        """
        Load a model and tokenizer.
        
        Args:
            model_config: Model configuration
            use_cache: Whether to use cached models
        """
        if self.current_model is not None and self.current_config.name == model_config.name:
            logger.info(f"Model {model_config.name} already loaded")
            return
        
        # Unload current model first
        if self.current_model is not None:
            self.unload_model()
        
        logger.info(f"Loading model: {model_config.name} ({model_config.parameters/1e9:.1f}B params)")
        start_time = time.time()
        
        try:
            # Check if model is cached
            if use_cache and model_config.name in self.model_cache:
                logger.info(f"Loading {model_config.name} from cache")
                self.current_model = self.model_cache[model_config.name]['model']
                self.current_tokenizer = self.model_cache[model_config.name]['tokenizer']
            else:
                # Load model and tokenizer fresh
                self._load_model_fresh(model_config)
                
                # Cache if requested and memory allows
                if use_cache and self._can_cache_model():
                    self.model_cache[model_config.name] = {
                        'model': self.current_model,
                        'tokenizer': self.current_tokenizer
                    }
                    logger.info(f"Model {model_config.name} cached in memory")
            
            self.current_config = model_config
            load_time = time.time() - start_time
            
            # Log memory usage
            memory_info = self._get_memory_usage()
            logger.info(f"Model {model_config.name} loaded in {load_time:.2f}s. "
                       f"Memory: {memory_info['used_gb']:.1f}GB GPU, "
                       f"{memory_info['used_ram_gb']:.1f}GB RAM")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_config.name}: {e}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_config = None
            raise
    
    def _load_model_fresh(self, model_config: ModelConfig) -> None:
        """Load model and tokenizer from scratch."""
        # Import here to avoid circular dependencies
        from vllm import LLM
        
        # Configure model loading
        model_kwargs = {
            'model': model_config.model_path,
            'trust_remote_code': model_config.trust_remote_code,
            'max_model_len': model_config.max_context_length,
            'gpu_memory_utilization': 0.9,  # Use most of GPU memory
            'swap_space': 4,  # 4GB swap space for larger models
        }
        
        # Set dtype if specified
        if model_config.torch_dtype != "auto":
            model_kwargs['dtype'] = model_config.torch_dtype
        
        # Load model with vLLM for optimized inference
        self.current_model = LLM(**model_kwargs)
        
        # Load tokenizer separately for token counting
        self.current_tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code
        )
        
        logger.debug(f"Model and tokenizer loaded for {model_config.name}")
    
    def unload_model(self) -> None:
        """Unload current model and free memory."""
        if self.current_model is None:
            logger.debug("No model to unload")
            return
        
        model_name = self.current_config.name if self.current_config else "unknown"
        logger.info(f"Unloading model: {model_name}")
        
        # Clear model references
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory after cleanup
        memory_info = self._get_memory_usage()
        logger.info(f"Model unloaded. Memory: {memory_info['used_gb']:.1f}GB GPU, "
                   f"{memory_info['used_ram_gb']:.1f}GB RAM")
    
    def warm_up_model(self, warmup_prompt: str = "Hello, how are you today?", 
                      num_warmup_runs: int = 3) -> None:
        """
        Warm up the model with dummy inferences (results discarded).
        
        Args:
            warmup_prompt: Prompt to use for warm-up
            num_warmup_runs: Number of warm-up inferences
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded for warm-up")
        
        logger.info(f"Warming up model {self.current_config.name} with {num_warmup_runs} runs")
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=20  # Short warm-up responses
        )
        
        start_time = time.time()
        
        for i in range(num_warmup_runs):
            logger.debug(f"Warm-up run {i+1}/{num_warmup_runs}")
            
            # Generate response (discarded)
            _ = self.current_model.generate([warmup_prompt], sampling_params)
            
            # Small delay between warm-ups
            if i < num_warmup_runs - 1:
                time.sleep(1)
        
        warmup_time = time.time() - start_time
        logger.info(f"Model warm-up completed in {warmup_time:.2f}s")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the currently loaded model."""
        if self.current_model is None or self.current_config is None:
            return {'error': 'No model loaded'}
        
        memory_info = self._get_memory_usage()
        
        return {
            'model_name': self.current_config.name,
            'model_path': self.current_config.model_path,
            'parameters': self.current_config.parameters,
            'generation': self.current_config.generation,
            'release_date': self.current_config.release_date,
            'max_context_length': self.current_config.max_context_length,
            'memory_usage': memory_info,
            'cache_status': self.current_config.name in self.model_cache
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using current model's tokenizer."""
        if self.current_tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        
        tokens = self.current_tokenizer.encode(text)
        return len(tokens)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated()
            memory_info.update({
                'total_gpu_gb': gpu_memory / (1024**3),
                'used_gpu_gb': gpu_used / (1024**3),
                'gpu_utilization': gpu_used / gpu_memory
            })
        
        # RAM memory
        ram_info = psutil.virtual_memory()
        memory_info.update({
            'total_ram_gb': ram_info.total / (1024**3),
            'used_ram_gb': ram_info.used / (1024**3),
            'ram_utilization': ram_info.percent / 100
        })
        
        return memory_info
    
    def _can_cache_model(self) -> bool:
        """Check if current memory usage allows for caching."""
        memory_info = self._get_memory_usage()
        
        # Don't cache if using >80% of GPU memory or >90% of RAM
        gpu_ok = memory_info.get('gpu_utilization', 0) < 0.8
        ram_ok = memory_info.get('ram_utilization', 0) < 0.9
        
        return gpu_ok and ram_ok
    
    def clear_cache(self) -> None:
        """Clear model cache to free memory."""
        if self.model_cache:
            logger.info(f"Clearing model cache ({len(self.model_cache)} models)")
            self.model_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_cache_status(self) -> Dict[str, any]:
        """Get information about model cache."""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'memory_usage': self._get_memory_usage()
        }
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        if self.current_model is not None:
            logger.debug("Cleaning up model manager")
            self.unload_model()
            self.clear_cache()