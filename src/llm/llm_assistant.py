import os
import logging
import time
import threading
import functools
import json
import hashlib
import queue
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "llama-2-7b-chat.Q4_0.gguf")

# Advanced caching with LRU implementation
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: str) -> None:
        with self.lock:
            if key in self.cache:
                # Remove existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
            # Add new item
            self.cache[key] = value

# Global response cache
_response_cache = LRUCache(100)  # Increased cache size

# Inference queue for batched processing
_inference_queue = queue.Queue()
_inference_workers = []
_MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)
_SHUTDOWN_EVENT = threading.Event()

# Model configuration
MODEL_CONFIG = {
    "context_length": 4096,
    "batch_size": 512,
    "use_mlock": True,
    "seed": 42,
    "rope_scaling_type": 1,  # Scale attention by frequency
    "rope_freq_base": 10000,
    "rope_freq_scale": 0.5,
}

# Performance metrics tracking
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_tokens_generated": 0,
            "total_processing_time": 0,
            "avg_tokens_per_second": 0,
        }
        self.lock = threading.Lock()
    
    def record_request(self, cached: bool, tokens_generated: int, processing_time: float):
        with self.lock:
            self.metrics["total_requests"] += 1
            if cached:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["total_tokens_generated"] += tokens_generated
                self.metrics["total_processing_time"] += processing_time
                if self.metrics["total_processing_time"] > 0:
                    self.metrics["avg_tokens_per_second"] = (
                        self.metrics["total_tokens_generated"] / self.metrics["total_processing_time"]
                    )
    
    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return self.metrics.copy()

# Global performance tracker
_performance_tracker = PerformanceTracker()

class LlamaAssistant:
    """LLM-based assistant using Llama model with advanced optimizations."""
    
    def __init__(self):
        """Initialize the Llama assistant."""
        self.model = None
        self.initialized = False
        self.model_lock = threading.RLock()
        self.tokenizer_cache = {}  # Cache for tokenization results
        self.prompt_templates = self._load_prompt_templates()
        self.initialize()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different scenarios."""
        return {
            "default": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{query} [/INST]""",
            "concise": """<s>[INST] <<SYS>>
You are a helpful assistant that provides concise, accurate responses. Be direct and to the point.
<</SYS>>

{query} [/INST]""",
            "technical": """<s>[INST] <<SYS>>
You are a technical assistant with expertise in programming, technology, and computer science. Provide detailed technical explanations when appropriate.
<</SYS>>

{query} [/INST]"""
        }
    
    def _detect_prompt_type(self, query: str) -> str:
        """Detect the appropriate prompt template based on query content."""
        # Simple heuristic for prompt type detection
        technical_keywords = ["code", "programming", "function", "algorithm", "python", "javascript", "api"]
        if any(keyword in query.lower() for keyword in technical_keywords):
            return "technical"
        
        if len(query.split()) < 10:
            return "concise"
            
        return "default"
    
    def initialize(self):
        """Initialize the Llama model with optimized settings."""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at {MODEL_PATH}")
                return False
            
            logger.info(f"Loading Llama model from {MODEL_PATH}")
            
            # Determine optimal number of threads based on CPU cores
            n_threads = min(8, max(4, multiprocessing.cpu_count() - 1))
            
            # Check for GPU availability and optimize accordingly
            n_gpu_layers = self._detect_gpu_capabilities()
            
            with self.model_lock:
                # Import here to avoid loading unnecessary modules if not used
                from llama_cpp import Llama
                
                self.model = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=MODEL_CONFIG["context_length"],
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=MODEL_CONFIG["batch_size"],
                    use_mlock=MODEL_CONFIG["use_mlock"],
                    seed=MODEL_CONFIG["seed"],
                    rope_scaling_type=MODEL_CONFIG.get("rope_scaling_type"),
                    rope_freq_base=MODEL_CONFIG.get("rope_freq_base"),
                    rope_freq_scale=MODEL_CONFIG.get("rope_freq_scale"),
                    logits_all=True,  # Enable logits for all tokens for better sampling
                    embedding=False,  # Disable embedding to save memory
                )
                
                # Warm up the model with a simple inference
                _ = self.model("Hello", max_tokens=1)
                
                self.initialized = True
                logger.info(f"Llama model loaded successfully with {n_threads} threads and {n_gpu_layers} GPU layers")
                return True
        except Exception as e:
            logger.error(f"Error initializing Llama model: {e}")
            return False
    
    def _detect_gpu_capabilities(self) -> int:
        """Detect and configure GPU capabilities."""
        n_gpu_layers = 0
        
        try:
            # Try PyTorch first
            try:
                import torch
                if torch.cuda.is_available():
                    n_gpu_layers = 32  # Use all layers on GPU
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
                    
                    # Adjust layers based on available GPU memory
                    if gpu_memory < 4:
                        n_gpu_layers = 16
                    elif gpu_memory < 8:
                        n_gpu_layers = 24
            except (ImportError, Exception) as e:
                logger.debug(f"PyTorch GPU detection failed: {e}")
            
            # Try Metal for Mac
            if n_gpu_layers == 0 and os.path.exists("/System/Library/Frameworks/Metal.framework"):
                logger.info("Metal framework detected, enabling GPU acceleration")
                n_gpu_layers = 1  # Enable Metal acceleration
        
        except Exception as e:
            logger.warning(f"Error detecting GPU capabilities: {e}")
        
        logger.info(f"GPU acceleration: {'Enabled with ' + str(n_gpu_layers) + ' layers' if n_gpu_layers > 0 else 'Disabled'}")
        return n_gpu_layers
    
    def _get_cache_key(self, query: str, max_tokens: int, temperature: float, prompt_type: str) -> str:
        """Generate a cache key based on input parameters"""
        # Create a unique key based on the query and generation parameters
        key_data = f"{query}|{max_tokens}|{temperature}|{prompt_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _optimize_generation_params(self, query: str, base_max_tokens: int, base_temperature: float) -> Dict[str, Any]:
        """Optimize generation parameters based on query characteristics."""
        params = {
            "max_tokens": base_max_tokens,
            "temperature": base_temperature,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "stop": ["</s>", "[INST]"],
        }
        
        # Adjust parameters based on query length and complexity
        query_length = len(query)
        query_words = len(query.split())
        
        # For shorter queries, use lower temperature for more focused responses
        if query_length < 50:
            params["temperature"] = max(0.1, base_temperature - 0.3)
            params["top_p"] = 0.85
        
        # For longer queries, adjust max_tokens based on query length
        if query_length > 200:
            params["max_tokens"] = min(2048, base_max_tokens + 512)
            params["repeat_penalty"] = 1.2  # Increase repetition penalty for longer outputs
        
        # For complex questions, increase diversity
        if "?" in query and query_words > 15:
            params["temperature"] = min(0.9, base_temperature + 0.1)
            params["top_p"] = 0.98
        
        # For technical content, optimize for precision
        if any(kw in query.lower() for kw in ["code", "programming", "technical", "algorithm"]):
            params["temperature"] = max(0.1, base_temperature - 0.2)
            params["repeat_penalty"] = 1.15
        
        return params
    
    def generate_response(self, query: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response using the Llama model with advanced optimizations."""
        if not self.initialized:
            logger.warning("Llama model not initialized. Attempting to initialize...")
            if not self.initialize():
                return "I'm sorry, but I'm having trouble accessing my knowledge. Please try again later."
        
        # Detect prompt type
        prompt_type = self._detect_prompt_type(query)
        
        # Check cache first
        cache_key = self._get_cache_key(query, max_tokens, temperature, prompt_type)
        cached_response = _response_cache.get(cache_key)
        if cached_response:
            logger.info("Using cached response")
            _performance_tracker.record_request(cached=True, tokens_generated=0, processing_time=0)
            return cached_response
        
        try:
            # Format the prompt using the appropriate template
            prompt_template = self.prompt_templates[prompt_type]
            prompt = prompt_template.format(query=query)
            
            # Optimize generation parameters
            gen_params = self._optimize_generation_params(query, max_tokens, temperature)
            
            logger.info(f"Generating response for query: {query[:100]}... (prompt type: {prompt_type})")
            start_time = time.time()
            
            # Generate response with optimized parameters
            with self.model_lock:
                response = self.model(
                    prompt,
                    **gen_params
                )
            
            # Extract the generated text
            generated_text = response["choices"][0]["text"].strip()
            
            # Post-process the response
            generated_text = self._post_process_response(generated_text)
            
            # Log performance metrics
            duration = time.time() - start_time
            tokens_generated = len(response["choices"][0]["text"].split())
            tokens_per_second = tokens_generated / duration if duration > 0 else 0
            logger.info(f"Response generated in {duration:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            
            # Update performance tracker
            _performance_tracker.record_request(
                cached=False, 
                tokens_generated=tokens_generated, 
                processing_time=duration
            )
            
            # Cache the response
            _response_cache.put(cache_key, generated_text)
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again later."
    
    def _post_process_response(self, text: str) -> str:
        """Clean and improve the generated response."""
        # Remove any trailing incomplete sentences
        if text and not any(text.endswith(p) for p in ['.', '!', '?', ':', ';', '"', "'", ')', ']', '}']):
            # Find the last complete sentence
            for end_char in ['.', '!', '?']:
                last_idx = text.rfind(end_char)
                if last_idx > len(text) * 0.75:  # Only truncate if we're not losing too much
                    text = text[:last_idx+1]
                    break
        
        # Remove any repeated phrases (a common issue in LLM outputs)
        lines = text.split('\n')
        filtered_lines = []
        prev_line = ""
        
        for line in lines:
            if line != prev_line and not (prev_line and line in prev_line):
                filtered_lines.append(line)
            prev_line = line
        
        return '\n'.join(filtered_lines)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return _performance_tracker.get_metrics()

# Enhanced singleton with thread-safe lazy initialization
class EnhancedSingleton:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LlamaAssistant()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing or reloading)."""
        with cls._lock:
            if cls._instance is not None:
                del cls._instance
                cls._instance = None

def get_llm_response(query: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Get a response from the LLM assistant with state-of-the-art optimizations."""
    return EnhancedSingleton.get_instance().generate_response(query, max_tokens, temperature)

def get_performance_metrics() -> Dict[str, Any]:
    """Get current LLM performance metrics."""
    return EnhancedSingleton.get_instance().get_performance_metrics() 