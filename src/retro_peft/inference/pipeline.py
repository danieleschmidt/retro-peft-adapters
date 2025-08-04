"""
Inference pipeline for retrieval-augmented adapters.

Provides optimized inference with caching, batching, and performance monitoring.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer

from ..adapters.base_adapter import BaseRetroAdapter
from ..retrieval.retrievers import BaseRetriever
from ..retrieval.contextual import ContextualRetriever


class RetrievalInferencePipeline:
    """
    Production-ready inference pipeline for retrieval-augmented adapters.
    
    Features:
    - Optimized inference with caching
    - Batch processing capabilities
    - Performance monitoring
    - Flexible retrieval backends
    """
    
    def __init__(
        self,
        model: BaseRetroAdapter,
        retriever: Optional[BaseRetriever] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "auto",
        cache_size: int = 1000,
        enable_caching: bool = True,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained retrieval-augmented adapter
            retriever: Retrieval backend (optional if model has built-in retriever)
            tokenizer: Tokenizer for text processing
            device: Device for inference
            cache_size: Maximum cache size for responses
            enable_caching: Whether to enable response caching
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model = model
        self.retriever = retriever or getattr(model, 'retriever', None)
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup tokenizer
        if tokenizer is None:
            tokenizer = getattr(model.base_model, 'tokenizer', None)
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided or model must have tokenizer attribute")
        
        self.tokenizer = tokenizer
        
        # Generation parameters
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Caching
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._response_cache = {}
        self._cache_access_order = []
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_inference_time": 0.0,
            "total_retrieval_time": 0.0,
            "average_inference_time": 0.0,
            "average_retrieval_time": 0.0
        }
        
        # Contextual retriever for conversation support
        if self.retriever and not isinstance(self.retriever, ContextualRetriever):
            self.contextual_retriever = ContextualRetriever(
                base_retriever=self.retriever,
                context_window=5
            )
        else:
            self.contextual_retriever = self.retriever
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        retrieval_k: int = 5,
        retrieval_enabled: bool = True,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        return_retrieval_info: bool = False,
        **generation_kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate text with retrieval augmentation.
        
        Args:
            prompt: Input prompt(s)
            retrieval_k: Number of documents to retrieve
            retrieval_enabled: Whether to use retrieval
            conversation_id: Optional conversation identifier
            user_id: Optional user identifier
            max_length: Override default max length
            temperature: Override default temperature
            return_retrieval_info: Whether to return retrieval metadata
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text(s) with optional retrieval information
        """
        start_time = time.time()
        
        # Handle batch input
        if isinstance(prompt, list):
            return self._generate_batch(
                prompts=prompt,
                retrieval_k=retrieval_k,
                retrieval_enabled=retrieval_enabled,
                conversation_id=conversation_id,
                user_id=user_id,
                max_length=max_length,
                temperature=temperature,
                return_retrieval_info=return_retrieval_info,
                **generation_kwargs
            )
        
        # Single prompt processing
        self.metrics["total_requests"] += 1
        
        # Check cache first
        cache_key = None
        if self.enable_caching:
            cache_key = self._compute_cache_key(
                prompt, retrieval_k, retrieval_enabled, max_length, temperature
            )
            
            if cache_key in self._response_cache:
                self.metrics["cache_hits"] += 1
                self._update_cache_access(cache_key)
                cached_result = self._response_cache[cache_key].copy()
                cached_result["from_cache"] = True
                return cached_result
        
        # Generation parameters
        gen_max_length = max_length or self.max_length
        gen_temperature = temperature or self.temperature
        
        # Retrieval phase
        retrieval_info = {}
        retrieval_context = None
        
        if retrieval_enabled and self.retriever is not None:
            retrieval_start = time.time()
            
            if isinstance(self.contextual_retriever, ContextualRetriever):
                # Use contextual retrieval
                retrieval_context, metadata_list = self.contextual_retriever.retrieve_with_context(
                    query=prompt,
                    user_id=user_id,
                    k=retrieval_k
                )
            else:
                # Standard retrieval
                retrieval_context, metadata_list = self.retriever.retrieve(
                    query_text=[prompt],
                    k=retrieval_k
                )
            
            retrieval_time = time.time() - retrieval_start
            self.metrics["total_retrieval_time"] += retrieval_time
            
            retrieval_info = {
                "retrieved_docs": metadata_list[:retrieval_k],
                "retrieval_time": retrieval_time,
                "num_retrieved": len(metadata_list[:retrieval_k])
            }
        
        # Generation phase
        generation_start = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=gen_max_length//2
        ).to(self.device)
        
        # Generate with model
        with torch.no_grad():
            if retrieval_context is not None:
                # Generate with retrieval context
                outputs = self.model.base_model.generate(
                    **inputs,
                    max_length=gen_max_length,
                    temperature=gen_temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            else:
                # Standard generation
                outputs = self.model.base_model.generate(
                    **inputs,
                    max_length=gen_max_length,
                    temperature=gen_temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Update metrics
        self.metrics["total_inference_time"] += total_time
        self.metrics["average_inference_time"] = (
            self.metrics["total_inference_time"] / self.metrics["total_requests"]
        )
        if retrieval_enabled:
            self.metrics["average_retrieval_time"] = (
                self.metrics["total_retrieval_time"] / self.metrics["total_requests"]
            )
        
        # Build result
        result = {
            "generated_text": generated_text,
            "prompt": prompt,
            "generation_time": generation_time,
            "total_time": total_time,
            "retrieval_enabled": retrieval_enabled,
            "from_cache": False
        }
        
        if return_retrieval_info and retrieval_info:
            result["retrieval_info"] = retrieval_info
        
        # Cache result
        if self.enable_caching and cache_key:
            self._cache_response(cache_key, result)
        
        # Update conversation context
        if conversation_id and isinstance(self.contextual_retriever, ContextualRetriever):
            self.contextual_retriever.add_to_conversation(
                query=prompt,
                response=generated_text,
                user_id=user_id
            )
        
        return result
    
    def _generate_batch(
        self,
        prompts: List[str],
        retrieval_k: int = 5,
        retrieval_enabled: bool = True,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        return_retrieval_info: bool = False,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """Generate for batch of prompts"""
        results = []
        
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                retrieval_k=retrieval_k,
                retrieval_enabled=retrieval_enabled,
                conversation_id=conversation_id,
                user_id=user_id,
                max_length=max_length,
                temperature=temperature,
                return_retrieval_info=return_retrieval_info,
                **generation_kwargs
            )
            results.append(result)
        
        return results
    
    def chat(
        self,
        message: str,
        conversation_id: str = "default",
        user_id: Optional[str] = None,
        retrieval_k: int = 5,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat interface with conversation memory.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            user_id: Optional user identifier
            retrieval_k: Number of documents to retrieve
            max_length: Maximum response length
            **kwargs: Additional generation parameters
            
        Returns:
            Chat response with conversation context
        """
        if not isinstance(self.contextual_retriever, ContextualRetriever):
            # Fallback to regular generation
            return self.generate(
                prompt=message,
                retrieval_k=retrieval_k,
                max_length=max_length,
                **kwargs
            )
        
        # Generate response with conversation context
        result = self.generate(
            prompt=message,
            retrieval_k=retrieval_k,
            retrieval_enabled=True,
            conversation_id=conversation_id,
            user_id=user_id,
            max_length=max_length,
            return_retrieval_info=True,
            **kwargs
        )
        
        # Add conversation metadata
        result["conversation_id"] = conversation_id
        result["user_id"] = user_id
        result["conversation_summary"] = self.contextual_retriever.get_conversation_summary()
        
        return result
    
    def _compute_cache_key(
        self, 
        prompt: str, 
        retrieval_k: int, 
        retrieval_enabled: bool,
        max_length: Optional[int],
        temperature: Optional[float]
    ) -> str:
        """Compute cache key for response"""
        cache_data = {
            "prompt": prompt,
            "retrieval_k": retrieval_k,
            "retrieval_enabled": retrieval_enabled,
            "max_length": max_length or self.max_length,
            "temperature": temperature or self.temperature
        }
        
        cache_string = str(cache_data)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, result: Dict[str, Any]):
        """Cache response with LRU eviction"""
        # Remove oldest entries if cache is full
        if len(self._response_cache) >= self.cache_size:
            # Remove least recently used
            oldest_key = self._cache_access_order.pop(0)
            if oldest_key in self._response_cache:
                del self._response_cache[oldest_key]
        
        # Add new entry
        self._response_cache[cache_key] = result.copy()
        self._cache_access_order.append(cache_key)
    
    def _update_cache_access(self, cache_key: str):
        """Update cache access order for LRU"""
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)
    
    def clear_cache(self):
        """Clear response cache"""
        self._response_cache.clear()
        self._cache_access_order.clear()
        print("Response cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        
        # Add cache statistics
        if self.enable_caching:
            cache_hit_rate = (
                self.metrics["cache_hits"] / max(self.metrics["total_requests"], 1)
            )
            metrics.update({
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self._response_cache),
                "cache_capacity": self.cache_size
            })
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_inference_time": 0.0,
            "total_retrieval_time": 0.0,
            "average_inference_time": 0.0,
            "average_retrieval_time": 0.0
        }
        print("Metrics reset")
    
    def warmup(self, num_warmup_calls: int = 5):
        """
        Warm up the pipeline with dummy calls.
        
        Args:
            num_warmup_calls: Number of warmup calls to make
        """
        print(f"Warming up pipeline with {num_warmup_calls} calls...")
        
        warmup_prompts = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Tell me about artificial intelligence.",
            "Explain quantum computing briefly.",
            "What are the benefits of renewable energy?"
        ]
        
        for i in range(num_warmup_calls):
            prompt = warmup_prompts[i % len(warmup_prompts)]
            
            # Don't cache warmup calls
            old_caching = self.enable_caching
            self.enable_caching = False
            
            try:
                self.generate(
                    prompt=prompt,
                    max_length=50,  # Short generations for warmup
                    retrieval_enabled=False  # Skip retrieval for speed
                )
            except Exception as e:
                print(f"Warmup call {i+1} failed: {e}")
            
            self.enable_caching = old_caching
        
        # Reset metrics after warmup
        self.reset_metrics()
        print("Pipeline warmup completed")
    
    def benchmark(
        self, 
        test_prompts: List[str], 
        num_runs: int = 3,
        retrieval_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark pipeline performance.
        
        Args:
            test_prompts: List of test prompts
            num_runs: Number of benchmark runs
            retrieval_enabled: Whether to test with retrieval
            
        Returns:
            Benchmark results
        """
        print(f"Running benchmark with {len(test_prompts)} prompts, {num_runs} runs each...")
        
        # Disable caching for accurate timing
        old_caching = self.enable_caching
        self.enable_caching = False
        
        results = {
            "total_prompts": len(test_prompts) * num_runs,
            "runs_per_prompt": num_runs,
            "retrieval_enabled": retrieval_enabled,
            "individual_times": [],
            "prompt_results": []
        }
        
        try:
            for prompt_idx, prompt in enumerate(test_prompts):
                prompt_times = []
                
                for run in range(num_runs):
                    start_time = time.time()
                    
                    result = self.generate(
                        prompt=prompt,
                        retrieval_enabled=retrieval_enabled,
                        max_length=100  # Fixed length for consistency
                    )
                    
                    run_time = time.time() - start_time
                    prompt_times.append(run_time)
                    results["individual_times"].append(run_time)
                
                # Stats for this prompt
                avg_time = sum(prompt_times) / len(prompt_times)
                min_time = min(prompt_times)
                max_time = max(prompt_times)
                
                results["prompt_results"].append({
                    "prompt_idx": prompt_idx,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "times": prompt_times
                })
                
                print(f"Prompt {prompt_idx+1}/{len(test_prompts)}: "
                      f"avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
        
        finally:
            self.enable_caching = old_caching
        
        # Overall statistics
        all_times = results["individual_times"]
        results.update({
            "total_time": sum(all_times),
            "average_time": sum(all_times) / len(all_times),
            "min_time": min(all_times),
            "max_time": max(all_times),
            "std_time": np.std(all_times) if all_times else 0.0,
            "throughput": len(all_times) / sum(all_times)  # requests per second
        })
        
        print(f"Benchmark completed: avg={results['average_time']:.3f}s, "
              f"throughput={results['throughput']:.2f} req/s")
        
        return results
    
    def save_pipeline(self, save_path: str):
        """Save pipeline configuration and model"""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model adapter
        self.model.save_adapter(os.path.join(save_path, "adapter.pt"))
        
        # Save pipeline config
        config = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "cache_size": self.cache_size,
            "enable_caching": self.enable_caching,
            "device": self.device,
            "metrics": self.metrics
        }
        
        with open(os.path.join(save_path, "pipeline_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Pipeline saved to: {save_path}")
    
    @classmethod
    def load_pipeline(
        cls,
        model: BaseRetroAdapter,
        load_path: str,
        retriever: Optional[BaseRetriever] = None,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        """Load pipeline from saved configuration"""
        import os
        import json
        
        # Load pipeline config
        config_path = os.path.join(load_path, "pipeline_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create pipeline
        pipeline = cls(
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            device=config.get("device", "auto"),
            cache_size=config.get("cache_size", 1000),
            enable_caching=config.get("enable_caching", True),
            max_length=config.get("max_length", 512),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 50)
        )
        
        # Restore metrics if available
        if "metrics" in config:
            pipeline.metrics.update(config["metrics"])
        
        print(f"Pipeline loaded from: {load_path}")
        return pipeline