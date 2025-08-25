"""
Simple adapter implementations that work without heavy dependencies.
Enhanced with robust error handling, validation, and monitoring.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..utils.validation import InputValidator, ValidationError
from ..utils.error_handling import ErrorHandler, resilient_operation, AdapterError

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class BaseRetroAdapter(ABC):
    """
    Simple base class for retrieval-augmented adapters.
    
    Enhanced with robust error handling, validation, and monitoring.
    Works with or without PyTorch dependencies.
    """
    
    def __init__(self, model_name: str = "base", **kwargs):
        """
        Initialize base adapter with validation and error handling.
        
        Args:
            model_name: Name of the base model
            **kwargs: Additional configuration parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Set up logging and error handling
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(self.logger)
        
        # Validate inputs
        try:
            self.model_name = InputValidator.validate_model_name(model_name)
            self.config = InputValidator.validate_adapter_config(kwargs)
        except ValidationError as e:
            self.logger.error(f"Validation failed during adapter initialization: {e}")
            raise AdapterError(f"Invalid adapter configuration: {e}")
        
        # Initialize state
        self.retriever = None
        self._generation_count = 0
        self._error_count = 0
        self._last_error = None
        
        # Log successful initialization
        self.logger.info(
            f"Initialized {self.__class__.__name__} with model '{self.model_name}'",
            extra={"config": self.config}
        )
    
    def set_retriever(self, retriever):
        """
        Set the retrieval component with validation.
        
        Args:
            retriever: Retrieval component instance
            
        Raises:
            AdapterError: If retriever is invalid
        """
        if retriever is None:
            self.logger.warning("Setting retriever to None - retrieval will be disabled")
            self.retriever = None
            return
        
        # Basic validation - check if retriever has required methods
        required_methods = ['search', 'get_document_count']
        for method in required_methods:
            if not hasattr(retriever, method):
                raise AdapterError(f"Retriever missing required method: {method}")
        
        self.retriever = retriever
        self.logger.info(
            f"Connected retriever with {retriever.get_document_count()} documents"
        )
    
    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Forward pass - to be implemented by subclasses"""
        pass
    
    @resilient_operation(context="generate", max_retries=2, default_return=None)
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response with optional retrieval augmentation.
        
        Enhanced with validation, error handling, and monitoring.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results and metadata
            
        Raises:
            AdapterError: If generation fails
        """
        try:
            # Validate input
            if not isinstance(prompt, str):
                raise ValidationError(f"Prompt must be string, got {type(prompt)}")
            
            if len(prompt.strip()) == 0:
                raise ValidationError("Prompt cannot be empty")
            
            # Sanitize prompt
            clean_prompt = InputValidator.validate_text_content(prompt, max_length=10000)
            
            # Track generation attempt
            self._generation_count += 1
            
            # Retrieve context if retriever is available
            context = []
            retrieval_time = 0.0
            
            if self.retriever:
                try:
                    import time
                    start_time = time.time()
                    
                    results = self.retriever.search(clean_prompt, k=kwargs.get('retrieval_k', 3))
                    context = [result['text'] for result in results]
                    
                    retrieval_time = time.time() - start_time
                    
                    self.logger.debug(
                        f"Retrieved {len(context)} documents in {retrieval_time:.3f}s"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Retrieval failed, continuing without context: {e}")
                    context = []
            
            # Simple mock generation with validation
            augmented_prompt = clean_prompt
            if context:
                # Limit context to prevent token overflow
                context_str = " ".join(context[:2])[:1000]  # Limit context length
                augmented_prompt = f"Context: {context_str}\n\nQuestion: {clean_prompt}"
            
            # Generate response
            generated_text = f"Generated response for: {augmented_prompt}"
            response = {
                "input": clean_prompt,
                "output": generated_text,
                "generated_text": generated_text,  # Add this field for quality gates compatibility
                "context_used": len(context) > 0,
                "context_sources": len(context),
                "retrieval_time": retrieval_time,
                "generation_count": self._generation_count,
                "model_name": self.model_name
            }
            
            self.logger.info(
                f"Generation completed successfully",
                extra={
                    "prompt_length": len(clean_prompt),
                    "context_sources": len(context),
                    "retrieval_time": retrieval_time
                }
            )
            
            return response
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            
            self.logger.error(f"Generation failed: {e}")
            
            if isinstance(e, ValidationError):
                raise AdapterError(f"Invalid input: {e}")
            else:
                raise AdapterError(f"Generation error: {e}")


class RetroLoRA(BaseRetroAdapter):
    """
    Simple LoRA adapter with retrieval augmentation.
    
    Mock implementation that demonstrates the interface.
    """
    
    def __init__(
        self, 
        model_name: str = "retro_lora",
        rank: int = 16,
        alpha: float = 32.0,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
    
    def forward(self, inputs: Any) -> Any:
        """Simple forward pass"""
        if isinstance(inputs, str):
            return f"RetroLoRA processed: {inputs}"
        return inputs
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Mock training step"""
        return {
            "loss": 0.1,
            "retrieval_loss": 0.02,
            "generation_loss": 0.08
        }
    
    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        # Mock calculation
        base_params = 1000
        lora_params = 2 * self.rank * 768  # Assuming hidden_size=768
        return base_params + lora_params


class MockAdapter(BaseRetroAdapter):
    """
    Minimal adapter implementation for testing and demonstrations.
    """
    
    def __init__(self, name: str = "mock", **kwargs):
        super().__init__(model_name=name, **kwargs)
        self.call_count = 0
    
    def forward(self, inputs: Any) -> Any:
        """Simple forward pass that tracks calls"""
        self.call_count += 1
        return f"Mock adapter {self.model_name} call #{self.call_count}: {inputs}"
    
    def reset_stats(self):
        """Reset call counter"""
        self.call_count = 0


# Create aliases for backward compatibility
if _TORCH_AVAILABLE:
    # If torch is available, create proper torch-based adapters
    class TorchBaseRetroAdapter(nn.Module, BaseRetroAdapter):
        def __init__(self, **kwargs):
            nn.Module.__init__(self)
            BaseRetroAdapter.__init__(self, **kwargs)
    
    # Override with torch version if available
    BaseRetroAdapter = TorchBaseRetroAdapter