"""
Simple RetroAdaLoRA mock for basic functionality.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if not _TORCH_AVAILABLE:
    class RetroAdaLoRA:
        """Mock RetroAdaLoRA when torch not available"""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RetroAdaLoRA requires additional dependencies: "
                "pip install torch transformers peft"
            )
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(*args, **kwargs)
else:
    # Try to import full implementation
    try:
        from .retro_adalora import RetroAdaLoRA as _RetroAdaLoRA
        
        class RetroAdaLoRA(_RetroAdaLoRA):
            """Use full implementation when torch available"""
            pass
    except (ImportError, SyntaxError):
        # Fallback to basic mock
        class RetroAdaLoRA:
            """Basic mock implementation"""
            
            def __init__(self, *args, **kwargs):
                print("Warning: Using mock RetroAdaLoRA implementation")
            
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(*args, **kwargs)