"""
Simple RetroIA3 mock for basic functionality.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

class RetroIA3:
    """Mock RetroIA3 implementation"""
    
    def __init__(self, *args, **kwargs):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "RetroIA3 requires additional dependencies: "
                "pip install torch transformers peft"
            )
        print("Warning: Using mock RetroIA3 implementation")
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)