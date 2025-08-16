"""
Parameter-efficient adapters with retrieval augmentation.

This module implements various adapter types enhanced with retrieval capabilities:
- RetroLoRA: Low-rank adaptation with retrieval integration
- RetroAdaLoRA: Adaptive rank allocation based on retrieval importance
- RetroIA3: Lightweight adapters with retrieval scaling
"""

# Direct imports for common components
try:
    from .simple_adapters import BaseRetroAdapter, RetroLoRA
    _BASE_IMPORTS_AVAILABLE = True
except ImportError:
    _BASE_IMPORTS_AVAILABLE = False

# Safe imports with dependency checking
def __getattr__(name):
    """Lazy import for adapters with proper dependency handling."""
    if name == "BaseRetroAdapter":
        from .simple_adapters import BaseRetroAdapter
        return BaseRetroAdapter
    elif name == "RetroLoRA":
        from .simple_adapters import RetroLoRA
        return RetroLoRA
    elif name == "RetroAdaLoRA":
        try:
            from .retro_adalora import RetroAdaLoRA
            return RetroAdaLoRA
        except ImportError as e:
            raise ImportError(f"RetroAdaLoRA requires additional dependencies: {e}")
    elif name == "RetroIA3":
        try:
            from .retro_ia3 import RetroIA3
            return RetroIA3  
        except ImportError:
            try:
                from .simple_ia3 import RetroIA3
                return RetroIA3
            except ImportError as e:
                raise ImportError(f"RetroIA3 requires additional dependencies: {e}")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BaseRetroAdapter",
    "RetroLoRA",
    "RetroAdaLoRA",
    "RetroIA3",
]
