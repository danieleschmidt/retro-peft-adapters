"""
Training infrastructure for retrieval-augmented adapters.

Provides specialized trainers for different training objectives:
- Contrastive retrieval training
- Multi-task learning
- Meta-learning for few-shot adaptation
- Curriculum learning with progressive difficulty
"""

from .contrastive import ContrastiveRetrievalTrainer
from .multi_task import MultiTaskRetroTrainer

__all__ = [
    "ContrastiveRetrievalTrainer",
    "MultiTaskRetroTrainer",
]


# Lazy imports for optional trainers
def __getattr__(name):
    """Lazy import for optional training components."""
    if name == "MetaLearningTrainer":
        from .meta_learning import MetaLearningTrainer

        return MetaLearningTrainer
    elif name == "CurriculumTrainer":
        from .curriculum import CurriculumTrainer

        return CurriculumTrainer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
