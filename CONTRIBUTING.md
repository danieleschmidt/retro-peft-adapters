# Contributing to Retro-PEFT-Adapters

We welcome contributions to the Retro-PEFT-Adapters project! This document provides guidelines for contributing code, reporting issues, and participating in the development process.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of PyTorch and transformers
- Familiarity with parameter-efficient fine-tuning (PEFT) concepts

### Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/retro-peft-adapters.git
   cd retro-peft-adapters
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,test,all-backends]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Development Setup

### Repository Structure

```
retro-peft-adapters/
├── src/retro_peft/          # Main package code
│   ├── adapters/            # Adapter implementations
│   ├── retrieval/           # Retrieval systems
│   ├── training/            # Training utilities
│   └── serving/             # Production serving
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Usage examples
└── benchmarks/              # Performance benchmarks
```

### Environment Configuration

Create a `.env` file for local development (do not commit this file):

```bash
# Development settings
PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0
WANDB_PROJECT=retro-peft-dev
```

## Contribution Workflow

### 1. Issue First

Before starting work on a significant change:

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** describing the problem or enhancement
3. **Discuss the approach** with maintainers
4. **Get approval** before implementing major changes

### 2. Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/description**: Feature development
- **bugfix/description**: Bug fixes
- **hotfix/description**: Critical production fixes

### 3. Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes** following code standards

3. **Write Tests** for new functionality

4. **Update Documentation** as needed

5. **Run Quality Checks**
   ```bash
   # Run tests
   pytest tests/ -v --cov=src/retro_peft
   
   # Run linting
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   
   # Security check
   bandit -r src/
   safety check
   ```

6. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add retrieval-augmented LoRA implementation"
   ```

7. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### 4. Commit Message Convention

We use [Conventional Commits](https://conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(adapters): implement RetroLoRA with cross-attention fusion
fix(retrieval): handle empty query in hybrid retriever
docs(api): add docstrings for RetroAdaLoRA class
```

## Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with the following specifics:

- **Line length**: 100 characters
- **Formatter**: Black
- **Import sorting**: isort with black profile
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting with plugins:
  - flake8-docstrings
  - flake8-type-checking
- **mypy**: Static type checking
- **bandit**: Security analysis

### API Design Principles

1. **Consistency**: Follow existing patterns
2. **Simplicity**: Prefer simple, clear interfaces
3. **Performance**: Consider memory and compute efficiency
4. **Extensibility**: Design for future enhancements
5. **Error Handling**: Provide clear error messages

### Example Code Structure

```python
"""Module for retrieval-augmented adapters."""

from typing import Optional, Dict, Any, List
import torch
from torch import nn
from transformers import PreTrainedModel


class RetroLoRA(nn.Module):
    """Retrieval-augmented LoRA adapter.
    
    Combines traditional LoRA adaptation with retrieval-augmented
    generation for improved domain adaptation.
    
    Args:
        base_model: The base transformer model to adapt
        r: LoRA rank parameter
        alpha: LoRA alpha scaling parameter
        dropout: Dropout probability
        target_modules: List of module names to adapt
        retrieval_dim: Dimension of retrieval embeddings
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> adapter = RetroLoRA(
        ...     base_model=model,
        ...     r=16,
        ...     alpha=32,
        ...     target_modules=["c_attn"]
        ... )
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        retrieval_dim: int = 768,
    ) -> None:
        super().__init__()
        # Implementation details...
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        retrieved_context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with retrieval augmentation."""
        # Implementation details...
```

## Testing

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions  
- **Property Tests**: Use Hypothesis for property-based testing
- **Performance Tests**: Benchmark critical paths
- **End-to-End Tests**: Test complete workflows

### Test Categories

```python
import pytest


class TestRetroLoRA:
    """Unit tests for RetroLoRA adapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
    
    def test_forward_pass(self):
        """Test forward pass without retrieval."""
    
    def test_retrieval_integration(self):
        """Test forward pass with retrieval."""


@pytest.mark.integration
class TestRetroLoRAIntegration:
    """Integration tests for RetroLoRA."""
    
    def test_training_workflow(self):
        """Test complete training workflow."""


@pytest.mark.slow
class TestRetroLoRAPerformance:
    """Performance tests for RetroLoRA."""
    
    def test_inference_speed(self):
        """Benchmark inference performance."""
```

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests

# With coverage
pytest --cov=src/retro_peft --cov-report=html

# Parallel execution
pytest -n auto
```

### Test Data

- Use fixtures for reusable test data
- Mock external dependencies (models, APIs)
- Include edge cases and error conditions
- Test with different model architectures

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def retrieve_and_generate(
    self,
    query: str,
    k: int = 5,
    max_length: int = 100,
) -> str:
    """Generate text with retrieval augmentation.
    
    Retrieves relevant context from the vector database and uses it
    to augment the generation process.
    
    Args:
        query: Input text query
        k: Number of documents to retrieve
        max_length: Maximum generation length
        
    Returns:
        Generated text with retrieval augmentation
        
    Raises:
        ValueError: If k is not positive
        RuntimeError: If retrieval index is not initialized
        
    Example:
        >>> adapter.retrieve_and_generate(
        ...     "Explain quantum computing",
        ...     k=3,
        ...     max_length=200
        ... )
        "Quantum computing leverages quantum mechanics..."
    """
```

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **Tutorials**: Step-by-step guides
3. **Examples**: Code samples and notebooks
4. **Architecture**: Design decisions and patterns
5. **Performance**: Benchmarks and optimization guides

### Building Documentation

```bash
cd docs/
make html  # Build HTML documentation
make livehtml  # Live reload during development
```

## Performance Considerations

### Memory Efficiency

- Use gradient checkpointing for large models
- Implement efficient caching strategies
- Consider mixed precision training
- Profile memory usage in tests

### Compute Optimization

- Vectorize operations where possible
- Use efficient attention mechanisms
- Cache frequently accessed data
- Parallelize independent operations

### Benchmarking

Include performance tests for:

- Training throughput (tokens/second)
- Inference latency (ms per generation)
- Memory usage (peak and average)
- Retrieval speed (queries/second)

## Community Guidelines

### Code Review Process

1. **All changes** require review by a maintainer
2. **Large changes** need multiple reviewers
3. **Security changes** require security team review
4. **Performance changes** need benchmark validation

### Review Criteria

- **Correctness**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Are there performance implications?
- **Security**: Are there security considerations?
- **Style**: Does it follow project conventions?

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Design discussions and questions
- **Discord/Slack**: Real-time community chat
- **Email**: Security issues and private matters

### Getting Help

1. **Check documentation** and existing issues first
2. **Search Discord/Slack** for similar questions
3. **Create detailed issues** with reproduction steps
4. **Join community calls** for complex discussions

## Recognition

Contributors are recognized through:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Mentioned in changelogs
- **Community highlights**: Featured in newsletters
- **Maintainer nomination**: For consistent contributors

Thank you for contributing to Retro-PEFT-Adapters! Your efforts help advance the field of efficient AI adaptation.