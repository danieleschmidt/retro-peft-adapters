"""Pytest configuration and shared fixtures for retro-peft-adapters tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
import warnings

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress specific warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        yield


@pytest.fixture
def mock_torch():
    """Mock torch module for testing without requiring PyTorch installation."""
    torch_mock = Mock()
    torch_mock.cuda = Mock()
    torch_mock.cuda.is_available = Mock(return_value=False)
    torch_mock.__version__ = "2.0.0"
    return torch_mock


@pytest.fixture
def mock_transformers():
    """Mock transformers module for testing without requiring installation."""
    transformers_mock = Mock()
    transformers_mock.__version__ = "4.30.0"
    return transformers_mock


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model_name": "test-model",
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    }


@pytest.fixture
def temp_index_path(tmp_path):
    """Temporary path for test vector indices."""
    return tmp_path / "test_index.faiss"