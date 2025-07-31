"""Basic tests to ensure package installation and imports work."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_package_import():
    """Test that the main package can be imported."""
    import retro_peft
    assert retro_peft.__version__ == "0.1.0"


def test_package_metadata():
    """Test package metadata is accessible."""
    import retro_peft
    assert hasattr(retro_peft, "__version__")
    assert hasattr(retro_peft, "__author__")
    assert hasattr(retro_peft, "__email__")