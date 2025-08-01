"""Comprehensive tests for retro-peft-adapters package."""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import time
import warnings
import re

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPackageBasics:
    """Test basic package functionality."""

    def test_package_import(self):
        """Test that the main package can be imported."""
        import retro_peft
        assert retro_peft.__version__ == "0.1.0"

    def test_package_metadata(self):
        """Test package metadata is accessible."""
        import retro_peft
        assert hasattr(retro_peft, "__version__")
        assert hasattr(retro_peft, "__author__")
        assert hasattr(retro_peft, "__email__")
        assert retro_peft.__author__ == "Daniel Schmidt"
        assert "terragonlabs.com" in retro_peft.__email__

    def test_package_all_exports(self):
        """Test that __all__ contains expected exports."""
        import retro_peft
        expected_exports = ["__version__", "__author__", "__email__"]
        assert retro_peft.__all__ == expected_exports


class TestVersioning:
    """Test version-related functionality."""

    def test_version_format(self):
        """Test version follows semantic versioning."""
        import retro_peft
        
        # Check if version follows semver pattern (X.Y.Z)
        semver_pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
        assert re.match(semver_pattern, retro_peft.__version__)

    def test_version_consistency(self):
        """Test version consistency across package files."""
        import retro_peft
        
        # Read version from pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                content = f.read()
                # Simple version extraction (more robust parsing would use toml)
                match = re.search(r'version = "([^"]+)"', content)
                if match:
                    pyproject_version = match.group(1)
                    assert retro_peft.__version__ == pyproject_version


class TestImportSafety:
    """Test that imports are safe and don't have side effects."""

    def test_import_no_external_deps(self):
        """Test that basic import doesn't require external dependencies."""
        # This should work even without torch, transformers etc installed
        import retro_peft
        assert retro_peft is not None

    def test_no_warnings_on_import(self):
        """Test that importing doesn't generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import retro_peft
            
            # Filter out deprecation warnings from dependencies
            relevant_warnings = [
                warning for warning in w 
                if 'retro_peft' in str(warning.filename)
            ]
            assert len(relevant_warnings) == 0


@pytest.mark.parametrize("attribute", ["__version__", "__author__", "__email__"])
def test_required_attributes_exist(attribute):
    """Parametrized test for required package attributes."""
    import retro_peft
    assert hasattr(retro_peft, attribute)
    assert getattr(retro_peft, attribute) is not None
    assert len(str(getattr(retro_peft, attribute))) > 0