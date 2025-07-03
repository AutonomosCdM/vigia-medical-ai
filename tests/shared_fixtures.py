"""
Shared test fixtures for VIGIA Medical AI tests.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image_path(temp_directory):
    """Create a sample image file for testing."""
    image_path = temp_directory / "sample_medical_image.jpg"
    # Create a minimal test image file
    image_path.write_bytes(b"fake_image_data_for_testing")
    return image_path


@pytest.fixture
def invalid_patient_codes() -> List[str]:
    """Provide invalid patient codes for testing."""
    return [
        "",  # Empty string
        "INVALID_123",  # Invalid format
        "ABC",  # Too short
        "12345678901234567890",  # Too long
        "PATIENT-INVALID",  # Invalid characters
        None,  # None value
    ]