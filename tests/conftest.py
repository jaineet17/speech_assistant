"""Pytest configuration file."""

import os
import sys
from pathlib import Path
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create necessary directories for tests
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment with necessary directories."""
    project_root = Path(__file__).parent.parent
    
    # Create required directories
    dirs = [
        project_root / "data" / "test_outputs",
        project_root / "models" / "stt",
        project_root / "models" / "tts",
        project_root / "temp",
        project_root / "uploads"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # No cleanup needed as these directories should persist

# Create a pytest fixture for temporary directories
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# Mock for audio processing to avoid actual audio operations during tests
@pytest.fixture
def mock_audio_data():
    """Provide mock audio data for testing."""
    import numpy as np
    # Create a simple sine wave as mock audio data
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio_data, sample_rate 