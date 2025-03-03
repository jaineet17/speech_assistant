"""Configuration settings for the speech assistant."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model ID
STT_MODEL_ID = "openai/whisper-tiny"  # Can change to "openai/whisper-base" for better accuracy

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORDING_DURATION = 5  # seconds

# API settings
API_HOST = "0.0.0.0"  # Allow external connections
API_PORT = 5000