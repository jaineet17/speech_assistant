"""
Environment variable loader utility.
"""

import os
import logging
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger("enhanced_speech_assistant.utils.env_loader")

def load_env_vars():
    """
    Load environment variables from .env file if available.
    """
    try:
        if DOTENV_AVAILABLE:
            # Try to load from .env file
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info(f"Loaded environment variables from {env_path}")
            else:
                logger.warning(f"No .env file found at {env_path}")
        else:
            logger.warning("python-dotenv not installed, skipping .env file loading")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}", exc_info=True)

def load_config_with_env_override(config_path=None):
    """Load configuration from file and override with environment variables.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration values
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.json"
    
    # Load configuration from file
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Load environment variables
    env_vars = load_env_vars()
    
    # Override configuration with environment variables
    
    # API configuration
    if "API_HOST" in env_vars:
        config.setdefault("api", {})["host"] = env_vars["API_HOST"]
    if "API_PORT" in env_vars:
        config.setdefault("api", {})["port"] = int(env_vars["API_PORT"])
    if "API_DEBUG" in env_vars:
        config.setdefault("api", {})["debug"] = env_vars["API_DEBUG"].lower() == "true"
    
    # Whisper configuration
    if "WHISPER_MODEL_ID" in env_vars:
        config.setdefault("models", {}).setdefault("whisper", {})["model_id"] = env_vars["WHISPER_MODEL_ID"]
    if "USE_ONNX" in env_vars:
        config.setdefault("models", {}).setdefault("whisper", {})["use_onnx"] = env_vars["USE_ONNX"].lower() == "true"
    if "USE_INT8" in env_vars:
        config.setdefault("models", {}).setdefault("whisper", {})["use_int8"] = env_vars["USE_INT8"].lower() == "true"
    
    # TTS configuration
    if "TTS_VOICE" in env_vars:
        config.setdefault("models", {}).setdefault("tts", {})["voice"] = env_vars["TTS_VOICE"]
    if "TTS_RATE" in env_vars:
        config.setdefault("models", {}).setdefault("tts", {})["rate"] = int(env_vars["TTS_RATE"])
    
    # LLM configuration
    if "USE_MOCK_LLM" in env_vars:
        config.setdefault("models", {}).setdefault("llm", {})["use_mock"] = env_vars["USE_MOCK_LLM"].lower() == "true"
    if "USE_OLLAMA" in env_vars:
        config.setdefault("models", {}).setdefault("llm", {})["use_ollama"] = env_vars["USE_OLLAMA"].lower() == "true"
    if "OLLAMA_MODEL" in env_vars:
        config.setdefault("models", {}).setdefault("llm", {})["model_name"] = env_vars["OLLAMA_MODEL"]
    if "LLM_TEMPERATURE" in env_vars:
        config.setdefault("models", {}).setdefault("llm", {})["temperature"] = float(env_vars["LLM_TEMPERATURE"])
    if "LLM_MAX_TOKENS" in env_vars:
        config.setdefault("models", {}).setdefault("llm", {})["max_tokens"] = int(env_vars["LLM_MAX_TOKENS"])
    
    return config 