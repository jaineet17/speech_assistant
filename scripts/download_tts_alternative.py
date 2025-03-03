#!/usr/bin/env python
"""Script to download a TTS model from HuggingFace."""

import os
import sys
from pathlib import Path
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR

def download_hf_tts_model():
    """Download TTS model from HuggingFace."""
    print("Downloading TTS model from HuggingFace...")
    
    save_dir = MODELS_DIR / "tts" / "speecht5"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download models
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Save models locally
        processor.save_pretrained(save_dir / "processor")
        model.save_pretrained(save_dir / "model")
        vocoder.save_pretrained(save_dir / "vocoder")
        
        print(f"TTS model downloaded successfully to {save_dir}")
        
        # Also update the config file to use this model
        config_path = Path(__file__).parent.parent / "src" / "config.py"
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Update model paths
        updated_config = config_content.replace(
            'TTS_MODEL_ID = "tts_models/en/vits/vits-ljspeech"',
            'TTS_MODEL_ID = "microsoft/speecht5_tts"'
        )
        updated_config = updated_config.replace(
            'TTS_ONNX_PATH = str(MODELS_DIR / "tts" / "vits_int8.onnx")',
            'TTS_ONNX_PATH = str(MODELS_DIR / "tts" / "speecht5_int8.onnx")'
        )
        
        with open(config_path, "w") as f:
            f.write(updated_config)
        
        print("Config updated to use HuggingFace TTS model")
        return True
    except Exception as e:
        print(f"Error downloading TTS model: {e}")
        return False

if __name__ == "__main__":
    success = download_hf_tts_model()
    if success:
        print("\nTTS model downloaded successfully!")
    else:
        print("\nFailed to download TTS model. Please check the error messages above.")
