#!/usr/bin/env python
"""Script to download required models."""

import os
import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import shutil

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STT_MODEL_ID, TTS_MODEL_ID, MODELS_DIR

def download_whisper():
    """Download Whisper model."""
    print(f"Downloading Whisper model: {STT_MODEL_ID}")
    
    save_dir = MODELS_DIR / "stt" / "whisper"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        processor = WhisperProcessor.from_pretrained(STT_MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_ID)
        
        # Save model locally
        processor.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        print(f"Whisper model downloaded successfully to {save_dir}")
        return True
    except Exception as e:
        print(f"Error downloading Whisper model: {e}")
        return False

def download_tts():
    """Download TTS model."""
    print(f"Downloading TTS model: {TTS_MODEL_ID}")
    
    save_dir = MODELS_DIR / "tts" / "vits"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use TTS library to download the model
        from TTS.api import TTS
        tts_model = TTS(TTS_MODEL_ID, progress_bar=True, gpu=torch.cuda.is_available())
        
        # Copy the model files to our directory
        model_path = tts_model.model_path
        config_path = tts_model.config_path
        vocoder_path = getattr(tts_model, 'vocoder_path', None)
        
        # Copy model file
        shutil.copy(model_path, save_dir / "model.pth")
        
        # Copy config file
        shutil.copy(config_path, save_dir / "config.json")
        
        # Copy vocoder if available
        if vocoder_path and os.path.exists(vocoder_path):
            vocoder_dir = save_dir / "vocoder"
            vocoder_dir.mkdir(exist_ok=True)
            shutil.copy(vocoder_path, vocoder_dir / "model.pth")
            
            # Try to copy vocoder config if it exists
            vocoder_config = os.path.join(os.path.dirname(vocoder_path), "config.json")
            if os.path.exists(vocoder_config):
                shutil.copy(vocoder_config, vocoder_dir / "config.json")
        
        print(f"TTS model downloaded successfully to {save_dir}")
        return True
    except Exception as e:
        print(f"Error downloading TTS model: {e}")
        return False

if __name__ == "__main__":
    print("Downloading models...")
    
    stt_success = download_whisper()
    tts_success = download_tts()
    
    if stt_success and tts_success:
        print("\nAll models downloaded successfully!")
    else:
        print("\nSome models could not be downloaded. Please check the error messages above.")