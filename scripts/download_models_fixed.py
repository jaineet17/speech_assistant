#!/usr/bin/env python
"""Script to download required models."""

import os
import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import shutil
import subprocess
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STT_MODEL_ID, MODELS_DIR

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

def download_tts_direct():
    """Download TTS model using direct pip command."""
    print("Downloading TTS model using direct approach...")
    
    save_dir = MODELS_DIR / "tts" / "vits"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Install specific TTS model using pip
        print("Installing TTS model via pip (this might take a few minutes)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "TTS==0.17.0"],
            check=True
        )
        
        # Use a direct approach to get the model files
        from TTS.utils.manage import ModelManager
        from TTS.utils.download import download_model
        
        # Get the model manager and list available models
        model_manager = ModelManager()
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Use a more reliable model
        
        # Download the model
        model_path, config_path, _ = download_model(model_name)
        
        # Copy files to our directory
        shutil.copy(model_path, save_dir / "model.pth")
        shutil.copy(config_path, save_dir / "config.json")
        
        print(f"TTS model downloaded successfully to {save_dir}")
        print(f"Model file: {model_path}")
        print(f"Config file: {config_path}")
        return True
    except Exception as e:
        print(f"Error downloading TTS model: {e}")
        print(f"Detailed error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Downloading models...")
    
    stt_success = download_whisper()
    tts_success = download_tts_direct()
    
    if stt_success and tts_success:
        print("\nAll models downloaded successfully!")
    else:
        print("\nSome models could not be downloaded. Please check the error messages above.")
