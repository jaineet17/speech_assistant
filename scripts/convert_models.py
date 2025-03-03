#!/usr/bin/env python
"""Script to convert models to ONNX format."""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STT_MODEL_ID, STT_ONNX_PATH, TTS_MODEL_ID, TTS_ONNX_PATH
from src.stt.whisper_converter import convert_whisper_to_onnx
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from src.tts.tts_converter import convert_vits_to_onnx

def convert_whisper():
    """Convert Whisper model to ONNX."""
    print(f"Converting Whisper model: {STT_MODEL_ID}")
    
    # Create output directory
    output_dir = os.path.dirname(STT_ONNX_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model
    model_path = convert_whisper_to_onnx(
        STT_MODEL_ID,
        output_dir,
        quantize=True
    )
    
    print(f"Whisper model converted successfully to {model_path}")
    return model_path

def convert_tts():
    """Convert TTS model to ONNX."""
    print(f"Converting TTS model: {TTS_MODEL_ID}")
    
    # Load TTS model
    try:
        # Get model path from TTS
        tts_obj = TTS(TTS_MODEL_ID, gpu=torch.cuda.is_available())
        model_path = tts_obj.model_path
        config_path = tts_obj.config_path
        
        # Load config
        config = VitsConfig()
        config.load_json(config_path)
        
        # Load model
        model = Vits.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=model_path)
        model.eval()
        
        # Create output directory
        output_dir = os.path.dirname(TTS_ONNX_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert model
        onnx_path = convert_vits_to_onnx(
            model,
            config,
            TTS_ONNX_PATH,
            quantize=True
        )
        
        # Copy config to output directory
        import shutil
        config_dest = os.path.join(output_dir, "config.json")
        shutil.copy(config_path, config_dest)
        
        print(f"TTS model converted successfully to {onnx_path}")
        return onnx_path
    
    except Exception as e:
        print(f"Error converting TTS model: {e}")
        return None

if __name__ == "__main__":
    print("Converting models to ONNX format...")
    
    stt_path = convert_whisper()
    tts_path = convert_tts()
    
    print("\nSummary:")
    print(f"STT ONNX model: {stt_path}")
    print(f"TTS ONNX model: {tts_path}")
    print("\nConversion complete!")