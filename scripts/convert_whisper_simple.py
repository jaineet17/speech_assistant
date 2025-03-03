#!/usr/bin/env python
"""Simplified script to convert Whisper to ONNX format."""

import os
import sys
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STT_MODEL_ID

def convert_whisper_simple():
    """Convert Whisper model to ONNX with minimal dependencies."""
    print(f"Converting {STT_MODEL_ID} to ONNX format...")
    
    # Create output directory
    output_dir = Path("models/stt/whisper_onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "model.onnx"
    
    # Load model from local directory if possible, otherwise from HF
    local_model_dir = Path("models/stt/whisper")
    if local_model_dir.exists():
        print(f"Loading from local directory: {local_model_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(str(local_model_dir))
    else:
        print(f"Loading from HuggingFace: {STT_MODEL_ID}")
        model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_ID)
    
    # Prepare dummy inputs
    processor = WhisperProcessor.from_pretrained(STT_MODEL_ID)
    dummy_input = torch.zeros((1, 80, 3000), dtype=torch.float32)
    
    # Export to ONNX directly using torch.onnx.export
    print(f"Exporting to ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch_size", 2: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14
        )
    
    print(f"ONNX model saved to {onnx_path}")
    return str(onnx_path)

if __name__ == "__main__":
    print("Converting Whisper model to ONNX format...")
    
    try:
        onnx_path = convert_whisper_simple()
        print(f"Conversion successful! Model saved to: {onnx_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
