#!/usr/bin/env python
"""A simpler approach to convert Whisper to ONNX."""

import os
import torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import onnxruntime as ort
import numpy as np

# Project paths
project_root = Path(__file__).parent.parent
output_dir = project_root / "models" / "stt" / "whisper_onnx"
output_dir.mkdir(parents=True, exist_ok=True)

# Model parameters
model_id = "openai/whisper-tiny"
output_path = output_dir / "model.onnx"

# Load PyTorch model
print(f"Loading model: {model_id}")
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.eval()

# Create input for export - using the encoder part only
dummy_input = torch.zeros((1, 80, 3000), dtype=torch.float32)

# Export to ONNX - encoder only
print(f"Exporting to ONNX: {output_path}")
with torch.no_grad():
    # We'll only export the encoder part 
    encoder = model.get_encoder()
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        input_names=["input_features"],
        output_names=["encoder_output"],
        dynamic_axes={
            "input_features": {0: "batch_size", 2: "sequence_length"},
            "encoder_output": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )

print(f"ONNX model saved to {output_path}")

# Test the ONNX model
print("Testing ONNX model...")
session = ort.InferenceSession(str(output_path))
onnx_input = dummy_input.numpy()

# Run inference
onnx_outputs = session.run(None, {"input_features": onnx_input})

print("ONNX conversion successful! The encoder part is now optimized.")
print("This will speed up the audio encoding part of the Whisper model.")