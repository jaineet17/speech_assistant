"""Convert Whisper model to ONNX format."""

import os
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from onnxruntime.quantization import quantize_dynamic, QuantType

def convert_whisper_to_onnx(model_id, output_dir, quantize=True):
    """Convert Whisper model to ONNX format.
    
    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Directory to save the ONNX model
        quantize: Whether to apply INT8 quantization
        
    Returns:
        Path to the ONNX model
    """
    print(f"Converting {model_id} to ONNX format...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export Whisper to ONNX using Optimum
    onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        from_transformers=True,
        export=True,
        provider="CPUExecutionProvider"
    )
    
    # Save the model
    onnx_path = output_path / "model.onnx"
    onnx_model.save_pretrained(output_path)
    
    # Apply quantization if requested
    if quantize:
        print("Applying INT8 quantization...")
        quantized_path = output_path / "model_int8.onnx"
        
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        
        print(f"Quantized model saved to {quantized_path}")
        return str(quantized_path)
    
    print(f"ONNX model saved to {onnx_path}")
    return str(onnx_path)

if __name__ == "__main__":
    # Example usage
    model_path = convert_whisper_to_onnx(
        "openai/whisper-tiny",
        "models/stt/whisper_tiny_onnx",
        quantize=True
    )
    print(f"Model converted and saved to {model_path}")