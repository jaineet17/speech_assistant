"""Convert TTS model to ONNX format."""

import os
import torch
import torch.nn as nn
from pathlib import Path
import importlib
from onnxruntime.quantization import quantize_dynamic, QuantType

class VITSExportWrapper(nn.Module):
    """Wrapper class for VITS model export to ONNX."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, x_lengths, scales):
        """Forward pass for ONNX export.
        
        Args:
            x: Text input tensor
            x_lengths: Length of text input
            scales: Scale factors for pitch, energy, and duration
            
        Returns:
            Audio output from the model
        """
        return self.model.inference(x, x_lengths, scales)

def convert_vits_to_onnx(model, config, output_path, quantize=True):
    """Convert VITS model to ONNX format.
    
    Args:
        model: VITS model instance
        config: Model configuration
        output_path: Path to save the ONNX model
        quantize: Whether to apply INT8 quantization
        
    Returns:
        Path to the ONNX model
    """
    print("Converting VITS model to ONNX format...")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wrapper for export
    model.eval()
    export_model = VITSExportWrapper(model)
    
    # Create dummy inputs
    dummy_input_text = torch.randint(0, 100, (1, 100), dtype=torch.long)
    dummy_text_lengths = torch.tensor([100], dtype=torch.long)
    dummy_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        export_model,
        (dummy_input_text, dummy_text_lengths, dummy_scales),
        output_path,
        input_names=["input_text", "text_lengths", "scales"],
        output_names=["output_audio"],
        dynamic_axes={
            "input_text": {0: "batch", 1: "text_length"},
            "text_lengths": {0: "batch"},
            "output_audio": {0: "batch", 1: "audio_length"}
        },
        opset_version=12
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Apply quantization if requested
    if quantize:
        quantized_path = str(output_path).replace(".onnx", "_int8.onnx")
        print(f"Applying INT8 quantization, saving to {quantized_path}...")
        
        quantize_dynamic(
            str(output_path),
            quantized_path,
            weight_type=QuantType.QInt8
        )
        
        return quantized_path
    
    return str(output_path)