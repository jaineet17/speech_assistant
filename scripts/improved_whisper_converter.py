#!/usr/bin/env python
"""Improved script to convert Whisper models to ONNX with validation."""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
import librosa
import soundfile as sf
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STT_MODEL_ID, MODELS_DIR

def convert_whisper_to_onnx(model_id, output_dir, quantize=True, validate=True):
    """Convert Whisper model to ONNX with validation.
    
    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Directory to save the ONNX model
        quantize: Whether to quantize the model to INT8
        validate: Whether to validate the converted model
        
    Returns:
        Dictionary with model paths and validation results
    """
    print(f"Converting {model_id} to ONNX format...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model paths
    onnx_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"
    
    # Load model
    print("Loading model from PyTorch...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    
    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    with torch.no_grad():
        # Create a sample input (typical spectrogram dimensions)
        dummy_input = torch.zeros((1, 80, 3000), dtype=torch.float32)
        
        # Export the model
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
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
    
    # Quantize if requested
    if quantize:
        print(f"Quantizing model to INT8: {int8_path}")
        quantize_dynamic(
            str(onnx_path),
            str(int8_path),
            weight_type=QuantType.QInt8
        )
    
    results = {
        "model_id": model_id,
        "onnx_path": str(onnx_path),
        "int8_path": str(int8_path) if quantize else None,
        "validation": None
    }
    
    # Validate the models if requested
    if validate:
        print("Validating converted models...")
        validation_results = validate_whisper_onnx(
            processor=processor,
            pytorch_model=model,
            onnx_path=onnx_path,
            int8_path=int8_path if quantize else None
        )
        results["validation"] = validation_results
    
    return results

def validate_whisper_onnx(processor, pytorch_model, onnx_path, int8_path=None):
    """Validate Whisper ONNX models by comparing outputs with PyTorch.
    
    Args:
        processor: Whisper processor
        pytorch_model: PyTorch Whisper model
        onnx_path: Path to the ONNX model
        int8_path: Path to the INT8 quantized model
        
    Returns:
        Dictionary with validation results
    """
    # Create a random input (typical spectrogram dimensions)
    input_features = torch.rand((1, 80, 3000), dtype=torch.float32)
    
    # Get PyTorch model output
    with torch.no_grad():
        pytorch_output = pytorch_model(input_features).logits
    
    # Validate FP32 ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(
        ["logits"], 
        {"input_features": input_features.numpy()}
    )[0]
    
    # Calculate difference
    fp32_diff = np.abs(pytorch_output.numpy() - onnx_output).mean()
    fp32_max_diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    
    results = {
        "fp32": {
            "mean_diff": float(fp32_diff),
            "max_diff": float(fp32_max_diff),
            "path": str(onnx_path)
        }
    }
    
    # Validate INT8 model if provided
    if int8_path:
        ort_session_int8 = ort.InferenceSession(str(int8_path))
        int8_output = ort_session_int8.run(
            ["logits"], 
            {"input_features": input_features.numpy()}
        )[0]
        
        int8_diff = np.abs(pytorch_output.numpy() - int8_output).mean()
        int8_max_diff = np.abs(pytorch_output.numpy() - int8_output).max()
        
        results["int8"] = {
            "mean_diff": float(int8_diff),
            "max_diff": float(int8_max_diff),
            "path": str(int8_path)
        }
    
    # Check if the difference is acceptable
    is_valid_fp32 = fp32_diff < 1e-4
    is_valid_int8 = int8_diff < 1e-2 if int8_path else None
    
    results["is_valid_fp32"] = is_valid_fp32
    results["is_valid_int8"] = is_valid_int8
    
    # Print validation results
    print(f"FP32 ONNX validation - Mean diff: {fp32_diff:.6f}, Max diff: {fp32_max_diff:.6f}")
    if int8_path:
        print(f"INT8 ONNX validation - Mean diff: {int8_diff:.6f}, Max diff: {int8_max_diff:.6f}")
    
    print(f"FP32 ONNX model is {'valid' if is_valid_fp32 else 'INVALID'}")
    if int8_path:
        print(f"INT8 ONNX model is {'valid' if is_valid_int8 else 'INVALID'}")
    
    return results

def benchmark_whisper_models(audio_path, processor, pytorch_model, onnx_path, int8_path=None, num_runs=5):
    """Benchmark Whisper models (PyTorch, ONNX, INT8).
    
    Args:
        audio_path: Path to an audio file for testing
        processor: Whisper processor
        pytorch_model: PyTorch Whisper model
        onnx_path: Path to the ONNX model
        int8_path: Path to the INT8 quantized model
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking Whisper models with {num_runs} runs...")
    
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Prepare input features
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features_np = input_features.numpy()
    
    results = {
        "pytorch": {"latencies": []},
        "onnx_fp32": {"latencies": []},
        "onnx_int8": {"latencies": []} if int8_path else None
    }
    
    # Benchmark PyTorch
    print("Benchmarking PyTorch model...")
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = pytorch_model(input_features)
        latency = time.time() - start_time
        results["pytorch"]["latencies"].append(latency)
        print(f"  Run {i+1}/{num_runs}: {latency:.4f}s")
    
    # Benchmark ONNX FP32
    print("Benchmarking ONNX FP32 model...")
    ort_session = ort.InferenceSession(str(onnx_path))
    for i in range(num_runs):
        start_time = time.time()
        _ = ort_session.run(["logits"], {"input_features": input_features_np})
        latency = time.time() - start_time
        results["onnx_fp32"]["latencies"].append(latency)
        print(f"  Run {i+1}/{num_runs}: {latency:.4f}s")
    
    # Benchmark ONNX INT8 if available
    if int8_path:
        print("Benchmarking ONNX INT8 model...")
        ort_session_int8 = ort.InferenceSession(str(int8_path))
        for i in range(num_runs):
            start_time = time.time()
            _ = ort_session_int8.run(["logits"], {"input_features": input_features_np})
            latency = time.time() - start_time
            results["onnx_int8"]["latencies"].append(latency)
            print(f"  Run {i+1}/{num_runs}: {latency:.4f}s")
    
    # Calculate statistics
    for model_key in results:
        if results[model_key]:
            latencies = results[model_key]["latencies"]
            results[model_key]["mean"] = np.mean(latencies)
            results[model_key]["std"] = np.std(latencies)
            results[model_key]["min"] = np.min(latencies)
            results[model_key]["max"] = np.max(latencies)
    
    # Calculate speedups
    pytorch_mean = results["pytorch"]["mean"]
    onnx_fp32_mean = results["onnx_fp32"]["mean"]
    results["speedup_fp32"] = pytorch_mean / onnx_fp32_mean
    
    if int8_path:
        onnx_int8_mean = results["onnx_int8"]["mean"]
        results["speedup_int8"] = pytorch_mean / onnx_int8_mean
        results["speedup_int8_vs_fp32"] = onnx_fp32_mean / onnx_int8_mean
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print(f"PyTorch: {results['pytorch']['mean']:.4f}s ± {results['pytorch']['std']:.4f}s")
    print(f"ONNX FP32: {results['onnx_fp32']['mean']:.4f}s ± {results['onnx_fp32']['std']:.4f}s " +
          f"(Speedup: {results['speedup_fp32']:.2f}x)")
    
    if int8_path:
        print(f"ONNX INT8: {results['onnx_int8']['mean']:.4f}s ± {results['onnx_int8']['std']:.4f}s " +
              f"(Speedup vs PyTorch: {results['speedup_int8']:.2f}x, " +
              f"vs ONNX FP32: {results['speedup_int8_vs_fp32']:.2f}x)")
    
    return results

def find_test_audio():
    """Find a test audio file in the project."""
    # Look in the data directory
    test_dir = Path(__file__).parent.parent / "data" / "audio_samples"
    test_files = list(test_dir.glob("*.wav"))
    
    if test_files:
        return str(test_files[0])
    
    # Create a sample audio file if none exists
    output_dir = Path(__file__).parent.parent / "data" / "audio_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "sample_tone.wav"
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    sf.write(str(output_path), audio, sample_rate)
    print(f"Created sample audio file: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    print("Improved Whisper ONNX Conversion")
    
    # Define model ID and output directory
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = STT_MODEL_ID
    
    output_dir = MODELS_DIR / "stt" / "whisper_onnx"
    
    # Convert model
    results = convert_whisper_to_onnx(
        model_id=model_id,
        output_dir=output_dir,
        quantize=True,
        validate=True
    )
    
    # Find a test audio file
    audio_path = find_test_audio()
    
    # Load models for benchmarking
    processor = WhisperProcessor.from_pretrained(model_id)
    pytorch_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    pytorch_model.eval()
    
    # Benchmark models
    benchmark_results = benchmark_whisper_models(
        audio_path=audio_path,
        processor=processor,
        pytorch_model=pytorch_model,
        onnx_path=results["onnx_path"],
        int8_path=results["int8_path"],
        num_runs=5
    )
    
    print("\nConversion and benchmarking complete!")