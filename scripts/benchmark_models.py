#!/usr/bin/env python
"""Script to benchmark model performance."""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STT_MODEL_ID, STT_ONNX_PATH
from src.stt.whisper_inference import WhisperONNX

def benchmark_whisper(audio_path, num_runs=5):
    """Benchmark Whisper model performance.
    
    Args:
        audio_path: Path to the audio file for testing
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    print(f"Benchmarking Whisper models with {num_runs} runs...")
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found")
        return None
    
    # Load the audio file
    audio, _ = librosa.load(audio_path, sr=16000)
    
    results = {
        "pytorch": {"latencies": []},
        "onnx": {"latencies": []},
    }
    
    # Benchmark PyTorch model
    try:
        print("Loading PyTorch Whisper model...")
        processor = WhisperProcessor.from_pretrained(STT_MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_ID)
        model.eval()
        
        print("Running PyTorch inference...")
        for i in range(num_runs):
            # Prepare input
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            pytorch_time = time.time() - start_time
            
            results["pytorch"]["latencies"].append(pytorch_time)
            print(f"PyTorch Run {i+1}/{num_runs}: {pytorch_time:.4f}s")
        
        # Get transcription for reference
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        results["reference_text"] = transcription
        
    except Exception as e:
        print(f"Error benchmarking PyTorch model: {e}")
        results["pytorch"] = None
    
    # Benchmark ONNX model
    try:
        print("Loading ONNX Whisper model...")
        whisper_onnx = WhisperONNX(STT_ONNX_PATH, STT_MODEL_ID)
        
        print("Running ONNX inference...")
        for i in range(num_runs):
            start_time = time.time()
            _ = whisper_onnx.transcribe(audio_path)
            onnx_time = time.time() - start_time
            
            results["onnx"]["latencies"].append(onnx_time)
            print(f"ONNX Run {i+1}/{num_runs}: {onnx_time:.4f}s")
        
    except Exception as e:
        print(f"Error benchmarking ONNX model: {e}")
        results["onnx"] = None
    
    # Calculate statistics
    if results["pytorch"]:
        pytorch_latencies = results["pytorch"]["latencies"]
        results["pytorch"]["mean"] = np.mean(pytorch_latencies)
        results["pytorch"]["std"] = np.std(pytorch_latencies)
        results["pytorch"]["min"] = np.min(pytorch_latencies)
        results["pytorch"]["max"] = np.max(pytorch_latencies)
    
    if results["onnx"]:
        onnx_latencies = results["onnx"]["latencies"]
        results["onnx"]["mean"] = np.mean(onnx_latencies)
        results["onnx"]["std"] = np.std(onnx_latencies)
        results["onnx"]["min"] = np.min(onnx_latencies)
        results["onnx"]["max"] = np.max(onnx_latencies)
    
    # Calculate speedup
    if results["pytorch"] and results["onnx"]:
        speedup = results["pytorch"]["mean"] / results["onnx"]["mean"]
        results["speedup"] = speedup
        print(f"\nONNX speedup: {speedup:.2f}x faster than PyTorch")
    
    return results

def plot_benchmark_results(results, output_path=None):
    """Plot benchmark results.
    
    Args:
        results: Benchmark results
        output_path: Path to save the plot
    """
    if not results:
        print("No benchmark results to plot")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    models = []
    means = []
    stds = []
    
    if results.get("pytorch"):
        models.append("PyTorch")
        means.append(results["pytorch"]["mean"])
        stds.append(results["pytorch"]["std"])
    
    if results.get("onnx"):
        models.append("ONNX")
        means.append(results["onnx"]["mean"])
        stds.append(results["onnx"]["std"])
    
    # Create bar chart
    plt.bar(models, means, yerr=stds, capsize=10, color=['#4285F4', '#34A853'])
    
    # Add text on bars
    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f"{v:.3f}s", ha='center')
    
    # Add speedup text if available
    if results.get("speedup"):
        plt.text(0.5, max(means) * 1.2, f"ONNX is {results['speedup']:.2f}x faster", 
                 ha='center', fontsize=12, fontweight='bold', color='red')
    
    # Add labels and title
    plt.ylabel('Inference Time (seconds)')
    plt.title('Whisper Model Inference Performance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check for audio file argument
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        print("No audio file provided. Using a default test file if available.")
        # Try to find a test audio file
        test_dir = Path(__file__).parent.parent / "data" / "audio_samples"
        test_files = list(test_dir.glob("*.wav"))
        
        if test_files:
            audio_path = str(test_files[0])
            print(f"Using test file: {audio_path}")
        else:
            print("No test audio files found. Please provide an audio file path.")
            sys.exit(1)
    
    # Run benchmark
    results = benchmark_whisper(audio_path)
    
    if results:
        # Print summary
        print("\nBenchmark Results Summary:")
        print("--------------------------")
        
        if results.get("pytorch"):
            print(f"PyTorch: {results['pytorch']['mean']:.4f}s ± {results['pytorch']['std']:.4f}s")
        
        if results.get("onnx"):
            print(f"ONNX: {results['onnx']['mean']:.4f}s ± {results['onnx']['std']:.4f}s")
        
        if results.get("speedup"):
            print(f"Speedup: {results['speedup']:.2f}x")
        
        # Save results as plot
        output_dir = Path(__file__).parent.parent / "data" / "test_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_dir / "whisper_benchmark.png"
        plot_benchmark_results(results, str(plot_path))