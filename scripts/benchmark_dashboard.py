#!/usr/bin/env python
"""Create a comprehensive benchmark dashboard for speech models."""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, DATA_DIR

def load_benchmark_data(json_path):
    """Load benchmark data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_benchmark_data(data, json_path):
    """Save benchmark data to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    return json_path

def plot_latency_comparison(benchmark_data, title="Model Latency Comparison", output_path=None):
    """Plot latency comparison between models."""
    plt.figure(figsize=(12, 7))
    
    # Extract data
    model_types = []
    mean_latencies = []
    std_latencies = []
    
    if "pytorch" in benchmark_data and benchmark_data["pytorch"]:
        model_types.append("PyTorch")
        mean_latencies.append(benchmark_data["pytorch"]["mean"])
        std_latencies.append(benchmark_data["pytorch"]["std"])
    
    if "onnx_fp32" in benchmark_data and benchmark_data["onnx_fp32"]:
        model_types.append("ONNX FP32")
        mean_latencies.append(benchmark_data["onnx_fp32"]["mean"])
        std_latencies.append(benchmark_data["onnx_fp32"]["std"])
    
    if "onnx_int8" in benchmark_data and benchmark_data["onnx_int8"]:
        model_types.append("ONNX INT8")
        mean_latencies.append(benchmark_data["onnx_int8"]["mean"])
        std_latencies.append(benchmark_data["onnx_int8"]["std"])
    
    # Set colors
    colors = ['#4285F4', '#34A853', '#FBBC05']
    
    # Create bar chart
    x = np.arange(len(model_types))
    bars = plt.bar(x, mean_latencies, yerr=std_latencies, capsize=10, color=colors[:len(model_types)])
    
    # Add labels and formatting
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Inference Time (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, model_types, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_latencies[i] * 1.1,
                 f'{mean_latencies[i]:.4f}s',
                 ha='center', va='bottom', fontsize=10)
    
    # Add speedup annotations
    if "speedup_fp32" in benchmark_data:
        plt.annotate(f"Speedup: {benchmark_data['speedup_fp32']:.2f}x faster",
                    xy=(0.5, 0.9),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center',
                    fontsize=12)
    
    if "speedup_int8" in benchmark_data and "speedup_int8_vs_fp32" in benchmark_data:
        plt.annotate(f"INT8 vs PyTorch: {benchmark_data['speedup_int8']:.2f}x faster\n"
                     f"INT8 vs FP32: {benchmark_data['speedup_int8_vs_fp32']:.2f}x faster",
                    xy=(0.5, 0.8),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center',
                    fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved latency comparison chart to {output_path}")
    
    return plt

def plot_complete_dashboard(benchmark_data, title="Model Performance Dashboard", output_path=None):
    """Create a comprehensive dashboard with multiple panels."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Panel 1: Latency Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract data
    model_types = []
    mean_latencies = []
    std_latencies = []
    
    if "pytorch" in benchmark_data and benchmark_data["pytorch"]:
        model_types.append("PyTorch")
        mean_latencies.append(benchmark_data["pytorch"]["mean"])
        std_latencies.append(benchmark_data["pytorch"]["std"])
    
    if "onnx_fp32" in benchmark_data and benchmark_data["onnx_fp32"]:
        model_types.append("ONNX FP32")
        mean_latencies.append(benchmark_data["onnx_fp32"]["mean"])
        std_latencies.append(benchmark_data["onnx_fp32"]["std"])
    
    if "onnx_int8" in benchmark_data and benchmark_data["onnx_int8"]:
        model_types.append("ONNX INT8")
        mean_latencies.append(benchmark_data["onnx_int8"]["mean"])
        std_latencies.append(benchmark_data["onnx_int8"]["std"])
    
    # Set colors
    colors = ['#4285F4', '#34A853', '#FBBC05']
    
    # Create bar chart
    x = np.arange(len(model_types))
    bars = ax1.bar(x, mean_latencies, yerr=std_latencies, capsize=10, color=colors[:len(model_types)])
    
    # Add labels and formatting
    ax1.set_xlabel('Model Type', fontsize=12)
    ax1.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax1.set_title('Model Latency Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_types, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_latencies[i] * 1.1,
                 f'{mean_latencies[i]:.4f}s',
                 ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Speedup Comparison (Horizontal Bar)
    ax2 = fig.add_subplot(gs[0, 1])
    
    speedup_labels = []
    speedup_values = []
    
    if "speedup_fp32" in benchmark_data:
        speedup_labels.append("ONNX FP32 vs PyTorch")
        speedup_values.append(benchmark_data["speedup_fp32"])
    
    if "speedup_int8" in benchmark_data:
        speedup_labels.append("ONNX INT8 vs PyTorch")
        speedup_values.append(benchmark_data["speedup_int8"])
    
    if "speedup_int8_vs_fp32" in benchmark_data:
        speedup_labels.append("ONNX INT8 vs ONNX FP32")
        speedup_values.append(benchmark_data["speedup_int8_vs_fp32"])
    
    # Create horizontal bar chart for speedups
    y_pos = np.arange(len(speedup_labels))
    speedup_bars = ax2.barh(y_pos, speedup_values, color=['#4285F4', '#34A853', '#FBBC05'][:len(speedup_labels)])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(speedup_labels, fontsize=10)
    ax2.set_xlabel('Speedup Factor (×)', fontsize=12)
    ax2.set_title('Optimization Speedup', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(speedup_bars):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{speedup_values[i]:.2f}×',
                 ha='left', va='center', fontsize=10)
    
    # Panel 3: Individual Run Latencies
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create dataframe for seaborn plot
    runs_data = []
    
    if "pytorch" in benchmark_data and benchmark_data["pytorch"]:
        for i, latency in enumerate(benchmark_data["pytorch"]["latencies"]):
            runs_data.append({"Model": "PyTorch", "Run": i+1, "Latency": latency})
    
    if "onnx_fp32" in benchmark_data and benchmark_data["onnx_fp32"]:
        for i, latency in enumerate(benchmark_data["onnx_fp32"]["latencies"]):
            runs_data.append({"Model": "ONNX FP32", "Run": i+1, "Latency": latency})
    
    if "onnx_int8" in benchmark_data and benchmark_data["onnx_int8"]:
        for i, latency in enumerate(benchmark_data["onnx_int8"]["latencies"]):
            runs_data.append({"Model": "ONNX INT8", "Run": i+1, "Latency": latency})
    
    df = pd.DataFrame(runs_data)
    
    # Create line plot
    sns.lineplot(data=df, x="Run", y="Latency", hue="Model", marker="o", ax=ax3)
    ax3.set_title("Inference Latency by Run", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 4: Additional Information
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create a text box with information
    info_text = f"Model Performance Summary\n"
    info_text += f"=========================\n\n"
    
    if "pytorch" in benchmark_data and benchmark_data["pytorch"]:
        info_text += f"PyTorch Model:\n"
        info_text += f"  Mean Latency: {benchmark_data['pytorch']['mean']:.4f}s\n"
        info_text += f"  Min Latency: {benchmark_data['pytorch']['min']:.4f}s\n"
        info_text += f"  Max Latency: {benchmark_data['pytorch']['max']:.4f}s\n\n"
    
    if "onnx_fp32" in benchmark_data and benchmark_data["onnx_fp32"]:
        info_text += f"ONNX FP32 Model:\n"
        info_text += f"  Mean Latency: {benchmark_data['onnx_fp32']['mean']:.4f}s\n"
        info_text += f"  Min Latency: {benchmark_data['onnx_fp32']['min']:.4f}s\n"
        info_text += f"  Max Latency: {benchmark_data['onnx_fp32']['max']:.4f}s\n\n"
    
    if "onnx_int8" in benchmark_data and benchmark_data["onnx_int8"]:
        info_text += f"ONNX INT8 Model:\n"
        info_text += f"  Mean Latency: {benchmark_data['onnx_int8']['mean']:.4f}s\n"
        info_text += f"  Min Latency: {benchmark_data['onnx_int8']['min']:.4f}s\n"
        info_text += f"  Max Latency: {benchmark_data['onnx_int8']['max']:.4f}s\n\n"
    
    info_text += f"Optimization Benefits:\n"
    if "speedup_fp32" in benchmark_data:
        info_text += f"  ONNX FP32 Speedup: {benchmark_data['speedup_fp32']:.2f}×\n"
    if "speedup_int8" in benchmark_data:
        info_text += f"  ONNX INT8 Speedup: {benchmark_data['speedup_int8']:.2f}×\n"
    if "speedup_int8_vs_fp32" in benchmark_data:
        info_text += f"  INT8 vs FP32 Speedup: {benchmark_data['speedup_int8_vs_fp32']:.2f}×\n"
    
    ax4.text(0, 1, info_text, fontsize=10, va='top', family='monospace')
    
    # Add title to the entire figure
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance dashboard to {output_path}")
    
    return fig

def create_performance_dashboard(benchmark_data, output_dir=None, prefix=None):
    """Create multiple visualization panels and save them."""
    if output_dir is None:
        output_dir = DATA_DIR / "benchmarks"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = prefix or f"benchmark_{timestamp}"
    
    # Save the benchmark data
    data_path = output_dir / f"{prefix}_data.json"
    save_benchmark_data(benchmark_data, data_path)
    
    # Create the latency comparison chart
    latency_path = output_dir / f"{prefix}_latency.png"
    plot_latency_comparison(benchmark_data, output_path=latency_path)
    
    # Create the complete dashboard
    dashboard_path = output_dir / f"{prefix}_dashboard.png"
    plot_complete_dashboard(benchmark_data, output_path=dashboard_path)
    
    return {
        "data_path": str(data_path),
        "latency_path": str(latency_path),
        "dashboard_path": str(dashboard_path)
    }

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # If a JSON file is provided, load it
        json_path = sys.argv[1]
        benchmark_data = load_benchmark_data(json_path)
        prefix = Path(json_path).stem
    else:
        # Create some sample data for testing
        benchmark_data = {
            "pytorch": {
                "latencies": [0.520, 0.510, 0.515, 0.505, 0.525],
                "mean": 0.515,
                "std": 0.008,
                "min": 0.505,
                "max": 0.525
            },
            "onnx_fp32": {
                "latencies": [0.320, 0.315, 0.310, 0.325, 0.318],
                "mean": 0.318,
                "std": 0.006,
                "min": 0.310,
                "max": 0.325
            },
            "onnx_int8": {
                "latencies": [0.150, 0.145, 0.155, 0.148, 0.152],
                "mean": 0.150,
                "std": 0.004,
                "min": 0.145,
                "max": 0.155
            },
            "speedup_fp32": 1.62,
            "speedup_int8": 3.43,
            "speedup_int8_vs_fp32": 2.12
        }
        prefix = "sample"
    
    # Create dashboard
    result = create_performance_dashboard(benchmark_data, prefix=prefix)
    
    print(f"Created performance dashboard:")
    print(f"  Data saved to: {result['data_path']}")
    print(f"  Latency chart saved to: {result['latency_path']}")
    print(f"  Dashboard saved to: {result['dashboard_path']}")
    
    # Display the dashboard
    plt.show()