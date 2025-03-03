"""Benchmarking utilities for measuring performance."""

import time
import numpy as np
import psutil
import os
import matplotlib.pyplot as plt
from pathlib import Path

class PerformanceTracker:
    """Class for tracking performance metrics."""
    
    def __init__(self, name="Performance"):
        """Initialize with a name.
        
        Args:
            name: Name for this performance tracking session
        """
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.metrics = {
            "latency": [],
            "memory": [],
            "cpu": []
        }
    
    def start(self):
        """Start tracking performance."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        return self
    
    def stop(self):
        """Stop tracking and record metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = self.process.cpu_percent()
        
        latency = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        
        self.metrics["latency"].append(latency)
        self.metrics["memory"].append(memory_used)
        self.metrics["cpu"].append(cpu_percent)
        
        return latency, memory_used, cpu_percent
    
    def track(self, func, *args, **kwargs):
        """Track performance of a function call.
        
        Args:
            func: Function to track
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        self.start()
        result = func(*args, **kwargs)
        latency, memory, cpu = self.stop()
        
        return result, {"latency": latency, "memory": memory, "cpu": cpu}
    
    def summary(self):
        """Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "name": self.name,
            "latency": {
                "mean": np.mean(self.metrics["latency"]),
                "std": np.std(self.metrics["latency"]),
                "min": np.min(self.metrics["latency"]),
                "max": np.max(self.metrics["latency"])
            },
            "memory": {
                "mean": np.mean(self.metrics["memory"]),
                "std": np.std(self.metrics["memory"]),
                "min": np.min(self.metrics["memory"]),
                "max": np.max(self.metrics["memory"])
            },
            "cpu": {
                "mean": np.mean(self.metrics["cpu"]),
                "std": np.std(self.metrics["cpu"]),
                "min": np.min(self.metrics["cpu"]),
                "max": np.max(self.metrics["cpu"])
            }
        }
    
    def plot(self, output_path=None):
        """Plot performance metrics.
        
        Args:
            output_path: Path to save the plot
        """
        summary = self.summary()
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
        # Latency plot
        axs[0].plot(self.metrics["latency"])
        axs[0].set_title(f"Latency (mean: {summary['latency']['mean']:.4f}s)")
        axs[0].set_ylabel("Seconds")
        axs[0].grid(True)
        
        # Memory plot
        axs[1].plot(self.metrics["memory"], color="green")
        axs[1].set_title(f"Memory Usage (mean: {summary['memory']['mean']:.2f} MB)")
        axs[1].set_ylabel("MB")
        axs[1].grid(True)
        
        # CPU plot
        axs[2].plot(self.metrics["cpu"], color="red")
        axs[2].set_title(f"CPU Usage (mean: {summary['cpu']['mean']:.2f}%)")
        axs[2].set_ylabel("Percent")
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {output_path}")
        
        plt.show()


def compare_models(models, test_func, test_data, num_runs=5, labels=None, output_path=None):
    """Compare performance of multiple models.
    
    Args:
        models: List of model instances to compare
        test_func: Function to test each model (takes model and test_data)
        test_data: Data to test with
        num_runs: Number of runs for each model
        labels: Labels for each model
        output_path: Path to save the plot
        
    Returns:
        Results summary
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(models))]
    
    results = []
    
    for i, (model, label) in enumerate(zip(models, labels)):
        print(f"Testing {label}...")
        
        tracker = PerformanceTracker(label)
        
        for j in range(num_runs):
            tracker.start()
            test_func(model, test_data)
            tracker.stop()
            
            print(f"  Run {j+1}/{num_runs}: {tracker.metrics['latency'][-1]:.4f}s")
        
        results.append(tracker.summary())
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    model_names = [r["name"] for r in results]
    latencies = [r["latency"]["mean"] for r in results]
    latency_stds = [r["latency"]["std"] for r in results]
    
    # Create bar chart
    plt.bar(range(len(model_names)), latencies, yerr=latency_stds, capsize=10)
    plt.xticks(range(len(model_names)), model_names)
    
    # Add text on bars
    for i, v in enumerate(latencies):
        plt.text(i, v + max(latency_stds) * 1.2, f"{v:.3f}s", ha='center')
    
    # Calculate speedup if there are at least 2 models
    if len(latencies) >= 2:
        speedup = latencies[0] / latencies[1]
        plt.text(0.5, max(latencies) * 1.3, f"Speedup: {speedup:.2f}x", 
                 ha='center', fontsize=12, fontweight='bold')
    
    # Add labels and title
    plt.ylabel('Inference Time (seconds)')
    plt.title('Model Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return results