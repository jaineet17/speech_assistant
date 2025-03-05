#!/usr/bin/env python3
"""
Development script to run both the frontend and backend servers.
"""

import os
import subprocess
import sys
import time
import signal
import atexit

# Define commands
BACKEND_CMD = ["python", "run.py", "--api"]
FRONTEND_CMD = ["cd", "frontend", "&&", "npm", "run", "dev"]

processes = []

def cleanup():
    """Kill all processes on exit."""
    for process in processes:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            print(f"Error terminating process: {e}")
            try:
                process.kill()
            except:
                pass

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print("\nShutting down servers...")
    cleanup()
    sys.exit(0)

def run_backend():
    """Run the backend API server."""
    print("Starting backend API server...")
    backend_process = subprocess.Popen(
        BACKEND_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(backend_process)
    return backend_process

def run_frontend():
    """Run the frontend development server."""
    print("Starting frontend development server...")
    # Use shell=True for complex commands with &&
    frontend_process = subprocess.Popen(
        " ".join(FRONTEND_CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        shell=True
    )
    processes.append(frontend_process)
    return frontend_process

def monitor_output(process, prefix):
    """Monitor and print process output with a prefix."""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"{prefix}: {line.rstrip()}")
    
    return_code = process.poll()
    if return_code != 0:
        print(f"{prefix} process exited with code {return_code}")
        cleanup()
        sys.exit(return_code)

def main():
    """Main function to run development servers."""
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend
    backend_process = run_backend()
    
    # Wait a bit for backend to initialize
    time.sleep(2)
    
    # Start frontend
    frontend_process = run_frontend()
    
    # Monitor processes in separate threads
    import threading
    backend_thread = threading.Thread(
        target=monitor_output, 
        args=(backend_process, "BACKEND"),
        daemon=True
    )
    frontend_thread = threading.Thread(
        target=monitor_output, 
        args=(frontend_process, "FRONTEND"),
        daemon=True
    )
    
    backend_thread.start()
    frontend_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(0.5)
            # Check if either process has terminated
            if backend_process.poll() is not None or frontend_process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 