#!/usr/bin/env python3
"""Script to run the speech assistant application."""

import os
import sys
import subprocess
import time
import argparse
import signal
import atexit

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the speech assistant application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the UI server")
    parser.add_argument("--api-port", type=int, default=5050, help="API server port")
    parser.add_argument("--ui-port", type=int, default=8501, help="UI server port")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()

def run_api(port, debug):
    """Run the API server."""
    env = os.environ.copy()
    env["API_PORT"] = str(port)
    env["API_DEBUG"] = "true" if debug else "false"
    
    print(f"Starting API server on port {port}...")
    return subprocess.Popen(
        [sys.executable, "api/enhanced_app.py"],
        env=env
    )

def run_ui(port):
    """Run the UI server."""
    env = os.environ.copy()
    
    print(f"Starting UI server on port {port}...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/enhanced_streamlit_app.py", "--server.port", str(port)],
        env=env
    )

def main():
    """Run the application."""
    args = parse_args()
    
    processes = []
    
    # Register cleanup function
    def cleanup():
        for process in processes:
            if process.poll() is None:
                process.terminate()
    
    atexit.register(cleanup)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nShutting down...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start API server
        if not args.ui_only:
            api_process = run_api(args.api_port, args.debug)
            processes.append(api_process)
            
            # Wait for API server to start
            print("Waiting for API server to start...")
            time.sleep(2)
        
        # Start UI server
        if not args.api_only:
            ui_process = run_ui(args.ui_port)
            processes.append(ui_process)
        
        # Wait for processes to finish
        for process in processes:
            process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 