#!/usr/bin/env python3
"""
Cleanup script to prepare the project for GitHub.
Removes temporary files, logs, and other artifacts.
"""

import os
import shutil
import glob
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Clean up temporary files and prepare for GitHub")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    return parser.parse_args()

def cleanup_project(dry_run=False):
    """Clean up temporary files and directories."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Directories to clean (empty but keep the directory)
    dirs_to_clean = [
        "temp",
        "responses",
        "uploads"
    ]
    
    # Directories to remove completely
    dirs_to_remove = [
        "**/__pycache__",
        "**/.pytest_cache",
        "**/.coverage",
        "**/.ipynb_checkpoints",
        "**/node_modules",
        "**/.DS_Store"
    ]
    
    # File patterns to remove
    files_to_remove = [
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/*.so",
        "**/*.log",
        "**/*.wav",
        "**/*.mp3",
        "**/*.webm",
        "**/.DS_Store",
        "**/.env"
    ]
    
    # Clean directories (keep but empty)
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            if dry_run:
                print(f"Would clean directory: {dir_path}")
            else:
                print(f"Cleaning directory: {dir_path}")
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                # Create a .gitkeep file to ensure the directory is tracked
                with open(os.path.join(dir_path, '.gitkeep'), 'w') as f:
                    pass
        else:
            if dry_run:
                print(f"Would create directory: {dir_path}")
            else:
                print(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
                # Create a .gitkeep file to ensure the directory is tracked
                with open(os.path.join(dir_path, '.gitkeep'), 'w') as f:
                    pass
    
    # Remove directories
    for pattern in dirs_to_remove:
        for dir_path in glob.glob(os.path.join(project_root, pattern), recursive=True):
            if os.path.exists(dir_path):
                if dry_run:
                    print(f"Would remove directory: {dir_path}")
                else:
                    print(f"Removing directory: {dir_path}")
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
    
    # Remove files
    for pattern in files_to_remove:
        for file_path in glob.glob(os.path.join(project_root, pattern), recursive=True):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                if dry_run:
                    print(f"Would remove file: {file_path}")
                else:
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)
    
    # Remove test_audio.wav in the root directory
    test_audio_path = os.path.join(project_root, "test_audio.wav")
    if os.path.exists(test_audio_path):
        if dry_run:
            print(f"Would remove file: {test_audio_path}")
        else:
            print(f"Removing file: {test_audio_path}")
            os.remove(test_audio_path)
    
    print("Cleanup complete!")

if __name__ == "__main__":
    args = parse_args()
    cleanup_project(dry_run=args.dry_run) 