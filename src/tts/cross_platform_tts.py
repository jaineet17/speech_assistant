"""
Cross-platform text-to-speech module using a more reliable approach.
"""

import os
import logging
import tempfile
import wave
import struct
import math
import subprocess
import platform
import threading
import concurrent.futures
from typing import Optional, Dict, Tuple
import sys
import time
import uuid
import io
import shutil
from collections import OrderedDict
from functools import lru_cache

# Configure logging
logger = logging.getLogger("enhanced_speech_assistant.tts")

# Cache for TTS outputs to avoid regenerating the same text
# Using OrderedDict to maintain insertion order for LRU cache behavior
_tts_cache = OrderedDict()
_tts_cache_lock = threading.Lock()
_MAX_CACHE_SIZE = 100  # Increased cache size for better performance

# Thread pool for parallel processing
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def text_to_speech(text: str, output_path: str = None) -> str:
    """
    Convert text to speech using platform-specific TTS
    Uses caching to improve performance for repeated phrases
    
    Args:
        text (str): Text to convert to speech
        output_path (str, optional): Path to save the audio file. If None, a temporary file is created.
    
    Returns:
        str: Path to the saved audio file
    """
    if not text:
        logger.warning("Empty text provided for TTS. Creating silent audio.")
        return create_silent_wav_file(0.5, output_path)
    
    logger.info(f"Converting text to speech: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Create a cache key based on the text
    cache_key = text
    
    # Check if we have this text in our cache
    with _tts_cache_lock:
        if cache_key in _tts_cache:
            cached_path = _tts_cache[cache_key]
            # Move this item to the end of the OrderedDict to mark it as recently used
            _tts_cache.move_to_end(cache_key)
            if os.path.exists(cached_path):
                logger.info(f"Using cached TTS audio for: {text[:30]}...")
                
                # If output_path is specified, copy the cached file to the requested location
                if output_path and output_path != cached_path:
                    try:
                        # Use shutil for faster file copying
                        shutil.copy2(cached_path, output_path)
                        return output_path
                    except Exception as e:
                        logger.warning(f"Failed to copy cached file: {e}")
                
                return cached_path
    
    # If no output path is specified, create a temporary one
    if not output_path:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"tts_output_{uuid.uuid4()}.wav")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    system = platform.system().lower()
    success = False
    
    try:
        start_time = time.time()
        
        # Optimize for each platform
        if system == 'darwin':  # macOS
            success = _tts_macos(text, output_path)
        elif system == 'windows':
            success = _tts_windows(text, output_path)
        elif system == 'linux':
            success = _tts_linux(text, output_path)
        else:
            logger.warning(f"Unsupported platform: {system}")
            
        duration = time.time() - start_time
        logger.info(f"TTS processing took {duration:.3f} seconds")
        
    except Exception as e:
        logger.error(f"Error during TTS: {e}")
        success = False
    
    # Verify the file was created successfully
    if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"Successfully saved speech to {output_path}")
        
        # Cache the result
        with _tts_cache_lock:
            _tts_cache[cache_key] = output_path
            # If cache is too large, remove the oldest items (beginning of OrderedDict)
            while len(_tts_cache) > _MAX_CACHE_SIZE:
                _, _ = _tts_cache.popitem(last=False)
                
        return output_path
    else:
        logger.warning(f"Failed to save speech to {output_path}, creating silent WAV instead")
        return create_silent_wav_file(len(text) / 15.0, output_path)  # Rough estimate: 15 chars per second

def _tts_macos(text: str, output_path: str) -> bool:
    """Optimized TTS for macOS"""
    try:
        # Use macOS say command directly to generate WAV file
        # Setting a faster speech rate and higher quality
        cmd = ['say', '-o', output_path, '--file-format=WAVE', '--data-format=LEI16@44100', 
               '--rate=180', text]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("macOS TTS process timed out")
        return False
    except Exception as e:
        logger.error(f"Error during macOS TTS: {e}")
        return False

def _tts_windows(text: str, output_path: str) -> bool:
    """Optimized TTS for Windows"""
    try:
        # Create a temporary text file for the content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_text:
            temp_text.write(text)
            temp_text_path = temp_text.name
        
        # PowerShell command for TTS with faster rate and better quality
        ps_script = f"""
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.Rate = 2  # Range: -10 to 10, default is 0
        $synth.Volume = 100
        $synth.SetOutputToWaveFile('{output_path.replace("'", "''")}')
        $synth.Speak([System.IO.File]::ReadAllText('{temp_text_path.replace("'", "''")}'))
        $synth.Dispose()
        """
        
        cmd = ['powershell', '-Command', ps_script]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        success = process.returncode == 0 and os.path.exists(output_path)
        
        # Clean up the temporary text file
        try:
            os.unlink(temp_text_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary text file: {e}")
            
        return success
    except subprocess.TimeoutExpired:
        logger.error("Windows TTS process timed out")
        return False
    except Exception as e:
        logger.error(f"Error during Windows TTS: {e}")
        return False

def _tts_linux(text: str, output_path: str) -> bool:
    """Optimized TTS for Linux"""
    try:
        # Try using espeak-ng first (better quality)
        cmd = ['espeak-ng', '-w', output_path, '-s', '160', '-v', 'en-us', text]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if process.returncode == 0:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fall back to espeak
            cmd = ['espeak', '-w', output_path, '-s', '160', text]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            return process.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Linux TTS process timed out")
        return False
    except Exception as e:
        logger.error(f"Error during Linux TTS: {e}")
        return False

def create_silent_wav_file(duration_seconds, output_path=None):
    """Create a WAV file with silence of specified duration"""
    try:
        # If output_path not specified, create a temp file
        if not output_path:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"silent_{uuid.uuid4()}.wav")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # WAV parameters - higher quality
        sample_rate = 44100
        num_channels = 1
        sample_width = 2  # 16-bit
        
        # Calculate number of frames
        num_frames = int(duration_seconds * sample_rate)
        
        # Create silent audio data (all zeros)
        audio_data = b'\x00' * (num_frames * sample_width * num_channels)
        
        # Create WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        
        logger.info(f"Created silent WAV file at {output_path} ({duration_seconds:.1f}s)")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating silent WAV file: {e}")
        return create_dummy_wav_file(output_path)

def create_dummy_wav_file(output_path=None):
    """
    Create a dummy WAV file with 1 second of silence.
    
    Args:
        output_path: Path to save the WAV file. If None, a temporary file is created.
        
    Returns:
        Path to the created WAV file.
    """
    if output_path is None:
        # Create a temporary file
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    try:
        # Create a simple WAV header for 1 second of silence
        with open(output_path, 'wb') as f:
            f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x10\x00data\x00\x00\x00\x00')
        
        logger.info(f"Created dummy WAV file at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating dummy WAV file: {e}")
        return None

def preload_tts_engine():
    """
    Preload the TTS engine to reduce first-time latency.
    This function runs a small TTS operation to initialize the engine.
    """
    logger.info("Preloading TTS engine...")
    
    def _preload():
        try:
            # Create a temporary directory for preloading
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate a short test file
                test_file = os.path.join(temp_dir, "preload_test.wav")
                
                # Use a very short text to minimize initialization time
                text_to_speech("Hello", test_file)
                
                # Check if the platform-specific engine needs additional initialization
                system = platform.system().lower()
                
                if system == "darwin":
                    # For macOS, initialize the 'say' command
                    subprocess.run(["say", "-v", "?"], capture_output=True)
                elif system == "windows":
                    # For Windows, initialize the speech synthesizer
                    try:
                        import win32com.client
                        speaker = win32com.client.Dispatch("SAPI.SpVoice")
                        voices = speaker.GetVoices()
                        if voices.Count > 0:
                            speaker.Voice = voices.Item(0)
                        speaker.Speak("", 3)  # SVSFlagsAsync = 3 (don't wait for completion)
                    except Exception as e:
                        logger.debug(f"Windows speech synthesizer preload error: {e}")
                elif system == "linux":
                    # For Linux, check if espeak is available
                    try:
                        subprocess.run(["espeak", "--version"], capture_output=True)
                    except Exception:
                        logger.debug("espeak not available for preloading on Linux")
                
                logger.info("TTS engine preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading TTS engine: {e}")
    
    # Run preloading in a separate thread to avoid blocking
    threading.Thread(target=_preload, daemon=True).start()
    return True

# Call preload on module import
preload_tts_engine() 