#!/usr/bin/env python
"""Script to run tests on the speech assistant components."""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stt.whisper_direct import WhisperDirect
from src.tts.simple_tts import SimpleTTS
from src.assistant.speech_assistant import SpeechAssistant

def test_whisper():
    """Test Whisper direct model."""
    print("\n===== Testing Whisper Direct Model =====")
    
    try:
        # Initialize model
        print("Loading Whisper model...")
        whisper_model = WhisperDirect("openai/whisper-tiny")
        print("Model loaded successfully!")
        
        # Find a test audio file
        test_dir = Path(__file__).parent.parent / "data" / "audio_samples"
        test_files = list(test_dir.glob("*.wav"))
        
        if not test_files:
            print("No test audio files found. Please add some wav files to data/audio_samples/")
            return False
        
        test_file = test_files[0]
        print(f"Testing transcription with {test_file}...")
        
        # Run transcription
        start_time = time.time()
        transcription = whisper_model.transcribe(str(test_file))
        inference_time = time.time() - start_time
        
        print(f"Transcription: \"{transcription}\"")
        print(f"Inference time: {inference_time:.4f}s")
        
        return True
    
    except Exception as e:
        print(f"Error testing Whisper model: {e}")
        return False

def test_tts():
    """Test TTS model."""
    print("\n===== Testing TTS Model =====")
    
    try:
        # Initialize model
        print("Initializing TTS engine...")
        tts_model = SimpleTTS()
        print("TTS engine initialized successfully!")
        
        # Test synthesis
        test_text = "This is a test of the text to speech system."
        output_path = Path(__file__).parent.parent / "data" / "test_outputs" / "tts_test.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Synthesizing speech for: \"{test_text}\"...")
        
        # Run synthesis
        start_time = time.time()
        tts_model.synthesize(test_text, str(output_path))
        inference_time = time.time() - start_time
        
        print(f"Speech synthesized to {output_path}")
        print(f"Inference time: {inference_time:.4f}s")
        
        return True
    
    except Exception as e:
        print(f"Error testing TTS model: {e}")
        return False

def test_assistant():
    """Test the speech assistant."""
    print("\n===== Testing Speech Assistant =====")
    
    # Check if whisper test succeeded
    whisper_success = test_whisper()
    if not whisper_success:
        print("Cannot test assistant because Whisper test failed.")
        return False
    
    # Check if TTS test succeeded
    tts_success = test_tts()
    if not tts_success:
        print("Cannot test assistant because TTS test failed.")
        return False
    
    try:
        # Initialize models
        print("Initializing models for assistant...")
        whisper_model = WhisperDirect("openai/whisper-tiny")
        tts_model = SimpleTTS()
        
        # Initialize assistant
        assistant = SpeechAssistant(whisper_model, tts_model)
        print("Assistant initialized successfully!")
        
        # Find a test audio file
        test_dir = Path(__file__).parent.parent / "data" / "audio_samples"
        test_files = list(test_dir.glob("*.wav"))
        
        if not test_files:
            print("No test audio files found. Please add some wav files to data/audio_samples/")
            return False
        
        test_file = test_files[0]
        print(f"Testing assistant with {test_file}...")
        
        # Process query
        start_time = time.time()
        result = assistant.process_query(str(test_file))
        total_time = time.time() - start_time
        
        print(f"Input text: \"{result['input_text']}\"")
        print(f"Response text: \"{result['response_text']}\"")
        print(f"Response audio: {result['response_audio']}")
        print(f"Total processing time: {total_time:.4f}s")
        
        return True
    
    except Exception as e:
        print(f"Error testing assistant: {e}")
        return False

def create_test_audio():
    """Create a test audio file if none exists."""
    print("\n===== Creating Test Audio File =====")
    
    test_dir = Path(__file__).parent.parent / "data" / "audio_samples"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = list(test_dir.glob("*.wav"))
    if test_files:
        print(f"Test audio files already exist: {', '.join(str(f.name) for f in test_files)}")
        return True
    
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        print("No test audio files found. Let's create a sample file.")
        
        # Create a simple tone
        sample_rate = 16000
        duration = 3  # seconds
        frequency = 440  # Hz (A4 note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add fade in/out
        fade_duration = 0.1  # seconds
        fade_samples = int(fade_duration * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
        # Save the tone
        output_path = test_dir / "sample_tone.wav"
        sf.write(str(output_path), tone, sample_rate)
        
        print(f"Sample audio created and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating test audio: {e}")
        print("Please manually add a WAV file to data/audio_samples/ for testing.")
        return False

if __name__ == "__main__":
    print("Running tests for speech assistant components...")
    
    # Create test audio if needed
    create_test_audio()
    
    # Run tests
    whisper_success = test_whisper()
    tts_success = test_tts()
    assistant_success = test_assistant()
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Whisper Direct: {'✅ Passed' if whisper_success else '❌ Failed'}")
    print(f"TTS: {'✅ Passed' if tts_success else '❌ Failed'}")
    print(f"Speech Assistant: {'✅ Passed' if assistant_success else '❌ Failed'}")
    
    if whisper_success and tts_success and assistant_success:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the API: python api/app.py")
        print("2. Run the UI: streamlit run ui/streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")