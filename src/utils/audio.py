"""Audio recording and processing utilities."""

import numpy as np
import soundfile as sf
import sounddevice as sd
import time
from pathlib import Path

class AudioRecorder:
    """Class for audio recording and processing."""
    
    def __init__(self, sample_rate=16000, channels=1, output_dir="data/test_outputs"):
        """Initialize with audio parameters.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            output_dir: Directory to save recordings
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def record(self, duration=5, filename=None, show_progress=True):
        """Record audio from the microphone.
        
        Args:
            duration: Recording duration in seconds
            filename: Filename to save recording (auto-generated if None)
            show_progress: Whether to print progress messages
            
        Returns:
            Path to the saved audio file
        """
        if show_progress:
            print(f"Recording for {duration} seconds...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        
        # Show progress indicator
        if show_progress:
            for i in range(10):
                time.sleep(duration / 10)
                print(".", end="", flush=True)
            print()
        else:
            sd.wait()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time())
            filename = f"recording_{timestamp}.wav"
        
        # Ensure the file has .wav extension
        if not filename.endswith(".wav"):
            filename += ".wav"
        
        # Save to file
        output_path = self.output_dir / filename
        sf.write(str(output_path), audio_data, self.sample_rate)
        
        if show_progress:
            print(f"Recording saved to {output_path}")
        
        return str(output_path)
    
    def play(self, file_path):
        """Play an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        sd.wait()