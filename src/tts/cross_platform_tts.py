"""Cross-platform TTS implementation using pyttsx3."""

import os
import tempfile
import pyttsx3
import soundfile as sf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossPlatformTTS:
    """TTS implementation using pyttsx3 for cross-platform support."""
    
    def __init__(self, voice=None, rate=170, volume=1.0):
        """Initialize with specific voice and parameters.
        
        Args:
            voice: Name or ID of the voice to use (optional)
            rate: Speech rate (words per minute, default: 170)
            volume: Volume from 0.0 to 1.0 (default: 1.0)
        """
        self.engine = pyttsx3.init()
        
        # Set properties
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice if specified
        if voice:
            voices = self.engine.getProperty('voices')
            for v in voices:
                if voice.lower() in v.name.lower() or voice == v.id:
                    self.engine.setProperty('voice', v.id)
                    logger.info(f"Using voice: {v.name}")
                    break
            else:
                available_voices = [v.name for v in voices]
                logger.warning(f"Voice '{voice}' not found. Available voices: {available_voices}")
                logger.info(f"Using default voice: {voices[0].name}")
    
    def get_available_voices(self):
        """Get a list of available voices.
        
        Returns:
            List of voice names
        """
        voices = self.engine.getProperty('voices')
        return [{'name': v.name, 'id': v.id, 'languages': v.languages} for v in voices]
    
    def synthesize(self, text, output_path):
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio output
            
        Returns:
            Path to the synthesized audio file
        """
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use a temporary file for the raw audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save speech to the temporary file
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Convert to the desired format (16kHz mono WAV)
            data, sample_rate = sf.read(temp_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = data.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling (for a more accurate resampling, consider using librosa)
                target_length = int(len(data) * 16000 / sample_rate)
                data = np.interp(
                    np.linspace(0, len(data), target_length),
                    np.arange(len(data)),
                    data
                )
            
            # Save to the output path
            sf.write(output_path, data, 16000)
            
            return output_path
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path) 