# src/tts/mac_tts.py
"""macOS-specific TTS implementation using the 'say' command."""

import os
import subprocess
import tempfile
import shutil

class MacTTS:
    """TTS implementation using macOS 'say' command."""
    
    def __init__(self, voice="Alex"):
        """Initialize with a specific voice.
        
        Args:
            voice: Name of the voice to use (default: "Alex")
        """
        self.voice = voice
        
        # Check if 'say' command exists
        if shutil.which("say") is None:
            raise RuntimeError("The 'say' command was not found. This TTS implementation requires macOS.")
    
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
        
        # Use temporary file for intermediate AIFF format
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate speech using 'say' command
            subprocess.run(
                ["say", "-v", self.voice, "-o", temp_path, text],
                check=True,
                capture_output=True
            )
            
            # Convert AIFF to WAV using 'afconvert' (comes with macOS)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", 
                 temp_path, output_path],
                check=True,
                capture_output=True
            )
            
            return output_path
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)