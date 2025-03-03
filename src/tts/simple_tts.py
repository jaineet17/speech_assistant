"""Simple TTS implementation using pyttsx3."""

import pyttsx3
import os

class SimpleTTS:
    """Simple TTS class using pyttsx3."""
    
    def __init__(self):
        """Initialize the TTS engine."""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speaking rate
        
        # Try to use a good voice if available
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a female voice
            female_voices = [v for v in voices if hasattr(v, 'gender') and v.gender == 'female']
            if female_voices:
                self.engine.setProperty('voice', female_voices[0].id)
            else:
                self.engine.setProperty('voice', voices[0].id)
    
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
        
        # Save speech to file
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        
        return output_path