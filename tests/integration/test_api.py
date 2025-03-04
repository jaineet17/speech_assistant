"""Integration tests for the API."""

import unittest
import sys
import os
from pathlib import Path
import json
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the Flask app
from api.enhanced_app import app


class TestAPI(unittest.TestCase):
    """Test cases for the API."""

    def setUp(self):
        """Set up the test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
    
    @patch('api.enhanced_app.whisper_model')
    @patch('api.enhanced_app.tts_model')
    @patch('api.enhanced_app.response_generator')
    def test_process_audio_endpoint(self, mock_response_generator, mock_tts, mock_whisper):
        """Test the process_audio endpoint."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
            # Generate a simple sine wave
            sample_rate = 16000
            duration = 1  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Mock the transcription
            mock_whisper.transcribe.return_value = "Hello, world!"
            
            # Mock the response generation
            mock_response_generator.generate_response.return_value = "Hi there!"
            
            # Mock the TTS
            output_path = os.path.join(os.path.dirname(temp_audio.name), "response.wav")
            mock_tts.synthesize.return_value = output_path
            
            # Send the request
            with open(temp_audio.name, 'rb') as f:
                response = self.app.post(
                    '/process_audio',
                    data={'audio': (f, 'test.wav')}
                )
            
            # Check the response
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['transcription'], "Hello, world!")
            self.assertEqual(data['response'], "Hi there!")
            self.assertIn('audio_url', data)


if __name__ == "__main__":
    unittest.main() 