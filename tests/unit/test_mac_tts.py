"""Unit tests for the MacTTS component."""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tts.mac_tts import MacTTS


class TestMacTTS(unittest.TestCase):
    """Test cases for the MacTTS class."""

    @patch("src.tts.mac_tts.shutil.which")
    def test_init_checks_say_command(self, mock_which):
        """Test that the constructor checks for the 'say' command."""
        # Mock 'say' command not found
        mock_which.return_value = None
        
        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            MacTTS()
        
        # Verify 'which' was called with 'say'
        mock_which.assert_called_once_with("say")
    
    @patch("src.tts.mac_tts.shutil.which")
    @patch("src.tts.mac_tts.subprocess.run")
    @patch("src.tts.mac_tts.os.makedirs")
    def test_synthesize_creates_output(self, mock_makedirs, mock_run, mock_which):
        """Test that synthesize creates the output file."""
        # Mock 'say' command found
        mock_which.return_value = "/usr/bin/say"
        
        # Mock successful subprocess runs
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create a temporary output path
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")
            
            # Create the TTS instance and synthesize
            tts = MacTTS(voice="Samantha")
            result = tts.synthesize("Hello, world!", output_path)
            
            # Check that makedirs was called
            mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)
            
            # Check that subprocess.run was called twice (say and afconvert)
            self.assertEqual(mock_run.call_count, 2)
            
            # Check that the result is the output path
            self.assertEqual(result, output_path)


if __name__ == "__main__":
    unittest.main() 