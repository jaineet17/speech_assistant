"""Unit tests for the WhisperDirect component."""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import torch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stt.whisper_direct import WhisperDirect


class TestWhisperDirect(unittest.TestCase):
    """Test cases for the WhisperDirect class."""

    @patch("src.stt.whisper_direct.WhisperProcessor")
    @patch("src.stt.whisper_direct.WhisperForConditionalGeneration")
    @patch("src.stt.whisper_direct.torch.device")
    @patch("src.stt.whisper_direct.torch.cuda.is_available")
    def test_init_loads_model(self, mock_cuda_available, mock_device, 
                             mock_model_class, mock_processor_class):
        """Test that the constructor loads the model correctly."""
        # Mock CUDA not available
        mock_cuda_available.return_value = False
        mock_device.return_value = "cpu"
        
        # Mock processor and model
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create WhisperDirect instance
        whisper = WhisperDirect(model_id="openai/whisper-tiny")
        
        # Check that the processor and model were loaded
        mock_processor_class.from_pretrained.assert_called_once_with("openai/whisper-tiny")
        mock_model_class.from_pretrained.assert_called_once_with("openai/whisper-tiny")
        
        # Check that the model was moved to the correct device
        mock_model.to.assert_called_once()
        
        # Check that the model was set to eval mode
        mock_model.eval.assert_called_once()
    
    @patch("src.stt.whisper_direct.WhisperProcessor")
    @patch("src.stt.whisper_direct.WhisperForConditionalGeneration")
    @patch("src.stt.whisper_direct.torch.cuda.is_available")
    @patch("src.stt.whisper_direct.librosa.load")
    def test_transcribe_short_audio(self, mock_librosa_load, mock_cuda_available,
                                   mock_model_class, mock_processor_class):
        """Test transcription of short audio files."""
        # Mock CUDA not available
        mock_cuda_available.return_value = False
        
        # Mock audio loading
        mock_audio = torch.zeros(16000 * 5)  # 5 seconds of silence
        mock_librosa_load.return_value = (mock_audio, 16000)
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_processor.return_value = MagicMock(input_features=torch.zeros((1, 80, 3000)))
        mock_processor.batch_decode.return_value = ["This is a test transcription"]
        
        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        
        # Create WhisperDirect instance
        whisper = WhisperDirect(model_id="openai/whisper-tiny")
        whisper.model = mock_model
        whisper.processor = mock_processor
        
        # Transcribe audio
        result = whisper.transcribe("test.wav")
        
        # Check that the result is correct
        self.assertEqual(result, "This is a test transcription")
        
        # Check that librosa.load was called
        mock_librosa_load.assert_called_once_with("test.wav", sr=16000)


if __name__ == "__main__":
    unittest.main() 