"""Direct implementation of Whisper without ONNX."""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os

class WhisperDirect:
    """Class for direct Whisper inference."""
    
    def __init__(self, model_id="openai/whisper-tiny"):
        """Initialize the Whisper model.
        
        Args:
            model_id: HuggingFace model ID or local path
        """
        print(f"Loading Whisper model: {model_id}")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.eval()
        
        # Try to use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Whisper model loaded on {self.device}")
    
    def transcribe(self, audio_path, language="en"):
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        # Load and preprocess audio
        audio_array, _ = librosa.load(audio_path, sr=16000)
        input_features = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        
        # Decode output
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription