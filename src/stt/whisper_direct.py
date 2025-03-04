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
        """Transcribe audio to text with chunking for long files.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription (default: "en")
                
        Returns:
            Transcribed text
        """
        # Load audio
        audio_array, _ = librosa.load(audio_path, sr=16000)
        
        # For very large files, process in chunks
        chunk_size = 16000 * 30  # 30 seconds per chunk
        transcription_parts = []
        
        # Process audio in chunks if it's large
        if len(audio_array) > chunk_size * 2:  # If longer than 1 minute
            print(f"Processing large audio file ({len(audio_array)/16000:.2f} seconds) in chunks")
            
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                
                # Process chunk
                features = self.processor(
                    chunk, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate tokens for this chunk
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        features,
                        language=language,
                        task="transcribe"
                    )
                
                # Decode output for this chunk
                chunk_text = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                transcription_parts.append(chunk_text)
                
            # Join all chunks
            transcription = " ".join(transcription_parts)
        else:
            # Process normally for shorter audio
            features = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    features,
                    language=language,
                    task="transcribe"
                )
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
        
        return transcription