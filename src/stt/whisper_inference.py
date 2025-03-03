"""Inference using ONNX Whisper model."""

import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperProcessor

class WhisperONNX:
    """Class for Whisper ONNX inference."""
    
    def __init__(self, model_path, processor_path_or_name="openai/whisper-tiny"):
        """Initialize the Whisper ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            processor_path_or_name: Path or name of the Whisper processor
        """
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.processor = WhisperProcessor.from_pretrained(processor_path_or_name)
        
        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
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
            return_tensors="np"
        ).input_features
        
        # Run inference
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: input_features.astype(np.float32)}
        )
        
        # Decode output
        predicted_ids = outputs[0]
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription