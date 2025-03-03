"""Inference using ONNX TTS model."""

import numpy as np
import onnxruntime as ort
import json
import soundfile as sf
import importlib
from TTS.tts.configs.vits_config import VitsConfig

class VITS_ONNX:
    """Class for VITS ONNX inference."""
    
    def __init__(self, model_path, config_path):
        """Initialize the VITS ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            config_path: Path to the model configuration
        """
        # Initialize ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        
        # Load config
        self.config = VitsConfig()
        self.config.load_json(config_path)
        
        # Setup text processor
        try:
            tokenizer_module = importlib.import_module("TTS.tts.utils.text.tokenizer")
            processor_class = getattr(tokenizer_module, self.config.text_encoder_params.get("text_processor", ""))
            self.text_processor = processor_class(
                self.config.text_encoder_params.get("phonemizer", None)
            )
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not initialize text processor: {e}")
            self.text_processor = None
        
        # Get input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def process_text(self, text):
        """Process text for model input.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Processed text tokens and lengths
        """
        if self.text_processor:
            tokens = self.text_processor.encode(text)
        else:
            # Simple fallback tokenization
            tokens = [ord(c) for c in text]
            
        token_ids = np.array([tokens], dtype=np.int64)
        text_lengths = np.array([len(tokens)], dtype=np.int64)
        
        return token_ids, text_lengths
    
    def synthesize(self, text, output_path):
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio output
            
        Returns:
            Path to the synthesized audio file
        """
        # Process text
        token_ids, text_lengths = self.process_text(text)
        
        # Default scales for pitch, energy, and duration
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
        
        # Prepare inputs
        inputs = {}
        if "input_text" in self.input_names:
            inputs["input_text"] = token_ids
        if "text_lengths" in self.input_names:
            inputs["text_lengths"] = text_lengths
        if "scales" in self.input_names:
            inputs["scales"] = scales
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Get audio output
        audio = outputs[0].squeeze()
        
        # Save to file
        sf.write(output_path, audio, self.config.audio.sample_rate)
        
        return output_path