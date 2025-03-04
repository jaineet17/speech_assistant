"""Enhanced speech assistant implementation with LLM integration."""

import os
import time
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSpeechAssistant:
    """Enhanced class for integrating STT, LLM, and TTS into a voice assistant."""
    
    def __init__(self, stt_model, tts_model, response_generator=None, config=None):
        """Initialize with STT, TTS, and LLM models.
        
        Args:
            stt_model: Speech-to-text model instance
            tts_model: Text-to-speech model instance
            response_generator: LLM response generator (optional)
            config: Configuration dictionary (optional)
        """
        self.stt_model = stt_model
        self.tts_model = tts_model
        
        # Get project root directory
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        self.output_dir = self.project_root / "data" / "test_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize response generator if not provided
        if response_generator is None:
            try:
                from .llm_response_generator import get_response_generator
                self.response_generator = get_response_generator(use_mock=True)
                logger.info("Using mock LLM response generator")
            except ImportError as e:
                logger.warning(f"LLM response generator not available, using fallback: {e}")
                self.response_generator = None
        else:
            self.response_generator = response_generator
            
        # Load configuration if provided
        self.config = config or {}
        
        # Session metadata
        self.session_id = int(time.time())
        self.conversation_history = []
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Speech assistant initialized with session ID: {self.session_id}")
    
    def process_query(self, audio_input_path):
        """Process voice query through the assistant pipeline.
        
        Args:
            audio_input_path: Path to the input audio file
            
        Returns:
            Dictionary with input text, response text, and audio path
        """
        start_time = time.time()
        
        # Step 1: Transcribe audio to text
        logger.info("Transcribing audio...")
        try:
            transcription = self.stt_model.transcribe(audio_input_path)
            logger.info(f"Transcription: {transcription}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            transcription = "Could not transcribe audio."
            
        # Track timings
        transcription_time = time.time() - start_time
        
        # Step 2: Generate response using LLM or fallback
        logger.info("Generating response...")
        response_start = time.time()
        
        if self.response_generator and transcription != "Could not transcribe audio.":
            try:
                response = self.response_generator.generate_response(transcription)
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                response = self._fallback_response(transcription)
        else:
            response = self._fallback_response(transcription)
            
        # Track timings
        response_time = time.time() - response_start
        
        # Step 3: Convert response to speech
        logger.info("Synthesizing speech...")
        synthesis_start = time.time()
        
        timestamp = int(time.time())
        output_path = str(self.session_dir / f"response_{timestamp}.wav")
        
        try:
            self.tts_model.synthesize(response, output_path)
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"TTS did not create the output file: {output_path}")
                
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            error_message = "I apologize, but I encountered an error generating speech."
            # Try to create a simple error response
            try:
                error_output_path = str(self.session_dir / f"error_{timestamp}.wav")
                self.tts_model.synthesize(error_message, error_output_path)
                output_path = error_output_path
            except Exception as e:
                raise RuntimeError(f"Failed to synthesize speech: {str(e)}")
                
        # Track timings
        synthesis_time = time.time() - synthesis_start
        total_time = time.time() - start_time
        
        # Store interaction in conversation history
        interaction = {
            "timestamp": timestamp,
            "input_audio": audio_input_path,
            "input_text": transcription,
            "response_text": response,
            "response_audio": output_path,
            "timings": {
                "transcription": transcription_time,
                "response_generation": response_time,
                "speech_synthesis": synthesis_time,
                "total": total_time
            }
        }
        
        self.conversation_history.append(interaction)
        
        # Save session metadata
        self._save_session_metadata()
        
        return {
            "input_text": transcription,
            "response_text": response,
            "response_audio": output_path,
            "timings": {
                "transcription": transcription_time,
                "response_generation": response_time,
                "speech_synthesis": synthesis_time,
                "total": total_time
            }
        }
    
    def _fallback_response(self, query):
        """Generate a fallback response based on the query text.
        
        Args:
            query: Transcribed query text
            
        Returns:
            Response text
        """
        # Basic rule-based responses
        query_lower = query.lower()
        
        if "hello" in query_lower or "hi" in query_lower:
            return "Hello! How can I help you today?"
        
        elif "how are you" in query_lower:
            return "I'm functioning well, thank you for asking. How can I assist you?"
        
        elif any(word in query_lower for word in ["time", "what time"]):
            current_time = time.strftime("%H:%M")
            return f"The current time is {current_time}."
        
        elif "weather" in query_lower:
            return "I'm sorry, I don't have access to weather information right now."
        
        elif any(word in query_lower for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help with?"
        
        elif "bye" in query_lower or "goodbye" in query_lower:
            return "Goodbye! Have a great day."
        
        elif "advice" in query_lower or "success" in query_lower or "life" in query_lower:
            return "Success in life often comes from persistence, continuous learning, and maintaining good relationships. Focus on setting clear goals, developing good habits, and taking consistent action toward what's important to you."
            
        elif "name" in query_lower or "who are you" in query_lower:
            return "My name is Speech Assistant, an AI voice system designed to help answer your questions and provide assistance."
        
        else:
            return "I heard you, but I'm not sure how to respond to that specific query. Could you try asking something else?"
    
    def _save_session_metadata(self):
        """Save session metadata to a file."""
        metadata_path = self.session_dir / "session_metadata.json"
        
        metadata = {
            "session_id": self.session_id,
            "timestamp": int(time.time()),
            "interactions": len(self.conversation_history),
            "conversation_history": self.conversation_history
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.debug(f"Session metadata saved to {metadata_path}")
    
    def reset_session(self):
        """Reset the current session."""
        # Save the old session
        self._save_session_metadata()
        
        # Create a new session
        self.session_id = int(time.time())
        self.conversation_history = []
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize response generator if not provided
        if self.response_generator is None:
            try:
                from .llm_response_generator import get_response_generator
                # Try to use Ollama first, fall back to mock if needed
                self.response_generator = get_response_generator(use_mock=False, use_ollama=True)
                logger.info("Using Ollama for LLM response generation")
            except ImportError as e:
                logger.warning(f"LLM response generator not available, using fallback: {e}")
                self.response_generator = None
        else:
            self.response_generator = self.response_generator