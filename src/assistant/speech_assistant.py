"""Speech assistant implementation."""

import os
import time
from pathlib import Path

class SpeechAssistant:
    """Class for integrating STT and TTS into a voice assistant."""
    
    def __init__(self, stt_model, tts_model=None):
        """Initialize with STT and TTS models.
        
        Args:
            stt_model: Speech-to-text model instance
            tts_model: Text-to-speech model instance
        """
        self.stt_model = stt_model
        self.tts_model = tts_model
        
        # Use absolute path for output directory
        project_root = Path(__file__).parent.parent.parent.absolute()
        self.output_dir = project_root / "data" / "test_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_query(self, audio_input_path):
        """Process voice query through the assistant pipeline.
        
        Args:
            audio_input_path: Path to the input audio file
            
        Returns:
            Dictionary with input text, response text, and audio path
        """
        # Step 1: Transcribe audio to text
        print("Transcribing audio...")
        transcription = self.stt_model.transcribe(audio_input_path)
        print(f"Transcription: {transcription}")
        
        # Step 2: Generate response (simple rule-based for now)
        print("Generating response...")
        response = self._generate_response(transcription)
        
        # Step 3: Convert response to speech
        print("Synthesizing speech...")
        timestamp = int(time.time())
        output_path = str(self.output_dir / f"response_{timestamp}.wav")
        
        try:
            self.tts_model.synthesize(response, output_path)
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"TTS did not create the output file: {output_path}")
                
        except Exception as e:
            print(f"Error in TTS: {e}")
            # Create a simple fallback response that explains the error
            error_message = f"I apologize, but I encountered an error generating speech. Here's my text response: {response}"
            raise RuntimeError(f"Failed to synthesize speech: {str(e)}")
        
        return {
            "input_text": transcription,
            "response_text": response,
            "response_audio": output_path
        }
    
    def _generate_response(self, query):
        """Generate a response based on the query text.
        
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
            return "Success in life often comes from persistence, continuous learning, and maintaining good relationships. Focus on setting clear goals, developing good habits, and taking consistent action toward what's important to you. Remember that success is personal - define it on your own terms."
            
        elif "name" in query_lower or "who are you" in query_lower:
            return "My name is Speech Assistant, an AI voice system designed to help answer your questions and provide assistance."
            
        elif "university" in query_lower or "college" in query_lower or "cmu" in query_lower:
            return "Carnegie Mellon University is known for its excellent programs in computer science, engineering, and many other fields. It's located in Pittsburgh, Pennsylvania."
        
        else:
            return "I heard you, but I'm not sure how to respond to that specific query. Could you try asking something else?"