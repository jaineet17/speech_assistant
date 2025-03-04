"""LLM integration for generating intelligent responses."""

import os
import json
import requests
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMResponseGenerator:
    """Class for generating responses using LLM API."""
    
    def __init__(self, config_path=None):
        """Initialize the LLM response generator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
        else:
            # Default configuration
            self.model = "gpt-3.5-turbo"
            self.temperature = 0.7
            self.max_tokens = 150
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.system_prompt = (
                "You are a helpful voice assistant that provides concise and informative "
                "responses. Keep your answers brief but accurate, with no more than 3 sentences "
                "unless absolutely necessary. Be friendly and conversational."
            )
            
        # Create conversation history
        self.conversation_history = []
        
        # Check if API key is available
        if not self.api_key:
            logger.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    def _load_config(self, config_path):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.model = config.get("model", "gpt-3.5-turbo")
            self.temperature = config.get("temperature", 0.7)
            self.max_tokens = config.get("max_tokens", 150)
            self.base_url = config.get("base_url", "https://api.openai.com/v1/chat/completions")
            self.system_prompt = config.get("system_prompt", "You are a helpful voice assistant.")
            
            logger.info(f"Loaded LLM configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def generate_response(self, user_query, conversation_context=None):
        """Generate a response using the LLM API.
        
        Args:
            user_query: The user's query text
            conversation_context: Optional additional context
            
        Returns:
            Response text from the LLM
        """
        if not self.api_key:
            return "I'm sorry, but I don't have access to my language model right now. Please check if the API key is set correctly."
        
        # Update conversation history
        if conversation_context:
            # If there's specific context for this query, add it
            self.conversation_history.append({"role": "system", "content": conversation_context})
            
        # Add user query to history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Prepare messages for API call
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add up to 3 most recent turns of conversation for context
        if len(self.conversation_history) > 0:
            messages.extend(self.conversation_history[-min(6, len(self.conversation_history)):])
        
        # Make API call
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            
            response_data = response.json()
            
            if response.status_code != 200:
                error_message = response_data.get("error", {}).get("message", "Unknown error")
                logger.error(f"API error: {error_message}")
                return f"I'm sorry, I encountered an error while processing your request."
            
            # Extract response text
            response_text = response_data["choices"][0]["message"]["content"].strip()
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history has been reset")
        
    def save_conversation(self, output_path):
        """Save the conversation history to a file.
        
        Args:
            output_path: Path to save the conversation
            
        Returns:
            Path to the saved file
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the conversation
            with open(output_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            
            logger.info(f"Conversation saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return None

class MockLLMResponseGenerator:
    """Mock LLM response generator for testing without API access."""
    
    def __init__(self):
        """Initialize the mock response generator."""
        self.conversation_history = []
        self.response_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Greetings! What do you need help with?"
            ],
            "weather": [
                "I don't have access to real-time weather data, but I can help with other questions.",
                "I can't check the weather right now, but I'm happy to assist with something else."
            ],
            "time": [
                "I don't have access to the current time in your location.",
                "Unfortunately, I can't tell you the exact time where you are."
            ],
            "thanks": [
                "You're welcome! Let me know if you need anything else.",
                "Happy to help! Is there anything else you need?",
                "My pleasure! Feel free to ask if you have more questions."
            ],
            "goodbye": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye for now! Feel free to come back if you have more questions."
            ],
            "fallback": [
                "I understand your question. Let me try to help with that.",
                "That's an interesting question. Here's what I can tell you.",
                "I'll do my best to address your query."
            ]
        }
    
    def generate_response(self, user_query, conversation_context=None):
        """Generate a mock response based on the user query.
        
        Args:
            user_query: The user's query text
            conversation_context: Optional additional context
            
        Returns:
            Response text
        """
        # Add user query to history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Determine response category
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            category = "greeting"
        elif any(word in query_lower for word in ["weather", "temperature", "forecast", "rain"]):
            category = "weather"
        elif any(word in query_lower for word in ["time", "clock", "hour"]):
            category = "time"
        elif any(word in query_lower for word in ["thanks", "thank you", "appreciate"]):
            category = "thanks"
        elif any(word in query_lower for word in ["bye", "goodbye", "see you", "farewell"]):
            category = "goodbye"
        else:
            category = "fallback"
        
        # Select a response from the appropriate category
        import random
        response_text = random.choice(self.response_templates[category])
        
        # If category is fallback, add a specific response to the query
        if category == "fallback":
            # Add a more specific response
            if "who" in query_lower and "you" in query_lower:
                response_text += " I'm a voice assistant designed to help answer your questions and provide assistance."
            elif "how" in query_lower and "work" in query_lower:
                response_text += " I use speech recognition to understand your questions and text-to-speech to respond."
            elif any(word in query_lower for word in ["advice", "help", "suggest"]):
                response_text += " Without more specific details, I'd recommend breaking down your problem into smaller steps and addressing each one methodically."
            else:
                response_text += " If you could provide more details or rephrase your question, I might be able to help better."
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Simulate API delay
        time.sleep(0.5)
        
        return response_text
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        
    def save_conversation(self, output_path):
        """Save the conversation history to a file."""
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the conversation
            with open(output_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            
            return output_path
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return None

# Factory function to get the appropriate response generator
def get_response_generator(use_mock=False, use_ollama=False, config_path=None):
    """Get an appropriate response generator based on configuration.
    
    Args:
        use_mock: Whether to use the mock generator
        use_ollama: Whether to use Ollama for local LLM inference
        config_path: Path to configuration file
        
    Returns:
        A response generator instance
    """
    if use_mock:
        return MockLLMResponseGenerator()
    elif use_ollama:
        try:
            from .ollama_integration import OllamaLLM
            # Load config if provided
            config = None
            if config_path:
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
                    config = full_config.get("models", {}).get("llm", {})
            return OllamaLLM(config=config)
        except ImportError as e:
            logger.error(f"Failed to import OllamaLLM: {e}")
            logger.warning("Falling back to mock LLM")
            return MockLLMResponseGenerator()
    else:
        return LLMResponseGenerator(config_path)