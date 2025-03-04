"""Ollama LLM integration for generating intelligent responses."""

import os
import json
import requests
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLLM:
    """Class for generating responses using Ollama's local LLM."""
    
    def __init__(self, model_name="llama3", config=None):
        """Initialize the Ollama LLM generator.
        
        Args:
            model_name: Name of the Ollama model to use (default: "llama3")
            config: Additional configuration
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", model_name)
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 150)
        self.base_url = self.config.get("base_url", "http://localhost:11434/api")
        
        # System prompt for assistant behavior
        self.system_prompt = self.config.get("system_prompt", 
            "You are a helpful voice assistant that provides concise and informative "
            "responses. Keep your answers brief but accurate, with no more than 3 sentences "
            "unless absolutely necessary. Be friendly and conversational."
        )
        
        # Create conversation history
        self.conversation_history = []
        
        # Check if Ollama is available
        self.check_availability()
    
    def check_availability(self):
        """Check if Ollama is available and the model is loaded."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama server not available at {self.base_url}")
                return False
            
            # Check if the model is available
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            # Add this new logic for model fallback
            if self.model_name not in available_models:
                # Try alternative formats - sometimes models are named differently
                if "llama3" in self.model_name and "llama3.2:latest" in available_models:
                    logger.info(f"Switching from {self.model_name} to llama3.2:latest")
                    self.model_name = "llama3.2:latest"
                    return True
                elif "llama3.2" in self.model_name and "llama3" in available_models:
                    logger.info(f"Switching from {self.model_name} to llama3")
                    self.model_name = "llama3"
                    return True
                    
                logger.warning(f"Model '{self.model_name}' not found in Ollama. Available models: {available_models}")
                logger.info(f"You may need to pull the model with: ollama pull {self.model_name}")
                return False
            
            logger.info(f"Ollama is available with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
            return False
    
    def generate_response(self, user_query, conversation_context=None):
        """Generate a response using Ollama's local LLM.
        
        Args:
            user_query: The user's query text
            conversation_context: Optional additional context
            
        Returns:
            Response text from the LLM
        """
        # Update conversation history
        if conversation_context:
            # If there's specific context for this query, add it
            self.conversation_history.append({"role": "system", "content": conversation_context})
            
        # Add user query to history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Prepare messages for API call
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Add previous conversation for context (limited to last 3 turns)
        if len(self.conversation_history) > 0:
            # Get up to the last 6 messages (3 turns)
            history_messages = self.conversation_history[-min(6, len(self.conversation_history)):]
            messages.extend(history_messages)
        
        # Make API call to Ollama
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            logger.debug(f"Sending request to Ollama: {payload}")
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30  # Increase timeout for slower models
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return f"I'm sorry, I encountered an error while processing your request."
            
            response_data = response.json()
            logger.debug(f"Ollama response: {response_data}")
            
            # Extract response text
            response_text = response_data.get("message", {}).get("content", "").strip()
            
            if not response_text:
                logger.error("Empty response from Ollama")
                return "I'm sorry, I couldn't generate a response at the moment."
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response. " \
                   "Please check if Ollama is running correctly."
    
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