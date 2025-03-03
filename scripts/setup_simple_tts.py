#!/usr/bin/env python
"""Script to set up a simple TTS system."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_simple_tts():
    """Set up a simple TTS system using pyttsx3."""
    try:
        # Install pyttsx3
        import subprocess
        print("Installing pyttsx3...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyttsx3"],
            check=True
        )
        
        # Create a simple TTS implementation
        tts_dir = Path(__file__).parent.parent / "src" / "tts"
        os.makedirs(tts_dir, exist_ok=True)
        
        # Create simple_tts.py
        simple_tts_path = tts_dir / "simple_tts.py"
        with open(simple_tts_path, "w") as f:
            f.write('''"""Simple TTS implementation using pyttsx3."""

import pyttsx3
import os

class SimpleTTS:
    """Simple TTS class using pyttsx3."""
    
    def __init__(self):
        """Initialize the TTS engine."""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speaking rate
        
        # Try to use a good voice if available
        voices = self.engine.getProperty('voices')
        if voices:
            # Prefer a female voice if available
            female_voices = [v for v in voices if hasattr(v, 'gender') and v.gender == 'female']
            if female_voices:
                self.engine.setProperty('voice', female_voices[0].id)
            else:
                self.engine.setProperty('voice', voices[0].id)
    
    def synthesize(self, text, output_path):
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio output
            
        Returns:
            Path to the synthesized audio file
        """
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save speech to file
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        
        return output_path
''')
        
        # Update config.py to use simple TTS
        config_path = Path(__file__).parent.parent / "src" / "config.py"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_content = f.read()
            
            # Update TTS paths
            updated_content = config_content.replace(
                'TTS_ONNX_PATH = str(MODELS_DIR / "tts" / "vits_int8.onnx")',
                'TTS_ONNX_PATH = str(MODELS_DIR / "tts" / "simple_tts_path")'
            )
            
            with open(config_path, "w") as f:
                f.write(updated_content)
        
        # Create a modified version of whisper_inference.py
        whisper_inference_path = Path(__file__).parent.parent / "src" / "stt" / "whisper_inference.py"
        with open(whisper_inference_path, "r") as f:
            whisper_content = f.read()
        
        # Modify to use regular ONNX path
        modified_content = whisper_content.replace(
            'STT_ONNX_PATH', 
            '"models/stt/whisper_onnx/model.onnx"'
        )
        
        with open(whisper_inference_path, "w") as f:
            f.write(modified_content)
        
        # Create a simple assistant implementation
        assistant_dir = Path(__file__).parent.parent / "src" / "assistant"
        os.makedirs(assistant_dir, exist_ok=True)
        
        assistant_path = assistant_dir / "speech_assistant.py"
        with open(assistant_path, "w") as f:
            f.write('''"""Simple speech assistant implementation."""

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
        
        # Import SimpleTTS if no TTS model provided
        if tts_model is None:
            from src.tts.simple_tts import SimpleTTS
            self.tts_model = SimpleTTS()
        else:
            self.tts_model = tts_model
        
        self.output_dir = Path("data/test_outputs")
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
        self.tts_model.synthesize(response, output_path)
        
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
        
        else:
            return "I heard you, but I'm not sure how to respond to that query. Could you try asking something else?"
''')
        
        # Create a simple API implementation
        api_dir = Path(__file__).parent.parent / "api"
        os.makedirs(api_dir, exist_ok=True)
        
        app_path = api_dir / "app.py"
        with open(app_path, "w") as f:
            f.write('''"""Flask API for the speech assistant."""

import os
import sys
import base64
from flask import Flask, request, jsonify, send_file, render_template_string
from pathlib import Path
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stt.whisper_inference import WhisperONNX
from src.tts.simple_tts import SimpleTTS
from src.assistant.speech_assistant import SpeechAssistant

app = Flask(__name__)

# Initialize components
try:
    whisper_model = WhisperONNX("models/stt/whisper_onnx/model.onnx", "openai/whisper-tiny")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# Use SimpleTTS
tts_model = SimpleTTS()
print("TTS model initialized")

# Initialize assistant
if whisper_model:
    assistant = SpeechAssistant(whisper_model, tts_model)
    print("Speech assistant initialized")
else:
    assistant = None
    print("Warning: Speech assistant not initialized due to missing models")

# Store temporary files
temp_files = {}

@app.route('/')
def index():
    """Render simple test page."""
    return """
    <html>
    <head>
        <title>Speech Assistant API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
            code { background-color: #eee; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Speech Assistant API</h1>
        
        <div class="endpoint">
            <h2>POST /transcribe</h2>
            <p>Upload an audio file to transcribe speech to text.</p>
            <form action="/transcribe" method="post" enctype="multipart/form-data">
                <input type="file" name="audio">
                <button type="submit">Transcribe</button>
            </form>
        </div>
        
        <div class="endpoint">
            <h2>POST /synthesize</h2>
            <p>Convert text to speech.</p>
            <form action="/synthesize" method="post">
                <textarea name="text" rows="4" cols="50">Hello, this is a test of the speech synthesis system.</textarea>
                <button type="submit">Synthesize</button>
            </form>
        </div>
        
        <div class="endpoint">
            <h2>Status</h2>
            <p>
                Whisper model: <strong>{"Loaded" if whisper_model else "Not loaded"}</strong><br>
                TTS model: <strong>{"Loaded" if tts_model else "Not loaded"}</strong><br>
                Assistant: <strong>{"Ready" if assistant else "Not available"}</strong>
            </p>
        </div>
    </body>
    </html>
    """

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text."""
    if not whisper_model:
        return jsonify({"error": "Whisper model not available"}), 503
    
    # Get file from request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Save to a temporary file
    timestamp = int(time.time())
    file_path = f"data/test_outputs/upload_{timestamp}.wav"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    audio_file.save(file_path)
    
    # Transcribe
    try:
        transcription = whisper_model.transcribe(file_path)
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Synthesize text to speech."""
    if not tts_model:
        return jsonify({"error": "TTS model not available"}), 503
    
    # Get text from request
    if request.form and 'text' in request.form:
        text = request.form['text']
    elif request.json and 'text' in request.json:
        text = request.json['text']
    else:
        return jsonify({"error": "No text provided"}), 400
    
    # Synthesize
    try:
        timestamp = int(time.time())
        output_path = f"data/test_outputs/synth_{timestamp}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tts_model.synthesize(text, output_path)
        
        # Return audio file
        return send_file(output_path, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/assistant', methods=['POST'])
def assistant_endpoint():
    """Process a request through the speech assistant."""
    if not assistant:
        return jsonify({"error": "Speech assistant not available"}), 503
    
    # Get file from request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Save to a temporary file
    timestamp = int(time.time())
    file_path = f"data/test_outputs/query_{timestamp}.wav"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    audio_file.save(file_path)
    
    # Process with assistant
    try:
        result = assistant.process_query(file_path)
        
        return jsonify({
            "input_text": result["input_text"],
            "response_text": result["response_text"],
            "audio_url": f"/audio/{os.path.basename(result['response_audio'])}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Retrieve audio file by filename."""
    file_path = f"data/test_outputs/{filename}"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/wav")
    else:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    print("Starting API server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
''')
        
        print("Simplified implementation files created successfully!")
        return True
    except Exception as e:
        print(f"Error setting up simple TTS: {e}")
        return False

if __name__ == "__main__":
    success = setup_simple_tts()
    if success:
        print("\nSimple TTS setup completed successfully!")
    else:
        print("\nFailed to set up simple TTS. Please check the error messages above.")
