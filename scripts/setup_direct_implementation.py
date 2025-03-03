#!/usr/bin/env python
"""Script to set up a direct implementation without ONNX conversion."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_direct_implementation():
    """Set up a direct implementation using PyTorch models."""
    try:
        # Create source directories
        src_dir = Path(__file__).parent.parent / "src"
        stt_dir = src_dir / "stt"
        tts_dir = src_dir / "tts"
        assistant_dir = src_dir / "assistant"
        
        os.makedirs(stt_dir, exist_ok=True)
        os.makedirs(tts_dir, exist_ok=True)
        os.makedirs(assistant_dir, exist_ok=True)
        
        # Install pyttsx3 for TTS
        import subprocess
        print("Installing pyttsx3...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyttsx3"],
            check=True
        )
        
        # Create direct implementation of Whisper
        whisper_direct_path = stt_dir / "whisper_direct.py"
        with open(whisper_direct_path, "w") as f:
            f.write('''"""Direct implementation of Whisper without ONNX."""

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
''')
        
        # Create simple TTS implementation
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
            # Try to find a female voice
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
        
        # Create assistant implementation
        assistant_path = assistant_dir / "speech_assistant.py"
        with open(assistant_path, "w") as f:
            f.write('''"""Speech assistant implementation."""

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
        
        # Create API implementation
        api_dir = Path(__file__).parent.parent / "api"
        os.makedirs(api_dir, exist_ok=True)
        
        app_path = api_dir / "app.py"
        with open(app_path, "w") as f:
            f.write('''"""Flask API for the speech assistant."""

import os
import sys
from flask import Flask, request, jsonify, send_file
from pathlib import Path
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stt.whisper_direct import WhisperDirect
from src.tts.simple_tts import SimpleTTS
from src.assistant.speech_assistant import SpeechAssistant

app = Flask(__name__)

# Initialize models and assistant
print("Initializing models...")

# Initialize Whisper
try:
    whisper_model = WhisperDirect("openai/whisper-tiny")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# Initialize TTS
try:
    tts_model = SimpleTTS()
    print("TTS model initialized")
except Exception as e:
    print(f"Error initializing TTS: {e}")
    tts_model = None

# Initialize assistant if models are available
if whisper_model and tts_model:
    assistant = SpeechAssistant(whisper_model, tts_model)
    print("Speech assistant initialized")
else:
    assistant = None
    print("Warning: Speech assistant not initialized due to missing models")

@app.route('/')
def index():
    """Home page."""
    return '''
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
                    Whisper model: <strong>''' + ("Loaded" if whisper_model else "Not loaded") + '''</strong><br>
                    TTS model: <strong>''' + ("Loaded" if tts_model else "Not loaded") + '''</strong><br>
                    Assistant: <strong>''' + ("Ready" if assistant else "Not available") + '''</strong>
                </p>
            </div>
        </body>
    </html>
    '''

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
        
        # Create UI implementation
        ui_dir = Path(__file__).parent.parent / "ui"
        os.makedirs(ui_dir, exist_ok=True)
        
        ui_path = ui_dir / "streamlit_app.py"
        with open(ui_path, "w") as f:
            f.write('''"""Streamlit UI for the speech assistant."""

import streamlit as st
import requests
import soundfile as sf
import sounddevice as sd
import numpy as np
import time
import os
from io import BytesIO

# Configure API URL
API_URL = "http://localhost:5000"

# Set page title
st.set_page_config(page_title="Speech Assistant", page_icon="🎤")

st.title("🎤 Speech Assistant")
st.write("A local speech recognition and text-to-speech system")

# Sidebar with options
st.sidebar.title("Options")
record_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    with st.spinner(f"Recording for {duration} seconds..."):
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Display progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()
        progress_bar.empty()
        st.success("Recording finished!")
        
        return audio, sample_rate

def save_audio(audio_data, sample_rate=16000):
    """Save audio data to a temporary file."""
    temp_dir = "data/test_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = f"{temp_dir}/streamlit_{int(time.time())}.wav"
    sf.write(temp_path, audio_data, sample_rate)
    
    return temp_path

# Create tabs
tab1, tab2, tab3 = st.tabs(["Assistant", "Speech-to-Text", "Text-to-Speech"])

with tab1:
    st.header("Voice Assistant")
    
    if st.button("Record Voice Query", key="record_assistant"):
        # Record audio
        audio_data, sample_rate = record_audio(duration=record_duration)
        
        # Save audio
        audio_path = save_audio(audio_data, sample_rate)
        
        # Play back the recording
        st.audio(audio_path)
        
        # Send to API
        with st.spinner("Processing..."):
            files = {'audio': open(audio_path, 'rb')}
            response = requests.post(f"{API_URL}/assistant", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("Result:")
                st.write(f"You said: {result['input_text']}")
                st.write(f"Assistant: {result['response_text']}")
                
                # Get audio response
                audio_url = f"{API_URL}{result['audio_url']}"
                audio_response = requests.get(audio_url)
                
                if audio_response.status_code == 200:
                    st.audio(audio_response.content, format="audio/wav")
            else:
                st.error(f"Error: {response.text}")

with tab2:
    st.header("Speech-to-Text")
    
    if st.button("Record for Transcription", key="record_transcribe"):
        # Record audio
        audio_data, sample_rate = record_audio(duration=record_duration)
        
        # Save audio
        audio_path = save_audio(audio_data, sample_rate)
        
        # Play back the recording
        st.audio(audio_path)
        
        # Send to API
        with st.spinner("Transcribing..."):
            files = {'audio': open(audio_path, 'rb')}
            response = requests.post(f"{API_URL}/transcribe", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("Transcription:")
                st.write(result['transcription'])
            else:
                st.error(f"Error: {response.text}")
    
    # File upload option
    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Transcribe File"):
            files = {'audio': uploaded_file}
            with st.spinner("Transcribing..."):
                response = requests.post(f"{API_URL}/transcribe", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Transcription:")
                    st.write(result['transcription'])
                else:
                    st.error(f"Error: {response.text}")

with tab3:
    st.header("Text-to-Speech")
    
    text_input = st.text_area("Enter text to synthesize:", "Hello, this is a test of the speech synthesis system.")
    
    if st.button("Generate Speech"):
        with st.spinner("Generating speech..."):
            response = requests.post(
                f"{API_URL}/synthesize", 
                json={"text": text_input}
            )
            
            if response.status_code == 200:
                st.audio(response.content, format="audio/wav")
            else:
                st.error(f"Error: {response.text}")

st.sidebar.markdown("---")
st.sidebar.caption("Speech Assistant | Local Inference")
''')
        
        print("Direct implementation files created successfully!")
        return True
    except Exception as e:
        print(f"Error setting up direct implementation: {e}")
        return False

if __name__ == "__main__":
    success = setup_direct_implementation()
    if success:
        print("\nDirect implementation setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python api/app.py")
        print("2. In a new terminal, start the UI: streamlit run ui/streamlit_app.py")
    else:
        print("\nFailed to set up direct implementation. Please check the error messages above.")
