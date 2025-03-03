"""Flask API for the speech assistant."""

import os
import sys
import traceback
from flask import Flask, request, jsonify, send_file
from pathlib import Path
import time

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add the project root to the Python path
sys.path.append(str(PROJECT_ROOT))

# Configure directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import components
from src.stt.whisper_direct import WhisperDirect
from src.tts.mac_tts import MacTTS
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

# Initialize TTS - Use Mac-specific implementation
try:
    tts_model = MacTTS("Samantha")  # Try a different voice
    print("Using macOS TTS engine")
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
    
    try:
        # Get file from request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save to a temporary file
        timestamp = int(time.time())
        file_path = OUTPUT_DIR / f"upload_{timestamp}.wav"
        audio_file.save(file_path)
        
        # Transcribe
        transcription = whisper_model.transcribe(str(file_path))
        return jsonify({"transcription": transcription})
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Synthesize text to speech."""
    if not tts_model:
        return jsonify({"error": "TTS model not available"}), 503
    
    try:
        # Get text from request
        if request.form and 'text' in request.form:
            text = request.form['text']
        elif request.json and 'text' in request.json:
            text = request.json['text']
        else:
            return jsonify({"error": "No text provided"}), 400
        
        # Synthesize
        timestamp = int(time.time())
        output_path = OUTPUT_DIR / f"synth_{timestamp}.wav"
        
        tts_model.synthesize(text, str(output_path))
        
        # Return audio file
        return send_file(str(output_path), mimetype="audio/wav")
    except Exception as e:
        error_msg = f"Error synthesizing speech: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/assistant', methods=['POST'])
def assistant_endpoint():
    """Process a request through the speech assistant."""
    if not assistant:
        return jsonify({"error": "Speech assistant not available"}), 503
    
    try:
        # Get file from request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save to a temporary file
        timestamp = int(time.time())
        file_path = OUTPUT_DIR / f"query_{timestamp}.wav"
        audio_file.save(file_path)
        
        # Process with assistant
        result = assistant.process_query(str(file_path))
        
        # Make sure the file exists
        response_audio_path = Path(result["response_audio"])
        if not response_audio_path.exists():
            return jsonify({"error": f"Generated audio file not found: {response_audio_path}"}), 500
        
        return jsonify({
            "input_text": result["input_text"],
            "response_text": result["response_text"],
            "audio_url": f"/audio/{response_audio_path.name}"
        })
    except Exception as e:
        error_msg = f"Error processing assistant request: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Retrieve audio file by filename."""
    try:
        file_path = OUTPUT_DIR / filename
        if file_path.exists():
            return send_file(str(file_path), mimetype="audio/wav")
        else:
            return jsonify({"error": f"Audio file not found: {file_path}"}), 404
    except Exception as e:
        error_msg = f"Error retrieving audio file: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    print(f"Starting API server on port 5050")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=5050, debug=True)