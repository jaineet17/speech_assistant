"""Enhanced Flask API for the speech assistant with LLM integration."""

import os
import sys
import traceback
from flask import Flask, request, jsonify, send_file
from pathlib import Path
import time
import json

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
from src.assistant.enhanced_speech_assistant import EnhancedSpeechAssistant
from src.assistant.llm_response_generator import get_response_generator
from src.utils.env_loader import load_config_with_env_override

app = Flask(__name__)

# Load configuration with environment variable overrides
config = load_config_with_env_override()

# Initialize models and assistant
print("Initializing models...")

# Initialize Whisper
try:
    whisper_config = config["models"]["whisper"]
    use_onnx = whisper_config.get("use_onnx", False)
    
    if use_onnx:
        from src.stt.whisper_inference import WhisperONNX
        onnx_path = PROJECT_ROOT / "models" / "stt" / "whisper_onnx" / "model.onnx"
        int8_path = PROJECT_ROOT / "models" / "stt" / "whisper_onnx" / "model_int8.onnx"
        
        if int8_path.exists():
            whisper_model = WhisperONNX(str(int8_path), whisper_config["model_id"])
            print("Whisper INT8 ONNX model loaded successfully")
        elif onnx_path.exists():
            whisper_model = WhisperONNX(str(onnx_path), whisper_config["model_id"])
            print("Whisper ONNX model loaded successfully")
        else:
            print("ONNX model not found, falling back to PyTorch")
            whisper_model = WhisperDirect(whisper_config["model_id"])
    else:
        whisper_model = WhisperDirect(whisper_config["model_id"])
        print("Whisper PyTorch model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print(traceback.format_exc())
    whisper_model = None

# Initialize TTS - Use Mac-specific implementation
try:
    tts_config = config["models"]["tts"]
    tts_model = MacTTS(tts_config.get("voice", "Samantha"))
    print(f"Using macOS TTS engine with voice: {tts_config.get('voice', 'Samantha')}")
except Exception as e:
    print(f"Error initializing MacTTS: {e}")
    print("Falling back to cross-platform TTS...")
    try:
        from src.tts.cross_platform_tts import CrossPlatformTTS
        tts_model = CrossPlatformTTS(
            voice=tts_config.get("voice"),
            rate=tts_config.get("rate", 170)
        )
        print(f"Using cross-platform TTS engine")
    except Exception as fallback_error:
        print(f"Error initializing cross-platform TTS: {fallback_error}")
        print(traceback.format_exc())
        tts_model = None

# Initialize LLM response generator
try:
    llm_config = config["models"]["llm"]
    use_mock = llm_config.get("use_mock", False)
    use_ollama = llm_config.get("use_ollama", True)
    
    response_generator = get_response_generator(use_mock=use_mock, use_ollama=use_ollama)
    print(f"LLM response generator initialized (Mock: {use_mock}, Ollama: {use_ollama})")
except Exception as e:
    print(f"Error initializing LLM response generator: {e}")
    print(traceback.format_exc())
    response_generator = None

# Initialize assistant
if whisper_model and tts_model:
    assistant = EnhancedSpeechAssistant(whisper_model, tts_model, response_generator, config)
    print("Speech assistant initialized")
else:
    assistant = None
    print("Speech assistant not initialized due to missing components")

# API routes
@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        "name": "Enhanced Speech Assistant API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/transcribe",
            "/synthesize",
            "/process_audio",
            "/assistant",
            "/health"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "ok",
        "timestamp": time.time(),
        "components": {
            "whisper": "available" if whisper_model else "unavailable",
            "tts": "available" if tts_model else "unavailable",
            "llm": "available" if response_generator else "unavailable"
        },
        "version": "0.1.0"
    }
    return jsonify(status)

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
        start_time = time.time()
        transcription = whisper_model.transcribe(str(file_path))
        inference_time = time.time() - start_time
        
        return jsonify({
            "transcription": transcription,
            "inference_time": inference_time
        })
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
        
        start_time = time.time()
        tts_model.synthesize(text, str(output_path))
        synthesis_time = time.time() - start_time
        
        # Return audio file
        return send_file(str(output_path), mimetype="audio/wav", 
                         download_name=f"speech_{timestamp}.wav",
                         as_attachment=False)
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
            "audio_url": f"/audio/{response_audio_path.name}",
            "timings": result["timings"]
        })
    except Exception as e:
        error_msg = f"Error processing assistant request: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/audio/<path:filename>', methods=['GET'])
def get_audio(filename):
    """Retrieve audio file by filename."""
    try:
        # Ensure the filename doesn't contain path traversal
        safe_filename = os.path.basename(filename)
        
        # First check current session directory if available
        if assistant and hasattr(assistant, 'session_dir'):
            file_path = assistant.session_dir / safe_filename
            if file_path.exists():
                return send_file(str(file_path), mimetype="audio/wav")
        
        # Then check general output directory
        file_path = OUTPUT_DIR / safe_filename
        if file_path.exists():
            return send_file(str(file_path), mimetype="audio/wav")
        else:
            return jsonify({"error": f"Audio file not found: {safe_filename}"}), 404
    except Exception as e:
        error_msg = f"Error retrieving audio file: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset the current session."""
    if not assistant:
        return jsonify({"error": "Speech assistant not available"}), 503
    
    try:
        new_session_id = assistant.reset_session()
        return jsonify({"status": "success", "new_session_id": new_session_id})
    except Exception as e:
        error_msg = f"Error resetting session: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

def main():
    """Run the Flask API server."""
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 5050)
    debug = api_config.get("debug", True)
    
    print(f"Starting Enhanced API server on {host}:{port}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()