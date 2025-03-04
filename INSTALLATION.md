# Installation Guide for Enhanced Speech Assistant

This guide will walk you through setting up the enhanced speech assistant with ONNX optimization and LLM integration.

## Prerequisites

- macOS (for the native TTS component)
- Python 3.9+
- Git
- OpenAI API key (optional, for LLM integration)

## Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/speech-assistant.git
cd speech-assistant
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up OpenAI API key (Optional, for LLM integration)**

```bash
# Option 1: Set as environment variable
export OPENAI_API_KEY=your_api_key_here

# Option 2: Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

5. **Download and convert models**

```bash
# Download Whisper model
python scripts/improved_whisper_converter.py

# Create sample audio for testing
mkdir -p data/audio_samples
```

## Configuration

The system uses a `config.json` file for configuration. You can edit this file to customize the behavior.

Key configuration options:
- `models.whisper.use_onnx`: Set to `true` to use the ONNX runtime (faster inference)
- `models.whisper.use_int8`: Set to `true` to use INT8 quantization (even faster, slight quality loss)
- `models.llm.use_mock`: Set to `true` to use the mock LLM (no API key needed)
- `api.port`: Change the API port (default: 5050)

## Directory Structure

```
speech-assistant/
├── api/                  # Flask API
├── data/                 # Data directory
│   ├── audio_samples/    # Test audio samples
│   └── test_outputs/     # Generated outputs
├── models/               # Model storage
│   ├── stt/              # Speech-to-text models
│   └── tts/              # Text-to-speech models
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── assistant/        # Assistant implementation
│   ├── stt/              # Speech-to-text component
│   └── tts/              # Text-to-speech component
└── ui/                   # Streamlit UI
```

## Running the Application

1. **Start the API server**

```bash
python api/enhanced_app.py
```

2. **In a separate terminal, start the UI**

```bash
streamlit run ui/enhanced_streamlit_app.py
```

3. **Open your browser and navigate to:**
   - UI: http://localhost:8501
   - API: http://localhost:5050

## Troubleshooting

### Missing ONNX model

If you see an error about a missing ONNX model:

```bash
# Run the converter script
python scripts/improved_whisper_converter.py
```

### TTS not working on macOS

Ensure you have granted terminal access to use the microphone and speech synthesis:
1. Go to System Preferences > Security & Privacy > Privacy
2. Enable access for the Terminal/Python application in both Microphone and Speech Recognition sections

### API connection issues

If the UI can't connect to the API:
1. Ensure the API is running
2. Check the port configuration in `config.json`
3. Verify there are no firewall issues blocking the connection

## Performance Benchmarking

To run performance benchmarks and generate optimization reports:

```bash
python scripts/benchmark_dashboard.py
```

This will generate visualization charts in the `data/benchmarks/` directory.