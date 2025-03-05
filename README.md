# Enhanced Speech Assistant

A modern speech recognition and text-to-speech system with LLM integration for intelligent responses, optimized for performance and user experience.

## Features

- **Speech Recognition**: Accurate transcription of spoken audio
- **Text-to-Speech**: Cross-platform TTS with caching for improved performance
- **LLM Integration**: Uses Llama 2 7B model for intelligent responses
- **Performance Optimization**: Caching, GPU acceleration, and threading for faster processing
- **Modern React Frontend**: Responsive UI with conversation history and audio playback
- **API Backend**: FastAPI-based backend with comprehensive error handling
- **Performance Metrics**: Built-in benchmarking to measure and optimize performance

## System Architecture

The system consists of three main components:
1. **Speech-to-Text (STT)**: Converts spoken audio to text
2. **LLM**: Generates intelligent responses to user queries using Llama 2
3. **Text-to-Speech (TTS)**: Converts response text to speech output with platform-specific optimizations

## Performance Optimization

This project focuses on optimization for better user experience:

- **GPU Acceleration**: Utilizes GPU layers for faster LLM inference
- **Caching**: Response and TTS caching to improve repeated query performance
- **Asynchronous Processing**: Non-blocking operations for better responsiveness
- **Platform-specific TTS**: Optimized TTS implementations for macOS, Windows, and Linux
- **Performance Metrics**: Real-time tracking of processing times

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js and npm for the frontend
- [Optional] CUDA-compatible GPU for acceleration

### Installation

#### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-assistant.git
cd speech-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the Llama 2 7B model (if not already downloaded)
# Place it in the models/ directory
```

#### Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Build the frontend (optional for production)
npm run build
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### Running the Application

#### Running the Backend

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the backend server
cd api
uvicorn enhanced_app:app --host 0.0.0.0 --port 5050 --reload
```

#### Running the Frontend

```bash
# In a separate terminal, navigate to the frontend directory
cd frontend

# Start the development server
npm run dev
```

Then open your browser and navigate to http://localhost:5173 (or the port shown in the terminal)

## Configuration

The application can be configured through environment variables or the `config.json` file. Key options include:

- **LLM Settings**: Model path, number of GPU layers, and thread count
- **TTS Settings**: Voice selection and speech rate
- **API Settings**: Host, port, and CORS configuration

## Features in Detail

### Speech Recognition

The system uses advanced speech recognition to accurately transcribe user audio. It supports various audio formats and includes fallback mechanisms for handling different recording devices.

### LLM Integration

The system uses the Llama 2 7B model for generating responses. Key features include:

- **Context-aware responses**: The model understands the context of the conversation
- **Performance optimization**: GPU acceleration and caching for faster responses
- **Error handling**: Graceful fallbacks for handling model errors

### Text-to-Speech

The TTS system includes:

- **Cross-platform support**: Works on macOS, Windows, and Linux
- **Performance optimization**: Caching and preloading for reduced latency
- **Fallback mechanisms**: Ensures audio is always generated even if the primary method fails

### Frontend

The React-based frontend provides:

- **Conversation history**: Maintains a record of the interaction
- **Audio playback**: Integrated audio player for TTS output
- **Responsive design**: Works well on different screen sizes
- **Performance metrics**: Displays processing times for transparency

## Performance Benchmarks

The system achieves the following performance metrics:

| Component | Average Processing Time |
|-----------|-------------------------|
| Speech-to-Text | 0.5-2.0 seconds |
| LLM Processing | 2.0-15.0 seconds (12-18 tokens/sec) |
| Text-to-Speech | 0.5-1.5 seconds |

## Future Improvements

- [ ] Implement streaming responses for real-time feedback
- [ ] Add support for more languages
- [ ] Integrate smaller, faster models for edge deployment
- [ ] Implement voice customization options
- [ ] Add user authentication and session management

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Meta for the Llama 2 model
- FastAPI for the backend framework
- React for the frontend framework