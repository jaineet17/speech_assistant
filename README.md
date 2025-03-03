# Speech Assistant

An efficient speech recognition and text-to-speech system with LLM integration for intelligent responses.

## Features

- Speech-to-Text using OpenAI's Whisper model
- Text-to-Speech using macOS native capabilities
- LLM integration for natural language responses
- Simple and intuitive web interface

## Components

- Speech recognition (STT) using Whisper
- Text-to-speech (TTS) using macOS voice synthesis
- Response generation with OpenAI API integration
- Flask API backend
- Streamlit UI frontend

## Setup

### Prerequisites

- Python 3.9+
- macOS (for the TTS component)
- OpenAI API key (for LLM responses)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/speech-assistant.git
   cd speech-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

1. Start the API server:
   ```
   python api/app.py
   ```

2. In a separate terminal, start the UI:
   ```
   streamlit run ui/streamlit_app.py
   ```

3. Open your browser and navigate to http://localhost:8501

## License

MIT
