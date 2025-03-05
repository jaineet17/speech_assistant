# Installation Guide for Enhanced Speech Assistant

This guide will walk you through setting up the enhanced speech assistant with Llama 2 integration and React frontend.

## Prerequisites

- Python 3.9+
- Node.js and npm (for the frontend)
- Git
- [Optional] CUDA-compatible GPU for acceleration

## Installation Steps

### Backend Setup

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

4. **Download the Llama 2 model**

You need to download the Llama 2 7B model. There are several options:

- Download from Hugging Face (requires account): https://huggingface.co/meta-llama/Llama-2-7b
- Use a quantized version like llama-2-7b.ggmlv3.q4_0.bin for better performance

Place the model file in the `models/` directory.

5. **Create necessary directories**

```bash
mkdir -p temp
mkdir -p responses
```

### Frontend Setup

1. **Navigate to the frontend directory**

```bash
cd frontend
```

2. **Install dependencies**

```bash
npm install
```

3. **[Optional] Build the frontend for production**

```bash
npm run build
```

## Running the Application

### Starting the Backend

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the backend server
cd api
uvicorn enhanced_app:app --host 0.0.0.0 --port 5050 --reload
```

### Starting the Frontend (Development Mode)

```bash
# In a separate terminal, navigate to the frontend directory
cd frontend

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:5173 (or the port shown in the terminal).

## Configuration

### Backend Configuration

The backend can be configured through environment variables or by editing the `config.json` file:

```json
{
  "llm": {
    "model_path": "models/llama-2-7b.ggmlv3.q4_0.bin",
    "n_gpu_layers": 1,
    "n_threads": 8
  },
  "tts": {
    "cache_size": 50
  },
  "api": {
    "port": 5050,
    "host": "0.0.0.0"
  }
}
```

Key configuration options:
- `llm.model_path`: Path to the Llama 2 model file
- `llm.n_gpu_layers`: Number of layers to offload to GPU (set to 0 for CPU-only)
- `llm.n_threads`: Number of CPU threads to use for inference
- `tts.cache_size`: Maximum number of TTS outputs to cache
- `api.port`: API server port
- `api.host`: API server host

### Frontend Configuration

The frontend connects to the backend API at `http://localhost:5050` by default. To change this:

1. Edit the API URL in `frontend/src/services/api.ts`
2. Rebuild the frontend if necessary

## Directory Structure

```
speech-assistant/
├── api/                  # FastAPI backend
├── frontend/             # React frontend
│   ├── src/              # Frontend source code
│   ├── public/           # Static assets
│   └── package.json      # Frontend dependencies
├── models/               # Model storage for Llama 2
├── src/                  # Backend source code
│   ├── llm/              # LLM implementation
│   ├── stt/              # Speech-to-text component
│   └── tts/              # Text-to-speech component
├── temp/                 # Temporary files for audio
└── responses/            # Saved conversation history
```

## Troubleshooting

### LLM not loading

If you see an error about the Llama 2 model:

1. Ensure the model file exists in the `models/` directory
2. Check the model path in `config.json`
3. Verify you have enough RAM (at least 8GB for the quantized model)
4. For GPU acceleration, ensure you have a compatible GPU and drivers

### TTS not working

Ensure you have the necessary system dependencies:

- **macOS**: No additional dependencies needed
- **Windows**: Ensure you have the Speech API installed
- **Linux**: Install espeak with `sudo apt-get install espeak`

### Frontend connection issues

If the frontend can't connect to the API:

1. Ensure the API is running
2. Check the API URL in the frontend code
3. Verify there are no CORS issues or firewall blocks

### Audio recording issues

If audio recording doesn't work:

1. Ensure your browser has permission to access the microphone
2. Try using a different browser (Chrome or Firefox recommended)
3. Check if your microphone is working with other applications