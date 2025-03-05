FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CPU support to reduce image size
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/audio_samples data/test_outputs models/stt models/tts

# Expose ports for API and UI
EXPOSE 5050 8501

# Set environment variables
ENV PYTHONPATH=/app

# Create a startup script
RUN echo '#!/bin/bash\n\
python api/enhanced_app.py &\n\
streamlit run ui/enhanced_streamlit_app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run when the container starts
CMD ["/app/start.sh"] 