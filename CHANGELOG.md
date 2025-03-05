# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern React frontend with conversation history
- Audio player with improved UI and controls
- Llama 2 7B model integration for LLM responses
- GPU acceleration for LLM processing
- Cross-platform TTS implementation with caching
- TTS preloading to reduce first-time latency
- Performance metrics tracking and display
- FastAPI backend with comprehensive error handling
- Environment variable support for configuration
- Health check endpoint in API

### Changed
- Replaced Streamlit UI with React frontend
- Improved project structure for better modularity
- Enhanced error handling and logging
- Optimized TTS implementation for all platforms
- Updated documentation with detailed instructions
- Improved audio processing pipeline

### Fixed
- Audio playback issues in the frontend
- TTS functionality across different platforms
- LLM response generation reliability
- Error handling in the API endpoints
- Performance bottlenecks in processing pipeline

## [0.1.0] - 2023-03-01

### Added
- Initial release
- Speech-to-text using Whisper model
- Text-to-speech using macOS native voices
- LLM integration with OpenAI API and Ollama
- ONNX runtime optimization
- Streamlit UI
- Flask API
- Basic benchmarking tools 