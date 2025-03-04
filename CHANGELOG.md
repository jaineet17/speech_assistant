# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-platform TTS implementation using pyttsx3
- Environment variable support for configuration
- Health check endpoint in API
- Comprehensive test suite with unit and integration tests
- Docker and docker-compose support
- GitHub Actions CI/CD pipeline
- Makefile for common development tasks
- Setup.py for proper package installation
- Code formatting and linting configuration
- Run script for easier application startup
- Contributing guidelines

### Changed
- Improved project structure for better modularity
- Enhanced error handling and logging
- Updated documentation with more detailed instructions

### Fixed
- MacOS TTS dependency by adding cross-platform alternative

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