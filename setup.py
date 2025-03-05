from setuptools import setup, find_packages

setup(
    name="speech_assistant",
    version="0.1.0",
    description="An efficient speech recognition and text-to-speech system with LLM integration",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.39.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "sounddevice>=0.4.6",
        "flask>=2.3.3",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "python-multipart>=0.0.6",
        "tqdm>=4.66.1",
        "matplotlib>=3.8.2",
        "pandas>=2.1.4",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "pyttsx3>=2.90",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "speech-assistant-api=api.enhanced_app:main",
            "speech-assistant-ui=ui.enhanced_streamlit_app:main",
        ],
    },
) 