#!/usr/bin/env python
"""Script to generate documentation for the project."""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import importlib
import inspect
import json

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def generate_module_documentation(module_path, output_path=None):
    """Generate documentation for a Python module.
    
    Args:
        module_path: Import path to the module (e.g., 'src.stt.whisper_direct')
        output_path: Path to save the documentation (defaults to docs/[module_name].md)
        
    Returns:
        Path to the generated documentation
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Determine output path
        if output_path is None:
            module_name = module_path.split('.')[-1]
            output_dir = project_root / "docs" / "api"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{module_name}.md"
        
        # Generate documentation content
        content = f"# {module_name.replace('_', ' ').title()} Module\n\n"
        
        # Add module docstring
        if module.__doc__:
            content += f"{module.__doc__.strip()}\n\n"
        
        # Find all classes and functions in the module
        objects = inspect.getmembers(module)
        
        # Document classes
        classes = [obj for obj in objects if inspect.isclass(obj[1]) and obj[1].__module__ == module.__name__]
        if classes:
            content += "## Classes\n\n"
            
            for name, cls in classes:
                content += f"### {name}\n\n"
                
                # Add class docstring
                if cls.__doc__:
                    content += f"{cls.__doc__.strip()}\n\n"
                
                # Document methods
                methods = [method for method in inspect.getmembers(cls) if inspect.isfunction(method[1]) and not method[0].startswith('_') or method[0] == '__init__']
                
                if methods:
                    content += "#### Methods\n\n"
                    
                    for method_name, method in methods:
                        content += f"##### `{method_name}`\n\n"
                        
                        # Add method docstring
                        if method.__doc__:
                            content += f"{method.__doc__.strip()}\n\n"
                        
                        # Add method signature
                        sig = inspect.signature(method)
                        content += f"Signature: `{method_name}{sig}`\n\n"
                
                content += "\n"
        
        # Document functions
        functions = [obj for obj in objects if inspect.isfunction(obj[1]) and obj[1].__module__ == module.__name__]
        if functions:
            content += "## Functions\n\n"
            
            for name, func in functions:
                content += f"### {name}\n\n"
                
                # Add function docstring
                if func.__doc__:
                    content += f"{func.__doc__.strip()}\n\n"
                
                # Add function signature
                sig = inspect.signature(func)
                content += f"Signature: `{name}{sig}`\n\n"
        
        # Save the documentation
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"Documentation generated: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating documentation for {module_path}: {e}")
        return None

def generate_api_documentation():
    """Generate documentation for the API endpoints."""
    output_path = project_root / "docs" / "api" / "endpoints.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Define API endpoints
        endpoints = [
            {
                "path": "/",
                "method": "GET",
                "description": "Home page with API documentation",
                "parameters": [],
                "response": "HTML page with API documentation"
            },
            {
                "path": "/transcribe",
                "method": "POST",
                "description": "Transcribe audio to text using Whisper",
                "parameters": [
                    {
                        "name": "audio",
                        "type": "file",
                        "required": True,
                        "description": "Audio file to transcribe (WAV or MP3)"
                    }
                ],
                "response": "JSON with transcription and inference time"
            },
            {
                "path": "/synthesize",
                "method": "POST",
                "description": "Convert text to speech",
                "parameters": [
                    {
                        "name": "text",
                        "type": "string",
                        "required": True,
                        "description": "Text to synthesize"
                    }
                ],
                "response": "Audio file (WAV)"
            },
            {
                "path": "/assistant",
                "method": "POST",
                "description": "Process a request through the speech assistant",
                "parameters": [
                    {
                        "name": "audio",
                        "type": "file",
                        "required": True,
                        "description": "Audio file with the query (WAV or MP3)"
                    }
                ],
                "response": "JSON with input text, response text, audio URL, and timings"
            },
            {
                "path": "/audio/<filename>",
                "method": "GET",
                "description": "Retrieve audio file by filename",
                "parameters": [
                    {
                        "name": "filename",
                        "type": "string",
                        "required": True,
                        "description": "Name of the audio file to retrieve"
                    }
                ],
                "response": "Audio file (WAV)"
            },
            {
                "path": "/reset",
                "method": "POST",
                "description": "Reset the current session",
                "parameters": [],
                "response": "JSON with status and new session ID"
            }
        ]
        
        # Generate documentation
        content = "# API Endpoints\n\n"
        content += "The Speech Assistant API provides the following endpoints:\n\n"
        
        for endpoint in endpoints:
            content += f"## {endpoint['method']} {endpoint['path']}\n\n"
            content += f"{endpoint['description']}\n\n"
            
            if endpoint['parameters']:
                content += "### Parameters\n\n"
                content += "| Name | Type | Required | Description |\n"
                content += "|------|------|----------|-------------|\n"
                
                for param in endpoint['parameters']:
                    content += f"| {param['name']} | {param['type']} | {param['required']} | {param['description']} |\n"
                
                content += "\n"
            
            content += f"### Response\n\n{endpoint['response']}\n\n"
            
            # Add example if available
            if endpoint['path'] == "/transcribe":
                content += "### Example\n\n"
                content += "```bash\n"
                content += "curl -X POST -F \"audio=@sample.wav\" http://localhost:5050/transcribe\n"
                content += "```\n\n"
                content += "```json\n"
                content += "{\n  \"transcription\": \"Hello, this is a test.\",\n  \"inference_time\": 0.523\n}\n"
                content += "```\n\n"
            elif endpoint['path'] == "/assistant":
                content += "### Example\n\n"
                content += "```bash\n"
                content += "curl -X POST -F \"audio=@query.wav\" http://localhost:5050/assistant\n"
                content += "```\n\n"
                content += "```json\n"
                content += "{\n"
                content += "  \"input_text\": \"What's the weather like today?\",\n"
                content += "  \"response_text\": \"I'm sorry, I don't have access to weather information right now.\",\n"
                content += "  \"audio_url\": \"/audio/response_1234567890.wav\",\n"
                content += "  \"timings\": {\n"
                content += "    \"transcription\": 0.512,\n"
                content += "    \"response_generation\": 0.123,\n"
                content += "    \"speech_synthesis\": 0.345,\n"
                content += "    \"total\": 0.980\n"
                content += "  }\n"
                content += "}\n"
                content += "```\n\n"
        
        # Save the documentation
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"API documentation generated: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating API documentation: {e}")
        return None

def generate_index_documentation():
    """Generate index documentation."""
    output_path = project_root / "docs" / "index.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get README content
        readme_path = project_root / "README.md"
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Generate index
        content = "# Speech Assistant Documentation\n\n"
        content += "Welcome to the Speech Assistant documentation!\n\n"
        
        content += "## Contents\n\n"
        content += "* [Installation Guide](INSTALLATION.md)\n"
        content += "* [LLM Integration](LLM_INTEGRATION.md)\n"
        content += "* [API Documentation](api/endpoints.md)\n"
        content += "* [Module Documentation](api/)\n"
        content += "* [Configuration Guide](configuration.md)\n"
        
        content += "\n## Overview\n\n"
        
        # Extract overview from README
        overview_start = readme_content.find("## Features")
        overview_end = readme_content.find("## Getting Started")
        
        if overview_start != -1 and overview_end != -1:
            overview = readme_content[overview_start:overview_end].strip()
            content += overview + "\n\n"
        
        # Save the documentation
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"Index documentation generated: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating index documentation: {e}")
        return None

def generate_configuration_documentation():
    """Generate configuration documentation."""
    output_path = project_root / "docs" / "configuration.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get config.json content
        config_path = project_root / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Generate documentation
        content = "# Configuration Guide\n\n"
        content += "The Speech Assistant uses a `config.json` file for configuration.\n\n"
        
        content += "## Configuration File\n\n"
        content += "The configuration file is located at the root of the project: `config.json`.\n\n"
        
        content += "### Example Configuration\n\n"
        content += "```json\n"
        content += json.dumps(config, indent=2)
        content += "\n```\n\n"
        
        content += "## Configuration Options\n\n"
        
        # Document Whisper configuration
        content += "### Whisper Models\n\n"
        content += "Configuration options for speech recognition models:\n\n"
        content += "| Option | Type | Description | Default |\n"
        content += "|--------|------|-------------|--------|\n"
        content += "| `models.whisper.model_id` | string | HuggingFace model ID | `\"openai/whisper-tiny\"` |\n"
        content += "| `models.whisper.use_onnx` | boolean | Use ONNX runtime | `true` |\n"
        content += "| `models.whisper.use_int8` | boolean | Use INT8 quantization | `true` |\n\n"
        
        # Document TTS configuration
        content += "### TTS Models\n\n"
        content += "Configuration options for text-to-speech models:\n\n"
        content += "| Option | Type | Description | Default |\n"
        content += "|--------|------|-------------|--------|\n"
        content += "| `models.tts.voice` | string | Voice name for macOS TTS | `\"Samantha\"` |\n"
        content += "| `models.tts.rate` | number | Speaking rate (words per minute) | `170` |\n\n"
        
        # Document LLM configuration
        content += "### LLM Integration\n\n"
        content += "Configuration options for language model integration:\n\n"
        content += "| Option | Type | Description | Default |\n"
        content += "|--------|------|-------------|--------|\n"
        content += "| `models.llm.use_mock` | boolean | Use mock LLM (no API key needed) | `true` |\n"
        content += "| `models.llm.model` | string | OpenAI model name | `\"gpt-3.5-turbo\"` |\n"
        content += "| `models.llm.temperature` | number | Response randomness (0-1) | `0.7` |\n"
        content += "| `models.llm.max_tokens` | number | Maximum response length | `150` |\n"
        content += "| `models.llm.system_prompt` | string | System prompt for the LLM | *See example* |\n\n"
        
        # Document API configuration
        content += "### API Settings\n\n"
        content += "Configuration options for the API server:\n\n"
        content += "| Option | Type | Description | Default |\n"
        content += "|--------|------|-------------|--------|\n"
        content += "| `api.host` | string | API host | `\"0.0.0.0\"` |\n"
        content += "| `api.port` | number | API port | `5050` |\n"
        content += "| `api.debug` | boolean | Enable debug mode | `true` |\n\n"
        
        # Document UI configuration
        content += "### UI Settings\n\n"
        content += "Configuration options for the Streamlit UI:\n\n"
        content += "| Option | Type | Description | Default |\n"
        content += "|--------|------|-------------|--------|\n"
        content += "| `ui.title` | string | Application title | `\"Speech Assistant\"` |\n"
        content += "| `ui.theme` | string | UI theme (\"light\" or \"dark\") | `\"light\"` |\n"
        content += "| `ui.recording_duration` | number | Default recording duration (seconds) | `5` |\n\n"
        
        # Save the documentation
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"Configuration documentation generated: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating configuration documentation: {e}")
        return None

def main():
    """Generate project documentation."""
    parser = argparse.ArgumentParser(description='Generate project documentation')
    parser.add_argument('--all', action='store_true', help='Generate all documentation')
    parser.add_argument('--module', type=str, help='Generate documentation for a specific module')
    parser.add_argument('--api', action='store_true', help='Generate API documentation')
    parser.add_argument('--index', action='store_true', help='Generate index documentation')
    parser.add_argument('--config', action='store_true', help='Generate configuration documentation')
    args = parser.parse_args()
    
    # Create docs directory if it doesn't exist
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Generate all documentation if requested
    if args.all:
        # Generate module documentation
        modules = [
            'src.stt.whisper_direct',
            'src.stt.whisper_inference',
            'src.tts.mac_tts',
            'src.tts.simple_tts',
            'src.assistant.enhanced_speech_assistant',
            'src.assistant.llm_response_generator'
        ]
        
        for module in modules:
            generate_module_documentation(module)
        
        # Generate API documentation
        generate_api_documentation()
        
        # Generate index documentation
        generate_index_documentation()
        
        # Generate configuration documentation
        generate_configuration_documentation()
        
        print("All documentation generated successfully!")
        return
    
    # Generate specific documentation
    if args.module:
        generate_module_documentation(args.module)
    
    if args.api:
        generate_api_documentation()
    
    if args.index:
        generate_index_documentation()
    
    if args.config:
        generate_configuration_documentation()
    
    # If no specific option is provided, print help
    if not any([args.all, args.module, args.api, args.index, args.config]):
        parser.print_help()

if __name__ == "__main__":
    main()