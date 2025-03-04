#!/usr/bin/env python
"""Test script to verify the complete enhanced speech assistant pipeline."""

import os
import sys
import time
import json
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from src.stt.whisper_direct import WhisperDirect
try:
    from src.stt.whisper_inference import WhisperONNX
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from src.tts.mac_tts import MacTTS
from src.tts.simple_tts import SimpleTTS
from src.assistant.enhanced_speech_assistant import EnhancedSpeechAssistant
from src.assistant.llm_response_generator import get_response_generator

def create_test_audio(output_path, duration=3, sample_rate=16000):
    """Create a test audio file with a voice-like pattern."""
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a mix of frequencies to simulate voice
    audio = 0.3 * np.sin(2 * np.pi * 240 * t)  # Base frequency
    audio += 0.2 * np.sin(2 * np.pi * 480 * t)  # First harmonic
    audio += 0.1 * np.sin(2 * np.pi * 720 * t)  # Second harmonic
    
    # Add some variation
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    audio = audio * envelope
    
    # Add fade in/out
    fade_samples = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    # Add a slight amount of noise
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise
    
    # Ensure we don't clip
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to file
    sf.write(output_path, audio, sample_rate)
    
    print(f"Created test audio file: {output_path}")
    return output_path

def test_whisper_models(test_audio_path):
    """Test Whisper models (PyTorch and ONNX if available)."""
    print("\n===== Testing Whisper Models =====")
    
    # Test PyTorch model
    print("\nTesting Whisper PyTorch model:")
    try:
        start_time = time.time()
        pytorch_model = WhisperDirect("openai/whisper-tiny")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        
        start_time = time.time()
        pytorch_result = pytorch_model.transcribe(test_audio_path)
        inference_time = time.time() - start_time
        print(f"PyTorch transcription: '{pytorch_result}'")
        print(f"Inference time: {inference_time:.4f}s")
        pytorch_success = True
    except Exception as e:
        print(f"Error with PyTorch model: {e}")
        pytorch_success = False
        pytorch_result = None
        inference_time = 0
    
    results = {
        "pytorch": {
            "success": pytorch_success,
            "result": pytorch_result,
            "inference_time": inference_time
        }
    }
    
    # Test ONNX model if available
    if ONNX_AVAILABLE:
        print("\nTesting Whisper ONNX model:")
        
        # Find ONNX model
        onnx_path = project_root / "models" / "stt" / "whisper_onnx" / "model.onnx"
        int8_path = project_root / "models" / "stt" / "whisper_onnx" / "model_int8.onnx"
        
        if int8_path.exists():
            onnx_model_path = int8_path
            model_type = "INT8"
        elif onnx_path.exists():
            onnx_model_path = onnx_path
            model_type = "FP32"
        else:
            print("ONNX model not found. Run scripts/improved_whisper_converter.py first.")
            results["onnx"] = {"success": False}
            return results
        
        try:
            start_time = time.time()
            onnx_model = WhisperONNX(str(onnx_model_path), "openai/whisper-tiny")
            load_time = time.time() - start_time
            print(f"ONNX {model_type} model loaded in {load_time:.2f}s")
            
            start_time = time.time()
            onnx_result = onnx_model.transcribe(test_audio_path)
            inference_time = time.time() - start_time
            print(f"ONNX transcription: '{onnx_result}'")
            print(f"Inference time: {inference_time:.4f}s")
            
            # Calculate speedup
            if pytorch_success:
                speedup = results["pytorch"]["inference_time"] / inference_time
                print(f"Speedup compared to PyTorch: {speedup:.2f}x")
            
            results["onnx"] = {
                "success": True,
                "result": onnx_result,
                "inference_time": inference_time,
                "model_type": model_type
            }
            
            if pytorch_success:
                results["speedup"] = speedup
        except Exception as e:
            print(f"Error with ONNX model: {e}")
            results["onnx"] = {"success": False, "error": str(e)}
    
    return results

def test_tts_models(test_text="This is a test of the speech synthesis system."):
    """Test TTS models."""
    print("\n===== Testing TTS Models =====")
    
    results = {}
    
    # Test Mac TTS
    print("\nTesting macOS TTS:")
    try:
        output_path = project_root / "data" / "test_outputs" / "mac_tts_test.wav"
        
        start_time = time.time()
        mac_tts = MacTTS("Samantha")
        load_time = time.time() - start_time
        print(f"TTS engine initialized in {load_time:.2f}s")
        
        start_time = time.time()
        mac_tts.synthesize(test_text, str(output_path))
        synthesis_time = time.time() - start_time
        print(f"Speech synthesized to {output_path}")
        print(f"Synthesis time: {synthesis_time:.4f}s")
        
        results["mac_tts"] = {
            "success": True,
            "output_path": str(output_path),
            "synthesis_time": synthesis_time
        }
    except Exception as e:
        print(f"Error with macOS TTS: {e}")
        results["mac_tts"] = {"success": False, "error": str(e)}
    
    # Test Simple TTS
    print("\nTesting Simple TTS (pyttsx3):")
    try:
        output_path = project_root / "data" / "test_outputs" / "simple_tts_test.wav"
        
        start_time = time.time()
        simple_tts = SimpleTTS()
        load_time = time.time() - start_time
        print(f"TTS engine initialized in {load_time:.2f}s")
        
        start_time = time.time()
        simple_tts.synthesize(test_text, str(output_path))
        synthesis_time = time.time() - start_time
        print(f"Speech synthesized to {output_path}")
        print(f"Synthesis time: {synthesis_time:.4f}s")
        
        results["simple_tts"] = {
            "success": True,
            "output_path": str(output_path),
            "synthesis_time": synthesis_time
        }
    except Exception as e:
        print(f"Error with Simple TTS: {e}")
        results["simple_tts"] = {"success": False, "error": str(e)}
    
    return results

def test_llm_integration():
    """Test LLM integration."""
    print("\n===== Testing LLM Integration =====")
    
    results = {}
    
    # Test Mock LLM
    print("\nTesting Mock LLM:")
    try:
        start_time = time.time()
        mock_llm = get_response_generator(use_mock=True)
        load_time = time.time() - start_time
        print(f"Mock LLM initialized in {load_time:.2f}s")
        
        test_queries = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Tell me a joke about programming."
        ]
        
        mock_responses = []
        for query in test_queries:
            start_time = time.time()
            response = mock_llm.generate_response(query)
            response_time = time.time() - start_time
            mock_responses.append({
                "query": query,
                "response": response,
                "response_time": response_time
            })
            print(f"Query: '{query}'")
            print(f"Response: '{response}'")
            print(f"Response time: {response_time:.4f}s\n")
        
        results["mock_llm"] = {
            "success": True,
            "responses": mock_responses
        }
    except Exception as e:
        print(f"Error with Mock LLM: {e}")
        results["mock_llm"] = {"success": False, "error": str(e)}
    
    # Test Ollama LLM
    print("\nTesting Ollama LLM:")
    try:
        from src.assistant.ollama_integration import OllamaLLM
        
        start_time = time.time()
        ollama_llm = OllamaLLM(model_name="llama3.2:latest")
        load_time = time.time() - start_time
        print(f"Ollama LLM initialized in {load_time:.2f}s")
        
        # Use just one test query to avoid long processing times
        test_query = "What are the benefits of using speech recognition technology?"
        
        print(f"Testing with query: '{test_query}'")
        start_time = time.time()
        response = ollama_llm.generate_response(test_query)
        response_time = time.time() - start_time
        
        print(f"Query: '{test_query}'")
        print(f"Response: '{response}'")
        print(f"Response time: {response_time:.4f}s")
        
        results["ollama_llm"] = {
            "success": True,
            "response": response,
            "response_time": response_time
        }
    except Exception as e:
        print(f"Error with Ollama LLM: {e}")
        print(f"Make sure Ollama is installed and running with: ollama pull llama3 && ollama serve")
        results["ollama_llm"] = {"success": False, "error": str(e)}
    
    return results

def test_full_pipeline(test_audio_path, use_onnx=False):
    """Test the full speech assistant pipeline."""
    print("\n===== Testing Full Pipeline =====")
    
    # Choose STT model
    if use_onnx and ONNX_AVAILABLE:
        onnx_path = project_root / "models" / "stt" / "whisper_onnx" / "model.onnx"
        int8_path = project_root / "models" / "stt" / "whisper_onnx" / "model_int8.onnx"
        
        if int8_path.exists():
            stt_model = WhisperONNX(str(int8_path), "openai/whisper-tiny")
            print("Using ONNX INT8 model for speech recognition")
        elif onnx_path.exists():
            stt_model = WhisperONNX(str(onnx_path), "openai/whisper-tiny")
            print("Using ONNX FP32 model for speech recognition")
        else:
            print("ONNX model not found, falling back to PyTorch")
            stt_model = WhisperDirect("openai/whisper-tiny")
    else:
        stt_model = WhisperDirect("openai/whisper-tiny")
        print("Using PyTorch model for speech recognition")
    
    # Choose TTS model
    try:
        tts_model = MacTTS("Samantha")
        print("Using macOS TTS engine")
    except Exception as e:
        print(f"macOS TTS not available: {e}. Using simple TTS fallback.")
        tts_model = SimpleTTS()
    
    # Choose LLM model
    try:
        from src.assistant.ollama_integration import OllamaLLM
        llm_model = OllamaLLM(model_name="llama3")
        print("Using Ollama LLM for response generation")
    except Exception as e:
        print(f"Ollama LLM not available: {e}. Using mock LLM fallback.")
        mock_llm = get_response_generator(use_mock=True)
        llm_model = mock_llm
    
    # Initialize the assistant
    try:
        assistant = EnhancedSpeechAssistant(stt_model, tts_model, llm_model)
        print("Speech assistant initialized")
        
        # Process the query
        start_time = time.time()
        result = assistant.process_query(test_audio_path)
        total_time = time.time() - start_time
        
        print(f"\nFull Pipeline Results:")
        print(f"Input text: '{result['input_text']}'")
        print(f"Response text: '{result['response_text']}'")
        print(f"Response audio saved to: {result['response_audio']}")
        
        # Show timings
        timings = result.get("timings", {})
        print(f"\nPerformance:")
        print(f"- Transcription: {timings.get('transcription', 0):.4f}s")
        print(f"- Response generation: {timings.get('response_generation', 0):.4f}s")
        print(f"- Speech synthesis: {timings.get('speech_synthesis', 0):.4f}s")
        print(f"- Total time: {timings.get('total', total_time):.4f}s")
        
        return {
            "success": True,
            "result": result,
            "total_time": total_time
        }
    except Exception as e:
        print(f"Error in full pipeline: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def save_test_results(results, output_path=None):
    """Save test results to a JSON file."""
    if output_path is None:
        output_path = project_root / "data" / "test_outputs" / f"test_results_{int(time.time())}.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add timestamp
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test the speech assistant pipeline')
    parser.add_argument('--use-onnx', action='store_true', help='Use ONNX model for speech recognition')
    parser.add_argument('--audio-path', type=str, help='Path to test audio file (will create one if not provided)')
    parser.add_argument('--save-results', action='store_true', help='Save test results to a JSON file')
    args = parser.parse_args()
    
    # Create or use test audio
    if args.audio_path:
        test_audio_path = args.audio_path
        print(f"Using provided test audio: {test_audio_path}")
    else:
        test_audio_path = str(project_root / "data" / "audio_samples" / "test_sample.wav")
        create_test_audio(test_audio_path)
    
    # Run tests
    all_results = {}
    
    # Test Whisper
    all_results["whisper"] = test_whisper_models(test_audio_path)
    
    # Test TTS
    all_results["tts"] = test_tts_models()
    
    # Test LLM
    all_results["llm"] = test_llm_integration()
    
    # Test full pipeline
    all_results["pipeline"] = test_full_pipeline(test_audio_path, use_onnx=args.use_onnx)
    
    # Print summary
    print("\n===== Test Summary =====")
    
    whisper_pytorch = "✅ Passed" if all_results["whisper"]["pytorch"]["success"] else "❌ Failed"
    whisper_onnx = "✅ Passed" if all_results["whisper"].get("onnx", {}).get("success", False) else "❌ Failed"
    tts_mac = "✅ Passed" if all_results["tts"].get("mac_tts", {}).get("success", False) else "❌ Failed"
    tts_simple = "✅ Passed" if all_results["tts"].get("simple_tts", {}).get("success", False) else "❌ Failed"
    llm_mock = "✅ Passed" if all_results["llm"].get("mock_llm", {}).get("success", False) else "❌ Failed"
    llm_ollama = "✅ Passed" if all_results["llm"].get("ollama_llm", {}).get("success", False) else "❌ Failed"
    pipeline = "✅ Passed" if all_results["pipeline"]["success"] else "❌ Failed"
    
    print(f"Whisper PyTorch: {whisper_pytorch}")
    print(f"Whisper ONNX: {whisper_onnx}")
    print(f"TTS (macOS): {tts_mac}")
    print(f"TTS (Simple): {tts_simple}")
    print(f"LLM (Mock): {llm_mock}")
    print(f"LLM (Ollama): {llm_ollama}")
    print(f"Full Pipeline: {pipeline}")
    
    # Save results if requested
    if args.save_results:
        save_test_results(all_results)
    
    # Overall success/failure
    if all([
        all_results["whisper"]["pytorch"]["success"],
        all_results["tts"].get("mac_tts", {}).get("success", False) or all_results["tts"].get("simple_tts", {}).get("success", False),
        all_results["llm"].get("mock_llm", {}).get("success", False) or all_results["llm"].get("ollama_llm", {}).get("success", False),
        all_results["pipeline"]["success"]
    ]):
        print("\n🎉 All critical tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the API: python api/enhanced_app.py")
        print("2. Run the UI: streamlit run ui/enhanced_streamlit_app.py")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        sys.exit(1)