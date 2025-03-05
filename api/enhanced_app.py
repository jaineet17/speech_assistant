"""
Enhanced Speech Assistant API with state-of-the-art inferencing
"""

import os
import sys
import time
import uuid
import logging
import asyncio
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import modules with error handling
try:
    from src.stt.speech_recognition import transcribe_audio
except ImportError as e:
    print(f"Error importing speech recognition module: {e}")
    # Define a fallback function
    def transcribe_audio(audio_file_path: str) -> str:
        return f"Error: Speech recognition module not available"

try:
    from src.tts.cross_platform_tts import text_to_speech, preload_tts_engine
except ImportError as e:
    print(f"Error importing TTS module: {e}")
    # Define a fallback function
    def text_to_speech(text: str, output_path: str) -> bool:
        with open(output_path, 'wb') as f:
            f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x10\x00data\x00\x00\x00\x00')
        return True
    
    def preload_tts_engine():
        pass

try:
    from src.llm.llm_assistant import get_llm_response, get_performance_metrics, EnhancedSingleton
except ImportError as e:
    print(f"Error importing LLM assistant module: {e}")
    # Define a fallback function
    def get_llm_response(query: str) -> str:
        return f"Error: LLM assistant module not available. Your query was: '{query}'"
    
    def get_performance_metrics() -> Dict[str, Any]:
        return {"error": "LLM metrics not available"}
    
    class EnhancedSingleton:
        @classmethod
        def get_instance(cls):
            return None

try:
    from src.utils.env_loader import load_env_vars
    # Load environment variables
    load_env_vars()
except ImportError as e:
    print(f"Error importing env_loader module: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enhanced_speech_assistant.api")

# Create directories for uploads and responses
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = TEMP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESPONSE_DIR = TEMP_DIR / "responses"
RESPONSE_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Speech Assistant API",
    description="API for the Enhanced Speech Assistant application with state-of-the-art inferencing",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# In-memory storage for transcription history
# In a production app, this would be a database
transcription_history: List[Dict[str, Any]] = []

# Performance metrics
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0,
    "total_response_time": 0,
    "requests_per_minute": {},
    "last_minute_requests": 0,
    "last_reset": datetime.now().isoformat()
}

def update_api_metrics(success: bool, response_time: float):
    """Update API performance metrics"""
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    api_metrics["total_requests"] += 1
    if success:
        api_metrics["successful_requests"] += 1
    else:
        api_metrics["failed_requests"] += 1
    
    api_metrics["total_response_time"] += response_time
    api_metrics["avg_response_time"] = api_metrics["total_response_time"] / api_metrics["total_requests"]
    
    # Track requests per minute
    if current_minute in api_metrics["requests_per_minute"]:
        api_metrics["requests_per_minute"][current_minute] += 1
    else:
        # Keep only the last 60 minutes
        if len(api_metrics["requests_per_minute"]) >= 60:
            oldest_key = min(api_metrics["requests_per_minute"].keys())
            api_metrics["requests_per_minute"].pop(oldest_key)
        api_metrics["requests_per_minute"][current_minute] = 1
    
    # Calculate requests in the last minute
    api_metrics["last_minute_requests"] = api_metrics["requests_per_minute"].get(current_minute, 0)

async def process_query_async(text: str) -> str:
    """
    Process a text query asynchronously and return a response using the LLM assistant.
    
    Args:
        text: The input text to process
        
    Returns:
        The response text
    """
    logger.info(f"Processing query: {text}")
    
    try:
        # Run the LLM in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            thread_pool,
            lambda: get_llm_response(text)
        )
        
        # Check if the response indicates an error with the LLM
        if "I'm sorry, but I'm having trouble accessing my knowledge" in response or "I encountered an error" in response:
            # If LLM failed, fall back to simple responses
            logger.warning("LLM response indicated an error, falling back to simple responses")
            return _generate_simple_response(text)
        return response
    except Exception as e:
        logger.error(f"Error using LLM assistant: {e}")
        # Fallback to simple responses if LLM fails
        return _generate_simple_response(text)

def _generate_simple_response(text: str) -> str:
    """Generate a simple response based on keywords in the input text."""
    if "hello" in text.lower() or "hi" in text.lower():
        return f"Hello! How can I help you today?"
    elif "how are you" in text.lower():
        return "I'm doing well, thank you for asking! How can I assist you?"
    elif "weather" in text.lower():
        return "I'm sorry, I don't have access to real-time weather data. Please check a weather service for accurate information."
    elif "time" in text.lower():
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    elif "date" in text.lower():
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
    elif "thank" in text.lower():
        return "You're welcome! Is there anything else I can help you with?"
    else:
        return f"I received your message: '{text}'. How can I assist you further?"

def save_to_history(input_text: str, response_text: str, timings: Dict[str, float]) -> None:
    """
    Save an interaction to the history.
    
    Args:
        input_text: The user's input text
        response_text: The assistant's response text
        timings: Timing information for the processing steps
    """
    try:
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "input_text": input_text,
            "response_text": response_text,
            "timestamp": datetime.now().isoformat(),
            "timings": timings
        }
        transcription_history.append(entry)
        
        # Keep only the last 100 entries
        if len(transcription_history) > 100:
            transcription_history.pop(0)
    except Exception as e:
        logger.error(f"Error saving to history: {e}", exc_info=True)

# Request models
class TextRequest(BaseModel):
    text: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Initializing Speech Assistant API components...")
    
    # Preload TTS engine in the background
    try:
        thread_pool.submit(preload_tts_engine)
        logger.info("TTS engine preloading initiated")
    except Exception as e:
        logger.error(f"Error preloading TTS engine: {e}")
    
    # Initialize LLM in the background
    try:
        thread_pool.submit(EnhancedSingleton.get_instance)
        logger.info("LLM initialization initiated")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
    
    logger.info("Speech Assistant API startup complete")

# Routes
@app.get("/")
async def root():
    return {"message": "Enhanced Speech Assistant API with state-of-the-art inferencing"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file and return the assistant's response.
    """
    start_time = time.time()
    success = False
    
    try:
        logger.info(f"Received audio file: {file.filename}")
        timings = {}
        
        # Save the uploaded file
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create unique filenames for this request
        request_id = str(uuid.uuid4())
        
        # Preserve the original file extension
        original_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".webm"
        if not original_ext:
            original_ext = ".webm"  # Default to webm if no extension
            
        input_file_path = os.path.join(temp_dir, f"input_{request_id}{original_ext}")
        output_file_path = os.path.join(temp_dir, f"output_{request_id}.wav")
        
        # Save the uploaded file
        try:
            with open(input_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info(f"Saved uploaded file to {input_file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Error saving uploaded file: {str(e)}"}
            )
        
        # Transcribe the audio
        transcribe_start = time.time()
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            transcription_result = await loop.run_in_executor(
                thread_pool,
                lambda: transcribe_audio(input_file_path)
            )
            
            # Check if the result is a dictionary (new format) or string (old format)
            if isinstance(transcription_result, dict):
                input_text = transcription_result.get("text", "")
                transcription_error = transcription_result.get("error")
            else:
                input_text = transcription_result
                transcription_error = None if input_text and not input_text.startswith(("Error:", "I couldn't understand")) else input_text
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            input_text = ""
            transcription_error = str(e)
            
        transcribe_end = time.time()
        timings["stt_time"] = round(transcribe_end - transcribe_start, 2)
        
        # Check if transcription failed
        if not input_text or transcription_error:
            logger.error(f"Transcription error: {transcription_error or 'Empty transcription'}")
            # Create a dummy response
            response_text = "I'm sorry, I couldn't understand the audio. Please try again or use the text input option."
            
            # Run TTS in thread pool
            loop = asyncio.get_event_loop()
            tts_success = await loop.run_in_executor(
                thread_pool,
                lambda: text_to_speech(response_text, output_file_path)
            )
            
            if not tts_success:
                logger.error("TTS failed for error response")
            
            # Return the error response with both input and response text
            response = {
                "input_text": input_text or "No transcription available",
                "text": response_text,
                "audio_url": f"/get-audio/{os.path.basename(output_file_path)}",
                "timings": {
                    "stt_time": timings.get("stt_time", 0),
                    "llm_time": 0,
                    "tts_time": 0,
                    "total_time": round(time.time() - start_time, 2)
                }
            }
            
            # Update API metrics
            update_api_metrics(success=False, response_time=time.time() - start_time)
            
            return response
        
        # Process the transcribed text
        process_start = time.time()
        response_text = await process_query_async(input_text)
        process_end = time.time()
        timings["llm_time"] = round(process_end - process_start, 2)
        
        # Convert response to speech
        tts_start = time.time()
        
        # Run TTS in thread pool
        loop = asyncio.get_event_loop()
        tts_success = await loop.run_in_executor(
            thread_pool,
            lambda: text_to_speech(response_text, output_file_path)
        )
        
        tts_end = time.time()
        timings["tts_time"] = round(tts_end - tts_start, 2)
        
        if not tts_success:
            logger.warning("TTS failed, but continuing with response")
        
        # Calculate total time
        total_time = time.time() - start_time
        timings["total_time"] = round(total_time, 2)
        
        # Save to history
        try:
            save_to_history(input_text, response_text, timings)
        except Exception as e:
            logger.error(f"Error saving to history: {e}", exc_info=True)
        
        # Return the response in the format expected by the frontend
        response = {
            "input_text": input_text,
            "text": response_text,
            "audio_url": f"/get-audio/{os.path.basename(output_file_path)}",
            "timings": timings
        }
        
        success = True
        return response
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing audio: {str(e)}"}
        )
    finally:
        # Update API metrics
        update_api_metrics(success=success, response_time=time.time() - start_time)

@app.post("/process-text")
async def process_text_endpoint(request: TextRequest):
    """
    Process text input and return the assistant's response.
    """
    start_time = time.time()
    success = False
    
    try:
        logger.info(f"Received text input: {request.text}")
        timings = {}
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create unique filename for this request
        request_id = str(uuid.uuid4())
        output_file_path = os.path.join(temp_dir, f"output_{request_id}.wav")
        
        # Process the text
        process_start = time.time()
        response_text = await process_query_async(request.text)
        process_end = time.time()
        timings["llm_time"] = round(process_end - process_start, 2)
        
        # Convert response to speech
        tts_start = time.time()
        
        # Run TTS in thread pool
        loop = asyncio.get_event_loop()
        tts_success = await loop.run_in_executor(
            thread_pool,
            lambda: text_to_speech(response_text, output_file_path)
        )
        
        tts_end = time.time()
        timings["tts_time"] = round(tts_end - tts_start, 2)
        
        if not tts_success:
            logger.warning("TTS failed, but continuing with response")
        
        # Calculate total time
        total_time = time.time() - start_time
        timings["total_time"] = round(total_time, 2)
        
        # Set STT time to 0 for text input
        timings["stt_time"] = 0
        
        # Save to history
        try:
            save_to_history(request.text, response_text, timings)
        except Exception as e:
            logger.error(f"Error saving to history: {e}", exc_info=True)
        
        # Return the response in the format expected by the frontend
        response = {
            "input_text": request.text,
            "text": response_text,
            "audio_url": f"/get-audio/{os.path.basename(output_file_path)}",
            "timings": timings
        }
        
        success = True
        return response
    except Exception as e:
        logger.error(f"Error processing text: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing text: {str(e)}"}
        )
    finally:
        # Update API metrics
        update_api_metrics(success=success, response_time=time.time() - start_time)

@app.get("/get-audio/{filename}")
async def get_audio(filename: str):
    """
    Get audio file by filename.
    """
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return JSONResponse(
                status_code=404,
                content={"error": "Audio file not found"}
            )
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error retrieving audio file: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving audio file: {str(e)}"}
        )

@app.get("/history")
async def get_history():
    """
    Get transcription history.
    """
    return {"history": transcription_history}

@app.get("/metrics")
async def get_metrics():
    """
    Get performance metrics for the system.
    """
    try:
        # Get LLM metrics
        llm_metrics = get_performance_metrics()
        
        # Combine with API metrics
        metrics = {
            "api": api_metrics,
            "llm": llm_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving metrics: {str(e)}"}
        )

@app.get("/status")
async def get_status():
    """
    Get system status with detailed component information.
    """
    try:
        # Check LLM status
        llm_instance = EnhancedSingleton.get_instance()
        llm_status = "ok" if llm_instance and llm_instance.initialized else "error"
        
        # Get basic metrics
        llm_metrics = get_performance_metrics()
        avg_tokens_per_sec = llm_metrics.get("avg_tokens_per_second", 0)
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": {
                    "status": "ok",
                    "requests_handled": api_metrics["total_requests"],
                    "avg_response_time": f"{api_metrics['avg_response_time']:.2f}s"
                },
                "llm": {
                    "status": llm_status,
                    "model": "Llama-2-7B-Chat-GGUF",
                    "performance": f"{avg_tokens_per_sec:.1f} tokens/sec",
                    "cache_hits": llm_metrics.get("cache_hits", 0)
                },
                "stt": {
                    "status": "ok",
                    "engine": "Google Speech Recognition"
                },
                "tts": {
                    "status": "ok",
                    "engine": "System TTS"
                }
            },
            "version": "1.1.0"
        }
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}", exc_info=True)
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "1.1.0"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enhanced_app:app", host="0.0.0.0", port=5050, reload=True)