"""
Speech recognition module for transcribing audio to text.
"""

import os
import logging
import tempfile
from typing import Optional
import speech_recognition as sr
import wave
import struct
import io
import math
import subprocess
import numpy as np
from pydub import AudioSegment
import time

# Configure logging
logger = logging.getLogger("enhanced_speech_assistant.speech_recognition")

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = 10  # seconds

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file to text."""
        start_time = time.time()
        
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            logger.error(f"Audio file does not exist or is empty: {audio_file_path}")
            return {"text": "", "error": "Audio file is empty or does not exist"}
        
        # Check if it's a webm file and handle it specially
        if audio_file_path.lower().endswith('.webm'):
            return self._handle_webm_file(audio_file_path)
        
        # For other files, try to create a valid WAV file first
        valid_wav_path = self._create_valid_wav(audio_file_path)
        if not valid_wav_path:
            logger.error(f"Failed to create a valid WAV file from {audio_file_path}")
            return {"text": "", "error": "Failed to process audio file"}
        
        result = self._try_transcribe(valid_wav_path)
        
        # Clean up temporary file if it's different from the original
        if valid_wav_path != audio_file_path and os.path.exists(valid_wav_path):
            try:
                os.remove(valid_wav_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {valid_wav_path}: {e}")
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result

    def _handle_webm_file(self, webm_file_path):
        """Special handling for webm files which can be problematic."""
        logger.info(f"Handling webm file: {webm_file_path}")
        
        # First try using ffmpeg directly to convert to wav
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Use ffmpeg to convert webm to wav with specific parameters
            cmd = [
                'ffmpeg', 
                '-i', webm_file_path, 
                '-acodec', 'pcm_s16le',  # Use PCM 16-bit encoding
                '-ar', '16000',          # Set sample rate to 16kHz
                '-ac', '1',              # Convert to mono
                '-y',                    # Overwrite output file if it exists
                temp_wav_path
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"ffmpeg conversion failed: {stderr.decode()}")
                # If ffmpeg fails, try pydub as fallback
                return self._try_pydub_conversion(webm_file_path)
            
            # Check if the converted file exists and has content
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                result = self._try_transcribe(temp_wav_path)
                
                # Clean up
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_wav_path}: {e}")
                
                return result
            else:
                logger.warning("ffmpeg produced an empty or non-existent file")
                return self._try_pydub_conversion(webm_file_path)
                
        except Exception as e:
            logger.error(f"Error during webm handling: {e}")
            return self._try_pydub_conversion(webm_file_path)
    
    def _try_pydub_conversion(self, audio_file_path):
        """Try to convert audio using pydub as a fallback."""
        logger.info(f"Trying pydub conversion for: {audio_file_path}")
        
        try:
            # Try to load with pydub
            audio = AudioSegment.from_file(audio_file_path)
            
            # Export as wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
            audio.export(temp_wav_path, format="wav")
            
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                result = self._try_transcribe(temp_wav_path)
                
                # Clean up
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_wav_path}: {e}")
                
                return result
            else:
                logger.warning("pydub produced an empty or non-existent file")
                return {"text": "", "error": "Failed to convert audio file"}
                
        except Exception as e:
            logger.error(f"Error during pydub conversion: {e}")
            return {"text": "", "error": f"Audio conversion failed: {str(e)}"}

    def _create_valid_wav(self, audio_file_path):
        """Create a valid WAV file from the input audio file."""
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            logger.error(f"Audio file does not exist or is empty: {audio_file_path}")
            return None
        
        # Check if the file already has a proper audio header
        try:
            with open(audio_file_path, 'rb') as f:
                header = f.read(12)  # Read the first 12 bytes to check for RIFF or FORM header
                
            # If it's already a WAV file with RIFF header, use it directly
            if header.startswith(b'RIFF') and b'WAVE' in header:
                logger.info(f"File is already a valid WAV: {audio_file_path}")
                return audio_file_path
                
            # If it's an AIFF file with FORM header, convert it
            if header.startswith(b'FORM') and b'AIFF' in header:
                logger.info(f"File is an AIFF file, converting: {audio_file_path}")
                # Will be converted below
            else:
                logger.warning(f"File doesn't have a recognized audio header: {audio_file_path}")
                # Will attempt conversion below
                
        except Exception as e:
            logger.error(f"Error checking file header: {e}")
            # Will attempt conversion below
        
        # Try to convert using pydub
        try:
            logger.info(f"Converting audio file using pydub: {audio_file_path}")
            audio = AudioSegment.from_file(audio_file_path)
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
            audio.export(temp_wav_path, format="wav")
            
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                logger.info(f"Successfully converted to WAV using pydub: {temp_wav_path}")
                return temp_wav_path
            else:
                logger.warning("pydub produced an empty or non-existent file")
                # Will try ffmpeg below
                
        except Exception as e:
            logger.warning(f"pydub conversion failed: {e}")
            # Will try ffmpeg below
        
        # Try to convert using ffmpeg
        try:
            logger.info(f"Converting audio file using ffmpeg: {audio_file_path}")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            cmd = [
                'ffmpeg', 
                '-i', audio_file_path, 
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                temp_wav_path
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"ffmpeg conversion failed: {stderr.decode()}")
                # Will create a simple WAV file below
            else:
                if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                    logger.info(f"Successfully converted to WAV using ffmpeg: {temp_wav_path}")
                    return temp_wav_path
                else:
                    logger.warning("ffmpeg produced an empty or non-existent file")
                    # Will create a simple WAV file below
                    
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}")
            # Will create a simple WAV file below
        
        # As a last resort, create a simple WAV file with a beep sound
        return self._create_simple_wav()
    
    def _create_simple_wav(self):
        """Create a simple WAV file with a beep sound as a last resort."""
        logger.info("Creating a simple WAV file with a beep sound")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        # Create a simple WAV file with a beep sound
        sample_rate = 16000
        duration = 1  # seconds
        frequency = 440  # Hz (A4 note)
        
        # Generate a sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (32767 * 0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        # Write to WAV file
        with wave.open(temp_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        logger.info(f"Created simple WAV file: {temp_wav_path}")
        return temp_wav_path

    def _try_transcribe(self, audio_file_path):
        """Try to transcribe the audio file using different recognition engines."""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                
            # First try Google's recognizer
            try:
                text = self.recognizer.recognize_google(audio_data)
                return {"text": text, "error": None, "engine": "google"}
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Could not request results from Google Speech Recognition service: {e}")
            
            # Try Sphinx as a fallback
            try:
                text = self.recognizer.recognize_sphinx(audio_data)
                return {"text": text, "error": None, "engine": "sphinx"}
            except sr.UnknownValueError:
                logger.warning("Sphinx could not understand audio")
                return {"text": "", "error": "Speech not recognized", "engine": "failed"}
            except sr.RequestError as e:
                logger.warning(f"Sphinx error: {e}")
                return {"text": "", "error": f"Recognition error: {str(e)}", "engine": "failed"}
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {"text": "", "error": f"Transcription error: {str(e)}", "engine": "failed"}

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe audio file to text using Google Speech Recognition.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text or error message
    """
    logger.info(f"Transcribing audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        error_msg = f"Audio file not found: {audio_file_path}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # First try with the original file
    result = _try_transcribe(audio_file_path)
    
    # If that fails, try to create a valid WAV file and transcribe that
    if result.startswith("Error:"):
        logger.warning(f"Failed to transcribe original file, attempting to create valid WAV")
        valid_wav_path = _create_valid_wav(audio_file_path)
        
        if valid_wav_path:
            logger.info(f"Created valid WAV file at {valid_wav_path}, attempting transcription")
            result = _try_transcribe(valid_wav_path)
            
            # Clean up the temporary file
            try:
                os.remove(valid_wav_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary WAV file: {e}")
        else:
            logger.error("Failed to create valid WAV file")
            # Create a silent WAV and return a friendly error
            silent_wav = _create_silent_wav()
            if silent_wav:
                result = "I couldn't hear anything. Please try speaking more clearly or use the text input option."
                try:
                    os.remove(silent_wav)
                except Exception as e:
                    logger.warning(f"Failed to remove silent WAV file: {e}")
    
    return result

def _try_transcribe(audio_file_path: str) -> str:
    """
    Attempt to transcribe an audio file.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text or error message
    """
    try:
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Check if the file exists and has content
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return "Error: Audio file not found"
        
        if os.path.getsize(audio_file_path) == 0:
            logger.error(f"Audio file is empty: {audio_file_path}")
            return "Error: Audio file is empty"
        
        # Try to load the audio file
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise and record
                try:
                    audio_data = r.record(source)
                except Exception as e:
                    logger.error(f"Error recording from file: {e}", exc_info=True)
                    return f"Error: Failed to process audio file: {str(e)}"
        except ValueError as e:
            # If the file can't be read as a supported format, try to convert it
            logger.warning(f"Could not read audio file as supported format: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return f"Error: Failed to load audio file: {str(e)}"
        
        # Try Google's speech recognition service
        try:
            logger.info("Attempting transcription with Google")
            text = r.recognize_google(audio_data, language="en-US")
            if text:
                logger.info(f"Successfully transcribed with Google: {text}")
                return text
        except sr.UnknownValueError:
            logger.warning("Google could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Google request error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with Google: {e}", exc_info=True)
        
        # If Google fails, try to use the local recognizer if available
        try:
            # Only import if needed to avoid startup errors
            import speech_recognition as sr_local
            
            logger.info("Attempting transcription with local recognizer")
            # Use a more lenient energy threshold for better recognition
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            
            # Try to recognize with the local recognizer
            text = r.recognize_sphinx(audio_data) if hasattr(r, 'recognize_sphinx') else None
            if text:
                logger.info(f"Successfully transcribed with local recognizer: {text}")
                return text
        except ImportError:
            logger.warning("Local speech recognition not available")
        except sr.UnknownValueError:
            logger.warning("Local recognizer could not understand audio")
        except Exception as e:
            logger.error(f"Error with local recognizer: {e}", exc_info=True)
        
        # If all services failed
        return "I couldn't understand what was said. Please try again or use the text input option."
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}", exc_info=True)
        return f"Error: {str(e)}"

def _create_valid_wav(input_path: str) -> Optional[str]:
    """
    Attempt to create a valid WAV file from the input audio file.
    
    Args:
        input_path: Path to the input audio file
        
    Returns:
        Path to the valid WAV file or None if failed
    """
    try:
        # Create a temporary file for the output
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Check if the file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            logger.error(f"Input file does not exist or is empty: {input_path}")
            return _create_simple_wav_file(output_path)
        
        # Check the file extension
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Try to read the file header to determine the type
        with open(input_path, 'rb') as f:
            file_header = f.read(32)  # Read first 32 bytes to check header
        
        # Check for webm signature (1A 45 DF A3)
        is_webm = False
        if file_ext == '.webm' or (len(file_header) >= 4 and file_header[0:4] == b'\x1a\x45\xdf\xa3'):
            logger.info("Detected webm file by extension or signature, using special handling")
            is_webm = True
            return _handle_webm_file(input_path, output_path)
        
        # Check for MP3 signature (ID3 or MPEG frame sync)
        is_mp3 = False
        if file_ext == '.mp3' or (len(file_header) >= 3 and file_header[0:3] == b'ID3') or \
           (len(file_header) >= 2 and file_header[0] == 0xFF and (file_header[1] & 0xE0) == 0xE0):
            logger.info("Detected MP3 file, using ffmpeg for conversion")
            try:
                import subprocess
                result = subprocess.run(
                    ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-y", output_path],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully converted MP3 using ffmpeg to {output_path}")
                    return output_path
            except Exception as e:
                logger.warning(f"MP3 conversion failed: {e}")
        
        # If it doesn't have a proper WAV header, create a WAV header
        if not (file_header.startswith(b'RIFF') or file_header.startswith(b'FORM')):
            logger.warning("File doesn't have a proper audio header, treating as raw PCM")
            
            # If we've already tried special handling for known formats, don't try raw PCM conversion
            if is_webm or is_mp3:
                return _try_pydub_conversion(input_path, output_path)
            
            # Create a new file with WAV header + original content
            with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                # Write WAV header
                sample_rate = 16000
                channels = 1
                bits_per_sample = 16
                
                # Get file size
                f_in.seek(0, 2)  # Seek to end
                file_size = f_in.tell()
                f_in.seek(0)  # Back to start
                
                # Calculate sizes for header
                data_size = file_size
                header_size = 36 + data_size
                
                # Write RIFF header
                f_out.write(b'RIFF')
                f_out.write(header_size.to_bytes(4, 'little'))
                f_out.write(b'WAVE')
                
                # Write format chunk
                f_out.write(b'fmt ')
                f_out.write((16).to_bytes(4, 'little'))  # Chunk size
                f_out.write((1).to_bytes(2, 'little'))  # Audio format (PCM)
                f_out.write(channels.to_bytes(2, 'little'))
                f_out.write(sample_rate.to_bytes(4, 'little'))
                f_out.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))  # Byte rate
                f_out.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))  # Block align
                f_out.write(bits_per_sample.to_bytes(2, 'little'))
                
                # Write data chunk
                f_out.write(b'data')
                f_out.write(data_size.to_bytes(4, 'little'))
                
                # Copy the original file content
                f_out.write(f_in.read())
            
            logger.info(f"Created WAV file with header at {output_path}")
            return output_path
        
        # If it has a proper header, use pydub to convert
        return _try_pydub_conversion(input_path, output_path)
    except Exception as e:
        logger.error(f"Error creating valid WAV file: {e}", exc_info=True)
        return None

def _handle_webm_file(input_path: str, output_path: str) -> Optional[str]:
    """
    Special handling for webm files.
    
    Args:
        input_path: Path to the input webm file
        output_path: Path to save the output WAV file
        
    Returns:
        Path to the valid WAV file or None if failed
    """
    try:
        logger.info(f"Handling webm file: {input_path}")
        
        # Try using ffmpeg directly first (more reliable for webm)
        try:
            logger.info("Attempting to convert webm using ffmpeg")
            import subprocess
            import shutil
            
            # Check if ffmpeg is available
            if shutil.which("ffmpeg"):
                logger.info("ffmpeg is available in PATH")
            else:
                logger.warning("ffmpeg not found in PATH, conversion may fail")
            
            # Run ffmpeg with verbose logging
            cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-y", output_path]
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully converted webm using ffmpeg to {output_path}")
                return output_path
            else:
                logger.warning(f"ffmpeg conversion failed with return code {result.returncode}")
                logger.warning(f"ffmpeg stderr: {result.stderr}")
                logger.warning(f"ffmpeg stdout: {result.stdout}")
        except Exception as e:
            logger.warning(f"ffmpeg webm conversion failed: {e}", exc_info=True)
        
        # Try using pydub as fallback
        try:
            import pydub
            logger.info("Attempting to convert webm using pydub")
            
            # Load the webm file
            audio = pydub.AudioSegment.from_file(input_path, format="webm")
            
            # Set parameters for better speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully converted webm using pydub to {output_path}")
                return output_path
            else:
                logger.warning("pydub conversion produced empty or non-existent file")
        except ImportError:
            logger.warning("pydub not available for webm conversion")
        except Exception as e:
            logger.warning(f"pydub webm conversion failed: {e}", exc_info=True)
        
        # If all conversion methods failed, create a simple WAV file
        logger.warning("All webm conversion approaches failed, creating a simple WAV file")
        return _create_simple_wav_file(output_path)
    except Exception as e:
        logger.error(f"Error handling webm file: {e}", exc_info=True)
        return _create_simple_wav_file(output_path)

def _create_simple_wav_file(output_path: str) -> Optional[str]:
    """
    Create a simple silent WAV file.
    Returns the path to the created file or None if failed.
    """
    try:
        # Parameters
        sample_rate = 16000
        duration = 1  # seconds
        
        # Create WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Generate silence
            silence = bytearray(int(duration * sample_rate) * 2)  # 2 bytes per sample
            wav_file.writeframes(silence)
        
        logger.info(f"Created silent WAV file at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating silent WAV file: {e}", exc_info=True)
        return None

def _create_silent_wav() -> Optional[str]:
    """
    Create a silent WAV file.
    
    Returns:
        Path to the silent WAV file or None if failed
    """
    try:
        # Create a temporary file
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Parameters
        sample_rate = 16000
        duration = 1  # seconds
        
        # Create WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Generate silence
            silence = bytearray(int(sample_rate * duration))
            wav_file.writeframes(silence)
        
        logger.info(f"Created silent WAV file at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating silent WAV file: {e}", exc_info=True)
        return None

def _try_pydub_conversion(input_path: str, output_path: str) -> Optional[str]:
    """
    Try to convert audio using pydub or ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the output WAV file
        
    Returns:
        Path to the valid WAV file or None if failed
    """
    try:
        # Try to use pydub for conversion
        try:
            import pydub
            logger.info("Attempting to convert audio using pydub")
            
            # Load the audio file (pydub can handle various formats)
            audio = pydub.AudioSegment.from_file(input_path)
            
            # Set parameters for better speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully converted audio using pydub to {output_path}")
                return output_path
        except ImportError:
            logger.warning("pydub not available, trying next approach")
        except Exception as e:
            logger.warning(f"pydub conversion failed: {e}")
        
        # Try using ffmpeg directly
        try:
            logger.info("Attempting to convert audio using ffmpeg")
            import subprocess
            
            # Run ffmpeg to convert the file
            result = subprocess.run(
                ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-y", output_path],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully converted audio using ffmpeg to {output_path}")
                return output_path
            else:
                logger.warning(f"ffmpeg conversion failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}")
        
        # If all conversion methods failed, create a simple WAV file
        logger.warning("All conversion approaches failed, creating a simple WAV file")
        return _create_simple_wav_file(output_path)
    except Exception as e:
        logger.error(f"Error in audio conversion: {e}", exc_info=True)
        return _create_simple_wav_file(output_path) 