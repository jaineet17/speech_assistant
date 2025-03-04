"""Enhanced Streamlit UI for the speech assistant."""

import streamlit as st
import requests
import soundfile as sf
import sounddevice as sd
import numpy as np
import time
import os
import json
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import base64

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {
        "api": {"host": "localhost", "port": 5050},
        "ui": {"title": "Speech Assistant", "theme": "light", "recording_duration": 5}
    }

# Configure API URL
API_HOST = config["api"].get("host", "localhost")
API_PORT = config["api"].get("port", 5050)
API_URL = f"http://{API_HOST}:{API_PORT}"

# UI configuration
UI_CONFIG = config.get("ui", {})
APP_TITLE = UI_CONFIG.get("title", "Speech Assistant")
APP_THEME = UI_CONFIG.get("theme", "light")
DEFAULT_RECORDING_DURATION = UI_CONFIG.get("recording_duration", 5)

# Set page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Speech Assistant\nAn efficient speech recognition and text-to-speech system with LLM integration."
    }
)

# Custom CSS
def local_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 1rem 2rem;
        }
        
        /* Cards for different sections */
        .stCard {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Response display */
        .response-card {
            background-color: #f0f7ff;
            border-radius: 15px;
            padding: 15px;
            margin-top: 20px;
            border-left: 5px solid #4285F4;
        }
        
        /* Query display */
        .query-card {
            background-color: #f0f2f5;
            border-radius: 15px;
            padding: 15px;
            margin-top: 20px;
            border-left: 5px solid #34A853;
        }
        
        /* Performance metrics */
        .metrics-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        
        .metric-card {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin-top: 10px;
            flex: 1;
            margin-right: 10px;
            min-width: 100px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4285F4;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: #666;
        }
        
        /* Icons */
        .icon-large {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Recording animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        
        .recording-icon {
            color: #f44336;
            animation: pulse 1s infinite;
        }
        
        /* Audio waveform */
        .audio-waveform {
            width: 100%;
            height: 80px;
            background-color: #f5f5f5;
            border-radius: 10px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 20px;
            padding: 10px 20px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Dark mode adjustments */
        .dark-mode .stCard {
            background-color: #1e1e1e;
        }
        
        .dark-mode .response-card {
            background-color: #2a3747;
            border-left: 5px solid #4285F4;
        }
        
        .dark-mode .query-card {
            background-color: #2a3a2a;
            border-left: 5px solid #34A853;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply dark mode if configured
if APP_THEME.lower() == "dark":
    st.markdown("""
    <style>
        body {
            color: #fff;
            background-color: #0e1117;
        }
        .dark-mode-marker {
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="dark-mode-marker"></div>', unsafe_allow_html=True)
else:
    st.markdown('<div></div>', unsafe_allow_html=True)

# Apply custom CSS
local_css()

# Functions

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    with st.spinner(f"🎙️ Recording for {duration} seconds..."):
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Display progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()
        progress_bar.empty()
        st.success("Recording finished!")
        
        return audio, sample_rate

def save_audio(audio_data, sample_rate=16000):
    """Save audio data to a temporary file."""
    temp_dir = "data/test_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = f"{temp_dir}/streamlit_{int(time.time())}.wav"
    sf.write(temp_path, audio_data, sample_rate)
    
    return temp_path

def get_audio_waveform(audio_data, n_points=100):
    """Generate a simplified audio waveform."""
    # Resampling to get n_points
    if len(audio_data) > n_points:
        indices = np.linspace(0, len(audio_data) - 1, n_points, dtype=int)
        waveform_data = audio_data[indices]
    else:
        waveform_data = audio_data
    
    # Normalize to [-1, 1]
    waveform_data = waveform_data / np.max(np.abs(waveform_data)) if np.max(np.abs(waveform_data)) > 0 else waveform_data
    
    return waveform_data

def plot_waveform(audio_data):
    """Plot audio waveform."""
    waveform = get_audio_waveform(audio_data)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(waveform, color='#4285F4')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, len(waveform))
    ax.axis('off')
    
    return fig

def display_metrics(timings):
    """Display performance metrics."""
    if not timings:
        return
    
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{timings.get('transcription', 0):.2f}s</div>
            <div class="metric-label">Transcription</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{timings.get('response_generation', 0):.2f}s</div>
            <div class="metric-label">Response Generation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{timings.get('speech_synthesis', 0):.2f}s</div>
            <div class="metric-label">Speech Synthesis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{timings.get('total', 0):.2f}s</div>
            <div class="metric-label">Total Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def check_api_status():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except:
        return False

# Main App

# Title and description
st.title(f"🎤 {APP_TITLE}")
st.markdown("An efficient speech recognition and text-to-speech system with LLM integration.")

# Check API status
api_status = check_api_status()
if not api_status:
    st.error(f"⚠️ Cannot connect to the API at {API_URL}. Make sure the server is running.")
    st.stop()

# Sidebar
st.sidebar.title("Options")
recording_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 15, DEFAULT_RECORDING_DURATION)

# Reset session button
if st.sidebar.button("Reset Conversation"):
    try:
        response = requests.post(f"{API_URL}/reset")
        if response.status_code == 200:
            st.sidebar.success("Conversation reset successfully!")
        else:
            st.sidebar.error(f"Error resetting conversation: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Error communicating with API: {str(e)}")

# Display information about the app
with st.sidebar.expander("About", expanded=False):
    st.markdown("""
        ## Speech Assistant
        
        This application uses:
        
        * **Whisper** for speech recognition
        * **LLM** for intelligent responses
        * **TTS** for realistic speech synthesis
        
        The system is optimized for edge deployment with ONNX runtime.
    """)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Assistant", "Speech-to-Text", "Text-to-Speech"])

with tab1:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Voice Assistant")
    st.markdown("Ask a question by recording your voice or uploading an audio file.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎙️ Record Voice Query", key="record_assistant_tab", use_container_width=True):
            # Record audio
            audio_data, sample_rate = record_audio(duration=recording_duration)
            
            # Display waveform
            fig = plot_waveform(audio_data)
            st.pyplot(fig)
            
            # Save audio
            audio_path = save_audio(audio_data, sample_rate)
            
            # Play back the recording
            st.audio(audio_path)
            
            # Send to API
            with st.spinner("Processing..."):
                files = {'audio': open(audio_path, 'rb')}
                response = requests.post(f"{API_URL}/assistant", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display query
                    st.markdown('<div class="query-card">', unsafe_allow_html=True)
                    st.markdown(f"**You said:** {result['input_text']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display response
                    st.markdown('<div class="response-card">', unsafe_allow_html=True)
                    st.markdown(f"**Assistant:** {result['response_text']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Get audio response
                    audio_url = f"{API_URL}{result['audio_url']}"
                    audio_response = requests.get(audio_url)
                    
                    if audio_response.status_code == 200:
                        st.audio(audio_response.content, format="audio/wav")
                    
                    # Show performance metrics
                    if "timings" in result:
                        st.markdown("### Performance Metrics")
                        display_metrics(result["timings"])
                else:
                    st.error(f"Error: {response.text}")
    
    with col2:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"], key="tab1_audio_upload")
        
        if uploaded_file is not None:
            # Display file details
            file_size_mb = uploaded_file.size / (1024 * 1024)
            est_duration = file_size_mb * 10  # Approximate: 10 seconds per MB
            
            # Calculate processing time estimates
            est_transcription_time = file_size_mb * 1.2  # ~1.2s per MB for transcription
            est_llm_time = 1.5  # Estimated LLM response time
            est_tts_time = 0.5  # Estimated TTS time
            est_total_time = est_transcription_time + est_llm_time + est_tts_time
            
            # Show file info in a card format with color coding by size
            card_color = "#e6f7ff"  # Light blue for small files
            if file_size_mb > 50:
                card_color = "#fff7e6"  # Light orange for medium files
            if file_size_mb > 100:
                card_color = "#fff1f0"  # Light red for large files
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; background-color: {card_color}; 
                        border-left: 5px solid #4285F4; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0;">File Information</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>Est. Duration:</strong> ~{int(est_duration/60)}m {int(est_duration%60)}s</p>
                <hr style="margin: 10px 0; border-color: rgba(0,0,0,0.1);">
                <p><strong>Processing Estimates:</strong></p>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Transcription:</span>
                    <span>{est_transcription_time:.1f}s</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Response Generation:</span>
                    <span>{est_llm_time:.1f}s</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Speech Synthesis:</span>
                    <span>{est_tts_time:.1f}s</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-weight: bold;">
                    <span>Total Processing:</span>
                    <span>{est_total_time:.1f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Play the audio with improved controls
            st.markdown("### Audio Preview")
            st.audio(uploaded_file)
            
            # Process button
            if st.button("Process File", key="tab1_process_file", use_container_width=True):
                # Create a progress bar container
                progress_placeholder = st.empty()
                results_placeholder = st.container()
                
                with st.spinner("Processing your audio..."):
                    # Show processing stages
                    progress_placeholder.markdown("""
                    <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                        <p>⏳ Uploading audio file...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Send to API
                    files = {'audio': uploaded_file}
                    
                    # Update progress
                    progress_placeholder.markdown("""
                    <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                        <p>✅ File uploaded</p>
                        <p>🔊 Transcribing audio...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make the API call
                    try:
                        response = requests.post(f"{API_URL}/assistant", files=files)
                        
                        # Update progress
                        progress_placeholder.markdown("""
                        <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                            <p>✅ File uploaded</p>
                            <p>✅ Audio transcribed</p>
                            <p>🤔 Generating response...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Final progress update
                            progress_placeholder.markdown("""
                            <div style="padding: 10px; border-radius: 5px; background-color: #f0fff4; 
                                       border: 1px solid #52c41a; margin-bottom: 10px;">
                                <p>✅ File uploaded</p>
                                <p>✅ Audio transcribed</p>
                                <p>✅ Response generated</p>
                                <p>✅ Speech synthesized</p>
                                <p style="font-weight: bold; color: #52c41a;">Processing complete! ✨</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with results_placeholder:
                                # Create tabbed result view
                                result_tabs = st.tabs(["Speech", "Text", "Performance"])
                                
                                with result_tabs[0]:
                                    # Get audio response
                                    audio_url = f"{API_URL}{result['audio_url']}"
                                    audio_response = requests.get(audio_url)
                                    
                                    if audio_response.status_code == 200:
                                        st.markdown("### Assistant Response")
                                        st.audio(audio_response.content, format="audio/wav")
                                    else:
                                        st.error("Failed to retrieve audio response")
                                
                                with result_tabs[1]:
                                    # Display query
                                    st.markdown('<div class="query-card">', unsafe_allow_html=True)
                                    st.markdown("### Your Question")
                                    st.markdown(f"{result['input_text']}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Display response
                                    st.markdown('<div class="response-card">', unsafe_allow_html=True)
                                    st.markdown("### Assistant's Answer")
                                    st.markdown(f"{result['response_text']}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with result_tabs[2]:
                                    # Show performance metrics
                                    if "timings" in result:
                                        st.markdown("### Performance Analysis")
                                        
                                        # Extract timings
                                        timings = result["timings"]
                                        transcription_time = timings.get('transcription', 0)
                                        response_time = timings.get('response_generation', 0)
                                        synthesis_time = timings.get('speech_synthesis', 0)
                                        total_time = timings.get('total', 0)
                                        
                                        # Calculate percentages for visualization
                                        if total_time > 0:
                                            trans_pct = (transcription_time / total_time) * 100
                                            resp_pct = (response_time / total_time) * 100
                                            synth_pct = (synthesis_time / total_time) * 100
                                        else:
                                            trans_pct = resp_pct = synth_pct = 33.3
                                        
                                        # Create a visual bar
                                        st.markdown(f"""
                                        <div style="width:100%; height:30px; background-color:#f0f0f0; border-radius:5px; margin-bottom:20px;">
                                          <div style="display:flex; height:100%;">
                                            <div style="width:{trans_pct}%; background-color:#4285F4; height:100%; display:flex; 
                                                     align-items:center; justify-content:center; color:white; font-size:0.8em;">
                                              {transcription_time:.1f}s
                                            </div>
                                            <div style="width:{resp_pct}%; background-color:#34A853; height:100%; display:flex; 
                                                     align-items:center; justify-content:center; color:white; font-size:0.8em;">
                                              {response_time:.1f}s
                                            </div>
                                            <div style="width:{synth_pct}%; background-color:#FBBC05; height:100%; display:flex; 
                                                     align-items:center; justify-content:center; color:white; font-size:0.8em;">
                                              {synthesis_time:.1f}s
                                            </div>
                                          </div>
                                        </div>
                                        <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
                                          <div style="display:flex; align-items:center;">
                                            <div style="width:10px; height:10px; background-color:#4285F4; margin-right:5px;"></div>
                                            <span>Transcription</span>
                                          </div>
                                          <div style="display:flex; align-items:center;">
                                            <div style="width:10px; height:10px; background-color:#34A853; margin-right:5px;"></div>
                                            <span>Response</span>
                                          </div>
                                          <div style="display:flex; align-items:center;">
                                            <div style="width:10px; height:10px; background-color:#FBBC05; margin-right:5px;"></div>
                                            <span>Synthesis</span>
                                          </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display metrics in a nice table
                                        st.markdown("#### Detailed Metrics")
                                        metrics_df = pd.DataFrame({
                                            "Process": ["Transcription", "Response Generation", "Speech Synthesis", "Total"],
                                            "Time (seconds)": [
                                                f"{transcription_time:.2f}s",
                                                f"{response_time:.2f}s", 
                                                f"{synthesis_time:.2f}s",
                                                f"{total_time:.2f}s"
                                            ],
                                            "Percentage": [
                                                f"{trans_pct:.1f}%",
                                                f"{resp_pct:.1f}%",
                                                f"{synth_pct:.1f}%",
                                                "100%"
                                            ]
                                        })
                                        st.table(metrics_df)
                                        
                                        # Compare with estimate
                                        ratio = total_time / est_total_time
                                        accuracy = 100 / ratio if ratio > 0 else 0
                                        
                                        st.markdown(f"""
                                        <div style="margin-top:15px; padding:10px; background-color:#f9f9f9; border-radius:5px;">
                                          <p><strong>Estimate Accuracy:</strong> {accuracy:.1f}%</p>
                                          <p>Estimated: {est_total_time:.1f}s vs Actual: {total_time:.1f}s</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # File processing rate
                                        if file_size_mb > 0:
                                            processing_rate = file_size_mb / total_time
                                            st.markdown(f"""
                                            <div style="margin-top:15px; padding:10px; background-color:#f9f9f9; border-radius:5px;">
                                              <p><strong>Processing Rate:</strong> {processing_rate:.2f} MB/second</p>
                                              <p>File size: {file_size_mb:.2f} MB processed in {total_time:.2f} seconds</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                        else:
                            # Error handling
                            progress_placeholder.markdown("""
                            <div style="padding: 10px; border-radius: 5px; background-color: #fff1f0; 
                                      border: 1px solid #f5222d; margin-bottom: 10px;">
                                <p>❌ Error occurred during processing</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with results_placeholder:
                                st.error(f"Error: {response.text}")
                    except Exception as e:
                        progress_placeholder.markdown("""
                        <div style="padding: 10px; border-radius: 5px; background-color: #fff1f0; 
                                  border: 1px solid #f5222d; margin-bottom: 10px;">
                            <p>❌ Connection error</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with results_placeholder:
                            st.error(f"Error connecting to API: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Speech-to-Text")
    st.markdown("Convert spoken audio to text using Whisper.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎙️ Record for Transcription", key="record_transcribe_tab", use_container_width=True):
            # Record audio
            audio_data, sample_rate = record_audio(duration=recording_duration)
            
            # Display waveform
            fig = plot_waveform(audio_data)
            st.pyplot(fig)
            
            # Save audio
            audio_path = save_audio(audio_data, sample_rate)
            
            # Play back the recording
            st.audio(audio_path)
            
            # Send to API
            with st.spinner("Transcribing..."):
                files = {'audio': open(audio_path, 'rb')}
                response = requests.post(f"{API_URL}/transcribe", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.markdown('<div class="response-card">', unsafe_allow_html=True)
                    st.markdown("### Transcription:")
                    st.markdown(result['transcription'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if 'inference_time' in result:
                        st.info(f"Transcription completed in {result['inference_time']:.2f} seconds")
                else:
                    st.error(f"Error: {response.text}")
    
    with col2:
        # File upload option
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"], key="tab2_audio_upload")
        
        if uploaded_file is not None:
            # Display file details
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>Est. processing time:</strong> {file_size_mb * 0.5:.1f} - {file_size_mb * 1.5:.1f} seconds</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Play the audio
            st.audio(uploaded_file)
            
            if st.button("Transcribe", key="tab2_transcribe_file", use_container_width=True):
                with st.spinner("Transcribing..."):
                    files = {'audio': uploaded_file}
                    response = requests.post(f"{API_URL}/transcribe", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown('<div class="response-card">', unsafe_allow_html=True)
                        st.markdown("### Transcription:")
                        st.markdown(result['transcription'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if 'inference_time' in result:
                            st.info(f"Transcription completed in {result['inference_time']:.2f} seconds")
                    else:
                        st.error(f"Error: {response.text}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Text-to-Speech")
    st.markdown("Convert text to natural-sounding speech.")
    
    text_input = st.text_area("Enter text to synthesize:", "Hello, this is a test of the speech synthesis system.", height=150)
    
    if st.button("🔊 Generate Speech", key="generate_speech_tab", use_container_width=True):
        with st.spinner("Generating speech..."):
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/synthesize", 
                json={"text": text_input}
            )
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                st.audio(response.content, format="audio/wav")
                st.success(f"Speech generated in {generation_time:.2f} seconds")
            else:
                st.error(f"Error: {response.text}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Speech Assistant | Efficient Edge AI Deployment")