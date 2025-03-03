"""Streamlit UI for the speech assistant."""

import streamlit as st
import requests
import soundfile as sf
import sounddevice as sd
import numpy as np
import time
import os
from io import BytesIO

# Configure API URL
API_URL = "http://localhost:5050"  # Update this line

# Set page title
st.set_page_config(page_title="Speech Assistant", page_icon="🎤")

st.title("🎤 Speech Assistant")
st.write("A local speech recognition and text-to-speech system")

# Sidebar with options
st.sidebar.title("Options")
record_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    with st.spinner(f"Recording for {duration} seconds..."):
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

# Create tabs
tab1, tab2, tab3 = st.tabs(["Assistant", "Speech-to-Text", "Text-to-Speech"])

with tab1:
    st.header("Voice Assistant")
    
    if st.button("Record Voice Query", key="record_assistant"):
        # Record audio
        audio_data, sample_rate = record_audio(duration=record_duration)
        
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
                st.subheader("Result:")
                st.write(f"You said: {result['input_text']}")
                st.write(f"Assistant: {result['response_text']}")
                
                # Get audio response
                audio_url = f"{API_URL}{result['audio_url']}"
                audio_response = requests.get(audio_url)
                
                if audio_response.status_code == 200:
                    st.audio(audio_response.content, format="audio/wav")
            else:
                st.error(f"Error: {response.text}")

with tab2:
    st.header("Speech-to-Text")
    
    if st.button("Record for Transcription", key="record_transcribe"):
        # Record audio
        audio_data, sample_rate = record_audio(duration=record_duration)
        
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
                st.subheader("Transcription:")
                st.write(result['transcription'])
            else:
                st.error(f"Error: {response.text}")
    
    # File upload option
    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Transcribe File"):
            files = {'audio': uploaded_file}
            with st.spinner("Transcribing..."):
                response = requests.post(f"{API_URL}/transcribe", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Transcription:")
                    st.write(result['transcription'])
                else:
                    st.error(f"Error: {response.text}")

with tab3:
    st.header("Text-to-Speech")
    
    text_input = st.text_area("Enter text to synthesize:", "Hello, this is a test of the speech synthesis system.")
    
    if st.button("Generate Speech"):
        with st.spinner("Generating speech..."):
            response = requests.post(
                f"{API_URL}/synthesize", 
                json={"text": text_input}
            )
            
            if response.status_code == 200:
                st.audio(response.content, format="audio/wav")
            else:
                st.error(f"Error: {response.text}")

st.sidebar.markdown("---")
st.sidebar.caption("Speech Assistant | Local Inference")