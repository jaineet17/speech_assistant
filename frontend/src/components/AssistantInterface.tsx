import React, { useState, useRef, useEffect } from 'react';
import './AssistantInterface.css';

interface TimingMetrics {
  stt_time: number;
  llm_time: number;
  tts_time: number;
  total_time: number;
}

interface ProcessResponse {
  text: string;
  input_text?: string;
  audio_url: string;
  timings: TimingMetrics;
}

// Define Message interface for conversation history
interface Message {
  id: string;
  text: string;
  isUser: boolean;
  audioUrl?: string;
  timestamp: Date;
}

const AssistantInterface: React.FC = () => {
  const [recording, setRecording] = useState<boolean>(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [timings, setTimings] = useState<TimingMetrics | null>(null);
  const [inputText, setInputText] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [audioPlayingId, setAudioPlayingId] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const conversationContainerRef = useRef<HTMLDivElement>(null);
  const textInputRef = useRef<HTMLTextAreaElement>(null);
  const audioRefs = useRef<{[key: string]: HTMLAudioElement | null}>({});
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (conversationContainerRef.current) {
      conversationContainerRef.current.scrollTop = conversationContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-resize text input as user types
  useEffect(() => {
    if (textInputRef.current) {
      textInputRef.current.style.height = 'auto';
      textInputRef.current.style.height = `${Math.min(textInputRef.current.scrollHeight, 120)}px`;
    }
  }, [inputText]);

  const startRecording = async () => {
    try {
      setError(null);
      audioChunksRef.current = [];
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Try to use higher quality audio if supported
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
        ? 'audio/webm;codecs=opus' 
        : 'audio/webm';
      
      const mediaRecorder = new MediaRecorder(stream, { 
        mimeType,
        audioBitsPerSecond: 128000 
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        setAudioBlob(audioBlob);
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to start recording. Please check your microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      
      // Release media stream
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  // Helper to add messages to conversation
  const addMessage = (text: string, isUser: boolean, audioUrl?: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser,
      audioUrl,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const processAudio = async () => {
    if (!audioBlob) {
      setError('No audio recorded.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      const response = await fetch('http://localhost:5050/process-audio', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      const data: ProcessResponse = await response.json();

      // Add user message with transcribed text (if available)
      if (data.input_text) {
        addMessage(data.input_text, true);
      }
      
      // Add assistant response with full URL
      const audioUrl = `http://localhost:5050${data.audio_url}`;
      addMessage(data.text, false, audioUrl);
      
      // Update timings
      setTimings(data.timings);
      
      // Clear audio after processing
      setAudioURL(null);
      setAudioBlob(null);
    } catch (err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const processText = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5050/process-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      const data: ProcessResponse = await response.json();
      
      // Add user message
      addMessage(inputText, true);
      
      // Add assistant response with full URL
      const audioUrl = `http://localhost:5050${data.audio_url}`;
      addMessage(data.text, false, audioUrl);
      
      // Update timings
      setTimings(data.timings);
      
      // Clear input text
      setInputText('');
    } catch (err) {
      console.error('Error processing text:', err);
      setError('Failed to process text. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      processText();
    }
  };

  const formatTime = (timestamp: Date): string => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleAudioPlay = (messageId: string) => {
    // Pause any currently playing audio
    if (audioPlayingId && audioPlayingId !== messageId && audioRefs.current[audioPlayingId]) {
      audioRefs.current[audioPlayingId]?.pause();
    }
    setAudioPlayingId(messageId);
  };

  const handleAudioEnded = () => {
    setAudioPlayingId(null);
  };

  return (
    <div className="assistant-interface">
      {/* Header with Navigation */}
      <header className="app-header">
        <h1 className="app-title">
          <span className="app-title-icon">🎙️</span>
          Speech Assistant
        </h1>
        <nav className="app-nav">
          <a href="#" className="nav-link">Home</a>
          <a href="#about" className="nav-link">About</a>
          <a href="#contribute" className="nav-link">Contribute</a>
        </nav>
      </header>
      
      <div className="assistant-controls">
        <p className="assistant-description">
          An intelligent voice assistant powered by Llama 2 and speech recognition technology.
          Interact using text or voice to get natural language responses.
        </p>
        
        {/* Conversation Container */}
        <div className="conversation-container" ref={conversationContainerRef}>
          {messages.length === 0 ? (
            <div className="empty-conversation">
              <span className="empty-conversation-icon">💬</span>
              <p>Start a conversation by speaking or typing a message</p>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.isUser ? 'user-message' : 'assistant-message'}`}>
                  <div className="message-bubble">
                    <p className="message-text">{message.text}</p>
                  </div>
                  {message.audioUrl && !message.isUser && (
                    <div className="message-audio">
                      <audio 
                        ref={(el) => audioRefs.current[message.id] = el}
                        controls 
                        src={message.audioUrl}
                        onPlay={() => handleAudioPlay(message.id)}
                        onEnded={handleAudioEnded}
                        onError={(e) => console.error("Audio playback error:", e)}
                      />
                      {audioPlayingId === message.id && (
                        <div className="audio-playing-indicator">Playing...</div>
                      )}
                    </div>
                  )}
                  <span className="message-time">{formatTime(message.timestamp)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Input Controls */}
        <div className="input-controls">
          <div className="record-controls">
            {recording ? (
              <button
                className="btn btn-icon stop-btn"
                onClick={stopRecording}
                disabled={loading}
              >
                <span className="icon">⏹️</span>
              </button>
            ) : (
              <button
                className="btn btn-icon record-btn"
                onClick={startRecording}
                disabled={loading}
              >
                <span className="icon">🎤</span>
              </button>
            )}
          </div>
          
          <div className="text-input-container">
            <textarea
              ref={textInputRef}
              className="text-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              disabled={loading || recording}
              rows={1}
            />
            <button
              className="btn btn-icon send-btn"
              onClick={processText}
              disabled={loading || recording || !inputText.trim()}
            >
              <span className="icon">📤</span>
            </button>
          </div>
        </div>
        
        {/* Audio Preview (when recording is completed) */}
        {audioURL && !recording && (
          <div className="audio-preview">
            <h3>Preview Recording</h3>
            <audio controls src={audioURL} />
            <button
              className="btn btn-primary process-btn"
              onClick={processAudio}
              disabled={loading}
            >
              {loading ? 'Processing...' : 'Process Recording'}
            </button>
          </div>
        )}
        
        {/* Error Message */}
        {error && <div className="error-message">{error}</div>}
        
        {/* Performance Metrics */}
        {timings && (
          <div className="performance-metrics">
            <h4>Performance Metrics</h4>
            <ul className="metrics-list">
              <li>
                <span className="metric-label">Speech-to-Text:</span>
                <span className="metric-value">{timings.stt_time.toFixed(2)}s</span>
              </li>
              <li>
                <span className="metric-label">LLM Processing:</span>
                <span className="metric-value">{timings.llm_time.toFixed(2)}s</span>
              </li>
              <li>
                <span className="metric-label">Text-to-Speech:</span>
                <span className="metric-value">{timings.tts_time.toFixed(2)}s</span>
              </li>
              <li>
                <span className="metric-label">Total Time:</span>
                <span className="metric-value">{timings.total_time.toFixed(2)}s</span>
              </li>
            </ul>
          </div>
        )}
      </div>
      
      {/* Footer with About and Contribute sections */}
      <footer className="app-footer">
        <div className="footer-sections">
          <div id="about" className="footer-section">
            <h3>
              <span className="section-icon">ℹ️</span>
              About
            </h3>
            <p>
              Speech Assistant is an open-source project that combines state-of-the-art speech recognition, 
              natural language processing, and text-to-speech technologies to create an interactive voice assistant.
            </p>
            <p>
              <strong>Key Features:</strong>
            </p>
            <ul>
              <li>Powered by Llama 2 7B language model</li>
              <li>Real-time speech recognition</li>
              <li>Natural-sounding text-to-speech</li>
              <li>Cross-platform compatibility</li>
              <li>Fully offline operation for privacy</li>
            </ul>
          </div>
          
          <div id="contribute" className="footer-section">
            <h3>
              <span className="section-icon">👥</span>
              Contribute
            </h3>
            <p>
              This project is open-source and welcomes contributions from developers, designers, 
              and enthusiasts. Here's how you can help:
            </p>
            <ul>
              <li>Report bugs and suggest features</li>
              <li>Improve documentation</li>
              <li>Submit pull requests</li>
              <li>Share the project with others</li>
              <li>Provide feedback on usability</li>
            </ul>
            <p>
              <a href="https://github.com/yourusername/speech-assistant" target="_blank" rel="noopener noreferrer">
                Visit our GitHub repository
              </a>
            </p>
          </div>
        </div>
        
        <div className="copyright">
          © {new Date().getFullYear()} Speech Assistant Project. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default AssistantInterface; 