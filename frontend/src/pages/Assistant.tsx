import React from 'react';
import AssistantInterface from '../components/AssistantInterface';
import './Assistant.css';

const Assistant: React.FC = () => {
  return (
    <div className="assistant-page">
      <div className="assistant-header">
        <h1>AI Voice Assistant</h1>
        <p className="assistant-intro">
          Interact with our advanced AI assistant using your voice. Ask questions, 
          get information, and receive spoken responses powered by state-of-the-art 
          speech recognition and natural language processing.
        </p>
      </div>
      
      <AssistantInterface />
      
      <div className="assistant-instructions">
        <h2>How to Use</h2>
        <ol className="instructions-list">
          <li>Click the <strong>microphone</strong> button and speak your question or request.</li>
          <li>Click <strong>stop</strong> when you're finished speaking.</li>
          <li>Review your recording and click <strong>Process Recording</strong> to process your question.</li>
          <li>Alternatively, type your message and press <strong>send</strong> or hit Enter.</li>
          <li>Wait for the AI to process your request and provide a response.</li>
          <li>Listen to the spoken response or read the text response provided.</li>
        </ol>
        <p className="tips">
          <strong>Tips:</strong> Speak clearly and try to minimize background noise for best results. 
          Complex questions may take longer to process.
        </p>
      </div>
    </div>
  );
};

export default Assistant; 