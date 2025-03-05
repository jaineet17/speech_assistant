import React from 'react';
import AssistantInterface from '../components/AssistantInterface';
import './Home.css';

const Home: React.FC = () => {
  return (
    <div className="home-page">
      <div className="hero-section">
        <h1>Enhanced Speech Assistant</h1>
        <p className="hero-description">
          Interact with our AI assistant using your voice. Ask questions, get information, 
          and receive spoken responses in real-time.
        </p>
      </div>
      
      <AssistantInterface />
      
      <div className="features-section">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Voice Recognition</h3>
            <p>Speak naturally and get accurate transcriptions of your questions.</p>
          </div>
          
          <div className="feature-card">
            <h3>AI-Powered Responses</h3>
            <p>Get intelligent answers powered by advanced language models.</p>
          </div>
          
          <div className="feature-card">
            <h3>Text-to-Speech</h3>
            <p>Listen to natural-sounding spoken responses from the assistant.</p>
          </div>
          
          <div className="feature-card">
            <h3>Performance Metrics</h3>
            <p>View detailed processing times for each step of the interaction.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 