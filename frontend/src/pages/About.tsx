import './About.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="container">
        <section className="about-section">
          <h1>About Speech Assistant</h1>
          
          <div className="card about-card">
            <h2>What is Speech Assistant?</h2>
            <p>
              Speech Assistant is a powerful application that combines speech recognition and 
              text-to-speech technologies to provide seamless voice interaction. It allows you to 
              record your voice, get accurate transcriptions, and even interact with an AI assistant.
            </p>
            
            <h2>Features</h2>
            <ul className="features-list">
              <li>
                <strong>Speech Recognition</strong> - Convert your spoken words to text with high accuracy
              </li>
              <li>
                <strong>Text-to-Speech</strong> - Convert text responses back to natural-sounding speech
              </li>
              <li>
                <strong>AI Assistant</strong> - Interact with an AI assistant that can answer questions and provide information
              </li>
              <li>
                <strong>Cross-Platform</strong> - Works on multiple platforms and devices
              </li>
            </ul>
            
            <h2>Technology</h2>
            <p>
              Speech Assistant is built using modern technologies:
            </p>
            <ul>
              <li>React and TypeScript for the frontend</li>
              <li>Flask for the backend API</li>
              <li>Whisper AI for speech recognition</li>
              <li>Advanced TTS engines for speech synthesis</li>
              <li>Large Language Models for intelligent responses</li>
            </ul>
            
            <h2>Open Source</h2>
            <p>
              This project is open source and available on 
              <a href="https://github.com/jaineet17/speech_assistant" target="_blank" rel="noopener noreferrer">
                GitHub
              </a>. 
              Contributions are welcome!
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About; 