import React from 'react';
import AssistantInterface from './components/AssistantInterface';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Speech Assistant</h1>
      </header>
      <main className="main-content">
        <AssistantInterface />
      </main>
      <footer className="app-footer">
        <p>© 2023 Speech Assistant</p>
      </footer>
    </div>
  );
}

export default App; 