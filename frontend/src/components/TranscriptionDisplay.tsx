import { useState } from 'react';
import './TranscriptionDisplay.css';

interface Transcription {
  id: string;
  text: string;
  timestamp: string;
}

interface TranscriptionDisplayProps {
  transcriptions: Transcription[];
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ transcriptions }) => {
  const [selectedTranscription, setSelectedTranscription] = useState<string | null>(null);

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text)
      .then(() => {
        alert('Transcription copied to clipboard!');
      })
      .catch(err => {
        console.error('Failed to copy text: ', err);
      });
  };

  const toggleTranscription = (id: string) => {
    if (selectedTranscription === id) {
      setSelectedTranscription(null);
    } else {
      setSelectedTranscription(id);
    }
  };

  if (transcriptions.length === 0) {
    return (
      <div className="no-transcriptions">
        <p>No transcriptions yet. Record and transcribe audio to see results here.</p>
      </div>
    );
  }

  return (
    <div className="transcription-list">
      {transcriptions.map((transcription) => (
        <div 
          key={transcription.id} 
          className={`transcription-item ${selectedTranscription === transcription.id ? 'expanded' : ''}`}
        >
          <div 
            className="transcription-header"
            onClick={() => toggleTranscription(transcription.id)}
          >
            <div className="transcription-preview">
              {transcription.text.length > 100 
                ? `${transcription.text.substring(0, 100)}...` 
                : transcription.text}
            </div>
            <div className="transcription-meta">
              <span className="transcription-date">{formatDate(transcription.timestamp)}</span>
              <button 
                className="expand-button"
                aria-label={selectedTranscription === transcription.id ? 'Collapse' : 'Expand'}
              >
                {selectedTranscription === transcription.id ? '−' : '+'}
              </button>
            </div>
          </div>
          
          {selectedTranscription === transcription.id && (
            <div className="transcription-content">
              <p>{transcription.text}</p>
              <div className="transcription-actions">
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => handleCopyText(transcription.text)}
                >
                  Copy Text
                </button>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default TranscriptionDisplay; 