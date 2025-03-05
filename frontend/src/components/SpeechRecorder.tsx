import { useState, useRef, useEffect } from 'react';
import './SpeechRecorder.css';

interface SpeechRecorderProps {
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: (blob: Blob) => void;
}

const SpeechRecorder: React.FC<SpeechRecorderProps> = ({
  isRecording,
  onStartRecording,
  onStopRecording,
}) => {
  const [recordingTime, setRecordingTime] = useState(0);
  const [permissionDenied, setPermissionDenied] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (isRecording) {
      startRecording();
    } else if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      stopRecording();
    }

    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    chunksRef.current = [];
    setRecordingTime(0);
    setPermissionDenied(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        onStopRecording(audioBlob);
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        
        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };
      
      mediaRecorder.start();
      
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setPermissionDenied(true);
      onStopRecording(new Blob());
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleRecordButtonClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      onStartRecording();
    }
  };

  return (
    <div className="speech-recorder">
      <div className="recorder-controls">
        <button 
          className={`record-button ${isRecording ? 'recording' : ''}`}
          onClick={handleRecordButtonClick}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isRecording ? (
            <span className="stop-icon"></span>
          ) : (
            <span className="mic-icon"></span>
          )}
        </button>
        
        <div className="recorder-status">
          {isRecording ? (
            <>
              <div className="recording-indicator">
                <span className="recording-dot"></span>
                Recording
              </div>
              <div className="recording-time">{formatTime(recordingTime)}</div>
            </>
          ) : (
            <div className="recorder-prompt">Click to start recording</div>
          )}
        </div>
      </div>
      
      {permissionDenied && (
        <div className="permission-error">
          Microphone access denied. Please allow microphone access to use this feature.
        </div>
      )}
    </div>
  );
};

export default SpeechRecorder; 