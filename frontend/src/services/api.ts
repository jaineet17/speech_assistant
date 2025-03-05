/**
 * API service for communicating with the backend
 */

import { AssistantResponse } from '../types';

const API_BASE_URL = 'http://localhost:5050';

/**
 * Transcribe audio using the backend API
 * @param audioBlob - The audio blob to transcribe
 * @returns Promise with the transcription result
 */
export const transcribeAudio = async (audioBlob: Blob): Promise<{ transcription: string; inference_time: number }> => {
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await fetch(`${API_BASE_URL}/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Server responded with status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error transcribing audio:', error);
    throw error;
  }
};

/**
 * Synthesize speech from text
 * @param text - The text to convert to speech
 * @returns Promise with the audio blob
 */
export const synthesizeSpeech = async (text: string): Promise<Blob> => {
  try {
    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch(`${API_BASE_URL}/synthesize`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Server responded with status: ${response.status}`);
    }

    return await response.blob();
  } catch (error) {
    console.error('Error synthesizing speech:', error);
    throw error;
  }
};

/**
 * Process audio with the assistant
 * @param audioBlob The audio blob to process
 * @returns The assistant's response
 */
export const processAudioWithAssistant = async (audioBlob: Blob): Promise<AssistantResponse> => {
  try {
    // Create a FormData object to send the audio file
    const formData = new FormData();
    
    // Determine the file extension based on the MIME type
    let fileExtension = '.webm';
    const mimeType = audioBlob.type;
    
    if (mimeType) {
      if (mimeType.includes('wav') || mimeType.includes('wave')) {
        fileExtension = '.wav';
      } else if (mimeType.includes('mp3')) {
        fileExtension = '.mp3';
      } else if (mimeType.includes('ogg')) {
        fileExtension = '.ogg';
      } else if (mimeType.includes('webm')) {
        fileExtension = '.webm';
      }
    }
    
    // Append the audio blob with the appropriate filename
    formData.append('file', audioBlob, `recording${fileExtension}`);
    
    // Send the request to the backend
    const response = await fetch(`${API_BASE_URL}/process-audio`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error processing audio:', error);
    throw error;
  }
};

/**
 * Process text with the assistant
 * @param text The text to process
 * @returns The assistant's response
 */
export const processTextWithAssistant = async (text: string): Promise<AssistantResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/process-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error processing text:', error);
    throw error;
  }
};

/**
 * Get transcription history
 * @returns The transcription history
 */
export const getTranscriptionHistory = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/history`);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting transcription history:', error);
    throw error;
  }
};

/**
 * Get system status
 * @returns The system status
 */
export const getSystemStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/status`);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting system status:', error);
    throw error;
  }
};

/**
 * Check API health
 * @returns The API health status
 */
export const checkApiHealth = async (): Promise<{
  status: string;
  timestamp: string;
}> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};

export const getAudioFile = async (audioPath: string): Promise<Blob> => {
  try {
    const response = await fetch(`${API_BASE_URL}${audioPath}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: ${response.status}`);
    }
    
    return await response.blob();
  } catch (error) {
    console.error('Error fetching audio:', error);
    throw error;
  }
}; 