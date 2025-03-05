/**
 * Interface for the assistant's response
 */
export interface AssistantResponse {
  input_text: string;
  response_text: string;
  audio_url: string;
  timings: Record<string, number>;
} 