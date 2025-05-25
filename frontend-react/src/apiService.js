// src/apiService.js

const API_BASE_URL = 'http://127.0.0.1:8000'; // Your FastAPI backend URL

export const askQuestion = async (question, sessionId = null) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question: question,
        session_id: sessionId 
      }),
    });

    if (!response.ok) {
      // Try to parse error message from backend, otherwise use default
      let errorData = { detail: `HTTP error! status: ${response.status}` };
      try {
        errorData = await response.json();
      } catch (e) {
        // console.warn("Could not parse error response JSON.");
      }
      console.error('API Error Response:', errorData);
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error asking question:', error);
    // Ensure the error thrown is an Error object with a message property
    if (error instanceof Error) {
      throw error;
    } else {
      throw new Error(String(error));
    }
  }
};
