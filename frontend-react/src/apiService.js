// src/apiService.js

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8000'; // Your FastAPI backend URL

/**
 * Sends a question to the backend API and returns the response.
 * @param {string} questionText The question to ask the assistant.
 * @returns {Promise<object>} The API response (answer and source documents).
 * @throws {Error} If the API request fails or returns an error.
 */
export const askQuestion = async (questionText) => {
  console.log("Sending question to API:", questionText);
  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // Ensure the question is definitely a string, though it should be already
      body: JSON.stringify({ question: String(questionText) }), 
    });

    console.log("API Response Status:", response.status);
    const responseData = await response.json();
    console.log("API Response Data:", responseData);

    if (!response.ok) {
      // Log the detailed error from the API if available
      console.error('API Error Response:', responseData);
      // Construct a more informative error message
      let errorMessage = `API request failed with status ${response.status}`;
      if (responseData && responseData.detail) {
        if (Array.isArray(responseData.detail) && responseData.detail.length > 0 && responseData.detail[0].msg) {
          errorMessage += `: ${responseData.detail[0].msg}`;
        } else if (typeof responseData.detail === 'string') {
          errorMessage += `: ${responseData.detail}`;
        }
      }
      throw new Error(errorMessage);
    }

    return responseData;
  } catch (error) {
    console.error('Error asking question:', error);
    throw error; // Re-throw the error to be caught by the UI component
  }
};
