// src/components/ChatInterface.js
import React, { useState, useEffect, useRef } from 'react';
import { askQuestion } from '../apiService';
import './ChatInterface.css';

function ChatInterface() {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: "Hello! Ask me anything about the loaded documents." }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null); // Stores error message for display
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);
    setError(null); // Clear previous errors

    try {
      const response = await askQuestion(currentInput);
      const assistantMessage = {
        sender: 'assistant',
        text: response.answer,
        sources: response.source_documents || [],
      };
      setMessages((prevMessages) => [...prevMessages, assistantMessage]);
    } catch (err) {
      console.error("Error fetching response:", err);
      const errorMessageText = err.message || 'Failed to get a response from the assistant.';
      setError(errorMessageText); // Set error for display below input
      const errorMessageForChat = { 
        sender: 'assistant', 
        text: `Error: ${errorMessageText}`,
        isError: true 
      };
      // Add error message to chat history as well
      setMessages((prevMessages) => [...prevMessages, errorMessageForChat]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>RAG Assistant Chat</h2>
      </div>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender} ${msg.isError ? 'error-message' : ''}`}>
            <div className="message-bubble">
              <p>{msg.text}</p>
              {msg.sender === 'assistant' && !msg.isError && msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx} title={source.content?.substring(0, 300) + (source.content?.length > 300 ? '...' : '')}>
                        {source.metadata?.source ? source.metadata.source.split(/[\\/]/).pop() : 'Unknown Source'}
                        {source.metadata?.page !== undefined && source.metadata.page !== null && ` (Page: ${source.metadata.page})`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} /> {/* For auto-scrolling */} 
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
      {error && <div className="error-indicator">{error}</div>} {/* Display error below input */}
    </div>
  );
}

export default ChatInterface;
