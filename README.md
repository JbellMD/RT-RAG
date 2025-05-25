# RAG-Powered Question-Answering Assistant with Chat UI

This project is a question-answering assistant that uses Retrieval-Augmented Generation (RAG) to pull answers from a custom document set. It features a Python-based FastAPI backend and a React frontend for a user-friendly chat interface.

## Project Overview

The assistant uses LangChain to create a pipeline that includes:
- Document ingestion and processing (PDFs, text files).
- Text splitting and embedding generation (using OpenAI).
- Vector store creation and retrieval (using FAISS).
- Conversational chain management with memory.
- A FastAPI backend to serve the RAG chain via an API.
- A React frontend providing a chat interface to interact with the assistant.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python (3.9+ recommended)
- Node.js and npm (for the React frontend)
- Pip (Python package installer)
- Git
- **Poppler**: Required for PDF processing by `UnstructuredPDFLoader`. Ensure its `bin` directory is in your system's PATH. (See `setup_windows.bat` for an example on Windows or install via your system's package manager on Linux/macOS).
- **Tesseract OCR**: Also required by `UnstructuredPDFLoader` for OCR in PDFs. Ensure it's installed and its directory is in your system's PATH. (See `setup_windows.bat` or install via system package manager).

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RT-RAG
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your API keys (e.g., OpenAI API key):
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # KMP_DUPLICATE_LIB_OK="TRUE" # Add this if you encounter OpenMP issues
    ```

5.  **Add custom documents:**
    Place your custom documents (e.g., `.txt`, `.pdf`, `.md` files) into the `data/` directory.

## How to Run

The application consists of two main parts: the FastAPI backend and the React frontend. You'll need to run them in separate terminals.

### 1. Backend (FastAPI Server)

   - Ensure your virtual environment is activated (`venv\Scripts\activate` or `source venv/bin/activate`).
   - Make sure your `.env` file is correctly set up with your `OPENAI_API_KEY`.
   - The first time you run, the RAG assistant will process documents in the `data/` directory and create a vector store. This might take some time depending on the number and size of your documents.

   Navigate to the project root directory (`RT-RAG`) and run:
   ```bash
   python api_main.py
   ```
   The API server should start, typically on `http://127.0.0.1:8000`.

### 2. Frontend (React App)

   - Open a new terminal.
   - Navigate to the React app directory:
     ```bash
     cd frontend-react
     ```
   - Install Node.js dependencies (if you haven't already):
     ```bash
     npm install
     ```
   - Start the React development server:
     ```bash
     npm start
     ```
   This will usually open the application automatically in your default web browser at `http://localhost:3000`.

### Using the Application

Once both the backend and frontend are running:
1. Open your web browser and go to `http://localhost:3000` (or the URL provided by the `npm start` command).
2. You should see the chat interface.
3. Type your questions about the documents you placed in the `data/` folder and press Enter or click Send.
4. The assistant will process your question and display the answer along with any relevant source document snippets.

## CLI (Legacy)

The original command-line interface can still be run (ensure your virtual environment is active and `.env` is set up):
```bash
python rag_assistant.py
```

This README provides a comprehensive guide to setting up and running the RAG assistant.
If you encounter any issues, please check the log files (`rag_assistant.log`) or console output for errors.

## Sample Inputs and Outputs

(Examples to be added after testing)

### Example 1

**Question:** ...

**Answer:** ...

## Optional Enhancements

- Session-based memory
- Intermediate reasoning steps (e.g., ReAct or CoT-style chaining)
- Basic logging or observability
