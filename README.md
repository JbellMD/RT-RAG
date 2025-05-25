# RAG-Powered Question-Answering Assistant

This project is a simple question-answering assistant that uses Retrieval-Augmented Generation (RAG) to pull answers from a custom document set. It's built as part of the Agentic AI Developer Certification Program.

## Project Overview

The assistant uses LangChain to create a pipeline that includes:
- Prompt formulation
- Vector store retrieval (e.g., FAISS, Chroma)
- LLM-generated response
- Document ingestion into the vector store

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
    ```

5.  **Add custom documents:**
    Place your custom documents (e.g., `.txt`, `.pdf`, `.md` files) into the `data/` directory.

## How to Run

(Instructions to be added once the script is developed)

```bash
python rag_assistant.py
```

## Sample Inputs and Outputs

(Examples to be added after testing)

### Example 1

**Question:** ...

**Answer:** ...

## Optional Enhancements

- Session-based memory
- Intermediate reasoning steps (e.g., ReAct or CoT-style chaining)
- Basic logging or observability
