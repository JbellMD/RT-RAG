# Main FastAPI application for the RAG Assistant UI

import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # To handle CORS for local development
from pydantic import BaseModel
import logging
from typing import Optional # Import Optional
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Import the RAG chain initializer from your existing script
from rag_assistant import initialize_rag_chain, setup_logging

# Setup logging (ensure it's called before other loggers if not already configured)
# This assumes setup_logging() configures a root logger or the specific loggers used.
if not logging.getLogger().hasHandlers():
    setup_logging() 
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="API for interacting with the RAG-powered Question-Answering Assistant",
    version="0.1.0"
)

# --- Exception Handler for Validation Errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request: {request.method} {request.url}")
    logger.error(f"Error details: {exc.errors()}") # This will print the detailed Pydantic errors
    # You can customize the response if needed, but FastAPI's default is usually fine
    # return JSONResponse(
    #     status_code=422,
    #     content={"detail": exc.errors()},
    # )
    # Re-raise to let FastAPI handle the default 422 response, or use the JSONResponse above
    # For just logging, we can let the default handler do its work after logging.
    # The default handler is part of Starlette, FastAPI's core.
    # To ensure the client gets the standard 422, we can call the default handler explicitly
    # or simply return what it would have returned.
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# --- CORS Middleware --- 
# Allows requests from your frontend (which will be on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity in local dev
    # allow_origins=["http://localhost:3000"], # Or specify your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic Models for Request/Response --- 
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None # Changed to Optional[str]

class AnswerResponse(BaseModel):
    answer: str
    sources: list = []
    session_id: Optional[str] = None # Changed to Optional[str]

# --- RAG Chain Initialization --- 
# We'll store the chain globally. 
# For more complex scenarios with multiple users/sessions, you might manage chains differently.
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG chain when the application starts."""
    global qa_chain
    logger.info("FastAPI application starting up...")
    logger.info("Attempting to initialize RAG chain...")
    qa_chain = initialize_rag_chain()
    if qa_chain is None:
        logger.error("CRITICAL: RAG chain initialization failed. API will not be functional.")
        # You might want to prevent the app from starting or handle this more gracefully
    else:
        logger.info("RAG chain initialized successfully. API is ready.")

# --- API Endpoints --- 
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question, gets an answer from the RAG chain, and returns it.
    """
    global qa_chain
    logger.info(f"Received request for /ask: {request.question}")

    if qa_chain is None:
        logger.error("RAG chain is not initialized. Cannot process question.")
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG chain not initialized. Please try again later.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Here, you would ideally pass the session_id to the chain if you implement per-session memory
        # For now, the global qa_chain uses its own ConversationBufferMemory
        logger.info(f"Invoking RAG chain for question: '{request.question}'")
        result = qa_chain.invoke({"question": request.question})
        
        answer = result.get('answer', "Sorry, I couldn't find an answer.")
        source_documents = result.get("source_documents", [])
        # Extract source filenames and ensure uniqueness
        raw_sources = [doc.metadata.get("source", "Unknown source") for doc in source_documents]
        unique_sources = sorted(list(set(raw_sources))) # Ensure uniqueness and consistent order

        logger.info(f"Successfully processed question. Answer: {answer[:50]}...")
        return AnswerResponse(
            answer=answer, 
            sources=unique_sources, 
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Error processing question in API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Assistant API. Use the /ask endpoint to ask questions."}

# --- Main block to run the server --- 
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is available before trying to run
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment. Please set it in .env file.")
        print("Error: OPENAI_API_KEY not found. Please create a .env file with your key and ensure it's loaded.")
    else:
        logger.info("Starting Uvicorn server for RAG Assistant API...")
        # For development, reload=True is useful. For production, set it to False.
        # The host '0.0.0.0' makes it accessible on your network, '127.0.0.1' for local only.
        uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
