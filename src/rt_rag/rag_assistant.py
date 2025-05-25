# Main script for the RAG-powered Question-Answering Assistant

import sys
import os
import logging
import json # Import the json module
import shutil # Added for rmtree

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Print PATH for debugging Poppler/Tesseract issues
logger = logging.getLogger(__name__) # Ensure logger is available early
# Basic config for early logging if not already set up by setup_logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

path_env_var = os.environ.get('PATH', 'PATH environment variable not found.')
# Split the PATH for better readability in logs, especially if it's very long
path_directories = path_env_var.split(os.pathsep)

# Log the PATH. Using logger if available, otherwise print.
try:
    logger.info("Python script sees the following PATH components:")
    for i, p_dir in enumerate(path_directories):
        logger.info(f"  PATH[{i}]: {p_dir}")
    if not any("poppler" in p.lower() for p in path_directories):
        logger.warning("Poppler's directory does NOT seem to be in the Python script's PATH!")
    if not any("tesseract" in p.lower() for p in path_directories):
        logger.warning("Tesseract's directory does NOT seem to be in the Python script's PATH!")
except NameError: # Fallback if logger isn't fully set up this early
    print("Python script sees the following PATH components:")
    for i, p_dir in enumerate(path_directories):
        print(f"  PATH[{i}]: {p_dir}")
    if not any("poppler" in p.lower() for p in path_directories):
        print("WARNING: Poppler's directory does NOT seem to be in the Python script's PATH!")
    if not any("tesseract" in p.lower() for p in path_directories):
        print("WARNING: Tesseract's directory does NOT seem to be in the Python script's PATH!")

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
    UnstructuredPDFLoader
)
from langchain.docstore.document import Document # Ensure Document is imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()

# --- Logging Setup ---
LOG_FILE = "rag_assistant.log"

def setup_logging():
    """Configures logging for the application."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler() # To also print to console
        ]
    )

setup_logging() 

logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Constants ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/"

def load_documents():
    """Loads documents from the data directory, supporting PDF, TXT, and JSON files."""
    all_documents = []
    
    # Supported file types and their loaders
    file_handlers = {
        ".pdf": {
            "loader_primary": UnstructuredPDFLoader,
            "loader_fallback": PyPDFLoader,
            "primary_kwargs": {"mode": "single", "strategy": "auto"},
            "fallback_kwargs": {},
            "log_msg_primary": "Loading {file} using UnstructuredPDFLoader...",
            "log_msg_fallback": "Attempting to load {file} with PyPDFLoader as a fallback..."
        },
        ".txt": {
            "loader_primary": TextLoader,
            "primary_kwargs": {"encoding": 'utf-8'},
            "log_msg_primary": "Loading {file} using TextLoader..."
        }
    }

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logger.info(f"Created data directory: {DATA_PATH}")

    for filename in os.listdir(DATA_PATH):
        file_ext = os.path.splitext(filename)[1].lower()
        file_path = os.path.join(DATA_PATH, filename)

        if file_ext in file_handlers:
            handler_config = file_handlers[file_ext]
            logger.info(handler_config["log_msg_primary"].format(file=filename))
            try:
                loader_class = handler_config["loader_primary"]
                loader = loader_class(file_path, **handler_config["primary_kwargs"])
                documents = loader.load()
                logger.info(f"Successfully loaded {filename} ({len(documents)} docs).")
                all_documents.extend(documents)
            except Exception as e_primary:
                logger.error(f"Error loading {filename} with {loader_class.__name__}: {e_primary}")
                if "loader_fallback" in handler_config:
                    logger.info(handler_config["log_msg_fallback"].format(file=filename))
                    try:
                        fallback_loader_class = handler_config["loader_fallback"]
                        loader_fallback = fallback_loader_class(file_path, **handler_config.get("fallback_kwargs", {}))
                        documents = loader_fallback.load()
                        logger.info(f"Successfully loaded {filename} with {fallback_loader_class.__name__} ({len(documents)} docs).")
                        all_documents.extend(documents)
                    except Exception as e_fallback:
                        logger.error(f"Error loading {filename} with {fallback_loader_class.__name__} as fallback: {e_fallback}")
        elif file_ext == ".json": # Custom handling for JSON files
            logger.info(f"Loading {filename} using custom JSON processing...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list): # Expecting a list of records
                    for i, record in enumerate(data):
                        content = record.get("publication_description", "") # Use 'publication_description'
                        if not content:
                            logger.warning(f"Record {i} in {filename} has no 'publication_description' or it's empty.")
                        
                        metadata = {
                            "source": filename, # Keep original filename as source
                            "title": record.get("title"),
                            "authors": record.get("authors"),
                            "year": record.get("year"),
                            # Add other fields from 'record' as needed, e.g., record_index
                            "record_index": i 
                        }
                        # Filter out None values from metadata for cleaner logs/usage
                        metadata = {k: v for k, v in metadata.items() if v is not None}
                        
                        doc = Document(page_content=content, metadata=metadata)
                        all_documents.append(doc)
                    logger.info(f"Successfully processed {filename} ({len(data)} records).")
                else:
                    logger.warning(f"Skipping {filename}: Expected a list of JSON objects, but got {type(data)}.")
            except json.JSONDecodeError as e_json_decode:
                logger.error(f"Error decoding JSON from {filename}: {e_json_decode}")
            except Exception as e_custom_json:
                logger.error(f"Error processing {filename} with custom JSON handler: {e_custom_json}")
        else:
            logger.info(f"Skipping unsupported file type: {filename}")

    if not all_documents:
        logger.warning(f"No processable documents found in {DATA_PATH}. Cannot proceed without data.")
        # Create a sample file only if the directory was truly empty of processable files
        with open(os.path.join(DATA_PATH, "sample_document.txt"), "w") as f:
            f.write("This is a sample document. Replace it with your own data or add supported file types.")
        logger.info(f"A sample document 'sample_document.txt' has been created in {DATA_PATH}. Please add actual documents and restart.")
        print(f"No processable documents found in {DATA_PATH}. A sample_document.txt has been created. Please add your documents and restart.")
        return all_documents # Return empty list

    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def get_text_chunks(documents):
    if not documents:
        logger.warning("No documents provided to get_text_chunks.")
        return []

    logger.info(f"Inspecting content of {len(documents)} documents before splitting:")
    for i, doc in enumerate(documents[:3]): 
        content_snippet = doc.page_content[:200].strip() 
        page_num = doc.metadata.get('page', 'N/A')
        source_file = doc.metadata.get('source', 'N/A')
        if not content_snippet:
            logger.info(f"  Document {i} (source: {source_file}, page: {page_num}) appears to have no extractable text content.")
        else:
            logger.info(f"  Document {i} (source: {source_file}, page: {page_num}) content snippet: '{content_snippet}...'" )
        
        if i == 0 and not content_snippet:
            logger.warning("The first document page appears empty. This might indicate a problem with PDF text extraction (e.g., image-based PDF or complex encoding).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs_with_content = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

    if not docs_with_content:
        logger.warning("No documents with actual text content found to split after filtering.")
        return []
        
    logger.info(f"Splitting {len(docs_with_content)} documents with actual content into chunks...")
    chunks = text_splitter.split_documents(docs_with_content)
    logger.info(f"Created {len(chunks)} text chunks.")
    return chunks

def get_vector_store(text_chunks, embeddings):
    """Creates a FAISS vector store, always rebuilding it."""
    # Always rebuild the vector store to ensure it reflects current documents
    logger.info(f"Attempting to remove existing vector store at {VECTORSTORE_PATH} if it exists...")
    if os.path.exists(VECTORSTORE_PATH):
        try:
            shutil.rmtree(VECTORSTORE_PATH) # Remove the directory and its contents
            logger.info(f"Successfully removed existing vector store directory: {VECTORSTORE_PATH}")
        except OSError as e:
            logger.error(f"Error removing directory {VECTORSTORE_PATH}: {e.strerror} - proceeding to create new.")
            # If removal fails, FAISS might still overwrite or handle it, or fail later.
            # For robustness, one might want to ensure the path is clear or use a new path.

    logger.info("Creating new vector store...")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True) # Ensure directory exists
    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info(f"Vector store created and saved to {VECTORSTORE_PATH}")
    return vectorstore

def initialize_rag_chain():
    """Initializes all components of the RAG chain and returns the chain."""
    logger.info("Initializing RAG assistant components...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in the .env file.")
        print("Error: OPENAI_API_KEY not found. Please create a .env file with your key.")
        return None

    documents = load_documents()
    if not documents:
        logger.warning(f"No documents found in {DATA_PATH}. Cannot proceed without data.")
        return None

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        logger.error("No text could be extracted from the documents. Check document content and loaders.")
        print("Error: No text could be extracted from documents.")
        return None

    logger.info("Initializing OpenAI embeddings model...")
    embeddings = OpenAIEmbeddings()

    vectorstore = get_vector_store(text_chunks, embeddings)
    if not vectorstore:
        logger.error("Failed to create or load vector store.")
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever created from vector store.")

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    logger.info(f"ChatOpenAI LLM initialized with model gpt-3.5-turbo.")

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    logger.info("ConversationBufferMemory initialized.")

    # Create ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False  # Disable verbose output
    )
    logger.info("ConversationalRetrievalChain created successfully.")
    return qa_chain

def main():
    """Main function to run the RAG assistant CLI."""
    logger.info("RAG Assistant CLI starting...")
    
    qa_chain = initialize_rag_chain()

    if not qa_chain:
        logger.error("Failed to initialize RAG chain. Exiting CLI.")
        return

    logger.info("RAG Assistant is ready. CLI started. Type 'exit' to quit.")
    print("\nRAG Assistant with session memory is ready. Type 'exit' to quit.")

    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() == 'exit':
            logger.info("User typed 'exit'. Shutting down.")
            break
        if not user_question.strip():
            continue

        logger.info(f"Received question: '{user_question}'")
        try:
            logger.info("Invoking ConversationalRetrievalChain...")
            result = qa_chain.invoke({"question": user_question})
            answer = result.get('answer', "Sorry, I couldn't find an answer.")
            
            logger.info(f"Retrieved answer: '{answer[:100]}...'" if len(answer) > 100 else f"Retrieved answer: '{answer}'")
            
            print("\nAnswer:")
            print(answer)

            if result.get('source_documents'):
                logger.info(f"Retrieved {len(result['source_documents'])} source documents.")
                # print("\nSources:")
                # for i, doc in enumerate(result['source_documents']):
                #     print(f"Source {i+1}: {doc.metadata.get('source', 'N/A')} (Page: {doc.metadata.get('page', 'N/A')})")
                #     # print(doc.page_content[:200] + "...") # Print snippet of source

        except Exception as e:
            logger.error(f"Error processing question '{user_question}': {e}", exc_info=True)
            print(f"Error: Could not process your question. {e}")

if __name__ == "__main__":
    main()
    logger.info("RAG Assistant finished.")
