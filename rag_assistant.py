# Main script for the RAG-powered Question-Answering Assistant

import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # To also print to console
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

DATA_PATH = "data/"
VECTORSTORE_PATH = "faiss_index"

def load_documents():
    """Loads documents from the data directory."""
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
    }
    all_documents = []
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        logger.warning(f"Data directory {DATA_PATH} is empty or does not exist.")
        return all_documents
        
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in loaders:
            logger.info(f"Loading {filename}...")
            try:
                loader_class = loaders[file_ext]
                loader = loader_class(filepath)
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                logger.info(f"Successfully loaded {filename} ({len(loaded_docs)} pages/docs).")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}", exc_info=True)
        else:
            logger.warning(f"Skipping {filename}, unsupported file type: {file_ext}")
    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    logger.info(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Created {len(texts)} text chunks.")
    return texts

def get_or_create_vectorstore(texts, embeddings):
    """Creates or loads a FAISS vector store."""
    if os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
        logger.info(f"Loading existing vector store from {VECTORSTORE_PATH}")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Successfully loaded vector store.")
    else:
        logger.info("Creating new vector store...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info(f"Vector store created and saved to {VECTORSTORE_PATH}")
    return vectorstore

def main():
    """Main function to run the RAG assistant."""
    logger.info("RAG Assistant starting...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in the .env file.")
        print("Error: OPENAI_API_KEY not found. Please create a .env file with your key.")
        return

    logger.info("Initializing RAG assistant components...")

    documents = load_documents()
    if not documents:
        logger.warning(f"No documents found in {DATA_PATH}. Cannot proceed without data.")
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            logger.info(f"Created data directory: {DATA_PATH}")
        with open(os.path.join(DATA_PATH, "sample_document.txt"), "w") as f:
            f.write("This is a sample document. Replace it with your own data.")
        logger.info(f"A sample document 'sample_document.txt' has been created in {DATA_PATH}. Please add actual documents and restart.")
        print(f"No documents found in {DATA_PATH}. A sample_document.txt has been created. Please add your documents and restart.")
        return

    texts = split_documents(documents)
    if not texts:
        logger.error("No text could be extracted from the documents. Check document content and loaders.")
        print("Error: No text could be extracted from documents.")
        return

    logger.info("Initializing OpenAI embeddings model...")
    embeddings = OpenAIEmbeddings()

    vectorstore = get_or_create_vectorstore(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever created from vector store.")

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    logger.info(f"ChatOpenAI LLM initialized with model gpt-3.5-turbo.")

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer' # Ensure the LLM's response is stored correctly in memory
    )
    logger.info("ConversationBufferMemory initialized.")

    # Create ConversationalRetrievalChain
    # This chain will use the LLM to condense the question and chat history into a standalone question,
    # then retrieve documents, and finally use the LLM again to answer based on context and history.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True, # Optionally return source documents
        output_key='answer' # Ensure the chain's final output key is 'answer'
    )
    logger.info("ConversationalRetrievalChain created successfully.")

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
            # The chain manages chat_history internally via the memory object.
            # We just need to pass the question.
            result = qa_chain.invoke({"question": user_question})
            answer = result.get('answer', "Sorry, I couldn't find an answer.")
            
            logger.info(f"Retrieved answer: '{answer[:100]}...'" if len(answer) > 100 else f"Retrieved answer: '{answer}'")
            
            print("\nAnswer:")
            print(answer)

            # Optionally, print source documents if you want to see them
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
