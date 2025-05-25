# Main script for the RAG-powered Question-Answering Assistant

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

DATA_PATH = "data/"
VECTORSTORE_PATH = "faiss_index"

def load_documents():
    """Loads documents from the data directory."""
    # Supporting multiple file types
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        # Add more loaders here if needed, e.g., for .md, .docx
    }
    all_documents = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in loaders:
            print(f"Loading {filename}...")
            try:
                loader_class = loaders[file_ext]
                if file_ext == '.pdf': # PyPDFLoader takes filepath directly
                    loader = loader_class(filepath)
                else: # Other loaders might need different instantiation
                    loader = loader_class(filepath)
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Skipping {filename}, unsupported file type.")
    return all_documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def get_or_create_vectorstore(texts, embeddings):
    """Creates or loads a FAISS vector store."""
    if os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
        print(f"Loading existing vector store from {VECTORSTORE_PATH}")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vector store...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"Vector store saved to {VECTORSTORE_PATH}")
    return vectorstore

def main():
    """Main function to run the RAG assistant."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file in the project root and add your OpenAI API key:")
        print("OPENAI_API_KEY='your_openai_api_key_here'")
        return

    print("Initializing RAG assistant...")

    # 1. Load documents
    documents = load_documents()
    if not documents:
        print(f"No documents found in {DATA_PATH}. Please add some documents to query.")
        # Create a dummy file to avoid errors if data/ is empty and to guide the user
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        with open(os.path.join(DATA_PATH, "sample_document.txt"), "w") as f:
            f.write("This is a sample document. Replace it with your own data.")
        print(f"A sample document 'sample_document.txt' has been created in {DATA_PATH}.")
        print("Please add your actual documents and restart the assistant.")
        return

    # 2. Split documents into chunks
    texts = split_documents(documents)
    if not texts:
        print("No text could be extracted from the documents. Please check your document content.")
        return

    # 3. Initialize embeddings model
    embeddings = OpenAIEmbeddings()

    # 4. Create or load vector store
    vectorstore = get_or_create_vectorstore(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Define prompt template
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Use three sentences maximum and keep the answer concise.

    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 7. Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nRAG Assistant is ready. Type 'exit' to quit.")

    # 8. Simple CLI for interaction
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() == 'exit':
            break
        if not user_question.strip():
            continue

        try:
            print("Thinking...")
            answer = rag_chain.invoke(user_question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
