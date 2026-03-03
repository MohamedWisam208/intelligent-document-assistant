import torch
import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

UPLOAD_DIR = "./data/uploads"
VECTORSTORE_DIR = "./data/vectorstore"
COLLECTION_NAME = "smart_assistant_docs"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Configure embeddings model as specified in Phase 1
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
def get_vectorstore() -> Chroma:
    """Returns the persistent Chroma vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=VECTORSTORE_DIR
    )

def ingest_documents() -> dict:
    """
    Reads PDFs from the upload directory, chunks them, and adds them to the vectorstore.
    Returns statistics about the ingestion process.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        
    documents = []
    # Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(UPLOAD_DIR)
    documents = loader.load()
    
    if not documents:
        return {"num_pages": 0, "num_chunks": 0, "message": "No documents found to ingest."}

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    # Store chunks in Chroma
    vectorstore = get_vectorstore()
    
    # Optional: We could clear the current store and reload, or just add.
    # For now, we simply add the new chunks. In a real scenario we'd track IDs.
    vectorstore.add_documents(chunks)
    
    return {
        "num_pages": len(documents),
        "num_chunks": len(chunks),
        "message": "Documents ingested successfully."
    }

def clear_vectorstore():
    """Removes the persistent vectorstore directory effectively clearing the DB."""
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
