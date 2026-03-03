import pytest
import os
from unittest.mock import patch, MagicMock

# Mock PyPDFDirectoryLoader and RecursiveCharacterTextSplitter and Chroma to run without keys or files
@patch("app.pipelines.retrieval.PyPDFDirectoryLoader")
@patch("app.pipelines.retrieval.RecursiveCharacterTextSplitter")
@patch("app.pipelines.retrieval.Chroma")
def test_ingest_documents_empty(mock_chroma, mock_splitter, mock_loader):
    from app.pipelines.retrieval import ingest_documents
    
    # Mock loader returning empty
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = []
    
    res = ingest_documents()
    assert res["num_pages"] == 0
    assert res["num_chunks"] == 0

@patch("app.pipelines.retrieval.PyPDFDirectoryLoader")
@patch("app.pipelines.retrieval.RecursiveCharacterTextSplitter")
@patch("app.pipelines.retrieval.Chroma")
def test_ingest_documents_with_docs(mock_chroma, mock_splitter, mock_loader):
    from app.pipelines.retrieval import ingest_documents
    from langchain_core.documents import Document
    
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [Document(page_content="test doc 1"), Document(page_content="test doc 2")]
    
    mock_splitter_instance = MagicMock()
    mock_splitter.return_value = mock_splitter_instance
    mock_splitter_instance.split_documents.return_value = [
        Document(page_content="chunk1"),
         Document(page_content="chunk2"),
         Document(page_content="chunk3")
    ]
    
    res = ingest_documents()
    assert res["num_pages"] == 2
    assert res["num_chunks"] == 3
