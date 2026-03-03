import pytest
from unittest.mock import patch, MagicMock

@patch("app.pipelines.generation.ChatGroq")
def test_create_generation_chain(mock_chatgroq):
    from app.pipelines.generation import create_generation_chain, get_rag_prompt
    
    # Simply test that the chain can be created without error
    chain = create_generation_chain()
    prompt = get_rag_prompt()
    
    assert prompt is not None
    assert chain is not None
    assert "context" in prompt.input_variables
    assert "history" in prompt.input_variables
    assert "question" in prompt.input_variables
