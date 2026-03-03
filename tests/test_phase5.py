import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

def test_check_faithfulness_pass():
    from app.guardrails.guardrails import check_faithfulness
    
    mock_llm = MagicMock()
    # Mock LLM response for YES
    mock_response = MagicMock()
    mock_response.content = "YES, the answer is fully stated in the context."
    mock_llm.invoke.return_value = mock_response
    
    res = check_faithfulness("The apple is red", [Document(page_content="The apple is red")], mock_llm)
    assert res["passed"] is True
    assert res["score"] == "YES"

def test_check_faithfulness_fail():
    from app.guardrails.guardrails import check_faithfulness
    
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "NO, the context says nothing about bananas."
    mock_llm.invoke.return_value = mock_response
    
    res = check_faithfulness("The banana is yellow", [Document(page_content="The apple is red")], mock_llm)
    assert res["passed"] is False
    assert res["score"] == "NO"
