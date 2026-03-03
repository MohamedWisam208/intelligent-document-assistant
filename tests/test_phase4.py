import pytest
import os
from app.pipelines.memory import create_session, get_history, add_turn, delete_session

def test_memory_lifecycle():
    # Create
    session_id = create_session("test_collection")
    assert session_id is not None
    
    # Empty history
    history = get_history(session_id)
    assert len(history) == 0
    
    # Add turn
    add_turn(session_id, "Hello", "Hi there")
    
    # Check history
    history = get_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "Hello"
    assert history[1]["content"] == "Hi there"
    
    # Cleanup
    delete_session(session_id)
    history = get_history(session_id)
    assert len(history) == 0
