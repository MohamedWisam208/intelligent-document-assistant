import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

SESSION_DIR = "./data/sessions"
SESSION_TTL_HOURS = 24  # Default TTL

def _get_session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def create_session(collection_name: str) -> str:
    """Creates a JSON file for a new session and returns the session ID."""
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)
        
    session_id = str(uuid.uuid4())
    now_str = datetime.now().isoformat()
    
    session_data = {
        "session_id": session_id,
        "collection_name": collection_name,
        "created_at": now_str,
        "last_active": now_str,
        "messages": []
    }
    
    with open(_get_session_path(session_id), "w") as f:
        json.dump(session_data, f, indent=4)
        
    return session_id

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Helper to read session data."""
    path = _get_session_path(session_id)
    if not os.path.exists(path):
        return None
        
    with open(path, "r") as f:
        return json.load(f)

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Returns the raw message history (list of dicts)."""
    data = get_session_data(session_id)
    if not data:
        return []
    return data.get("messages", [])

def add_turn(session_id: str, question: str, answer: str):
    """Appends HumanMessage + AIMessage pair, updates last_active timestamp."""
    data = get_session_data(session_id)
    if not data:
        return
        
    data["messages"].append({"role": "human", "content": question})
    data["messages"].append({"role": "ai", "content": answer})
    data["last_active"] = datetime.now().isoformat()
    
    with open(_get_session_path(session_id), "w") as f:
        json.dump(data, f, indent=4)

def trim_history(session_id: str, max_turns: int = 10):
    """Keeps only last N turns to bound prompt size."""
    # Each turn is 2 messages (human + ai)
    data = get_session_data(session_id)
    if not data:
        return
        
    messages = data["messages"]
    max_messages = max_turns * 2
    if len(messages) > max_messages:
        data["messages"] = messages[-max_messages:]
        
    with open(_get_session_path(session_id), "w") as f:
        json.dump(data, f, indent=4)

def clear_history(session_id: str):
    """Wipes messages, keeps session file and collection_name."""
    data = get_session_data(session_id)
    if not data:
        return
        
    data["messages"] = []
    data["last_active"] = datetime.now().isoformat()
    
    with open(_get_session_path(session_id), "w") as f:
        json.dump(data, f, indent=4)

def delete_session(session_id: str):
    """Removes JSON file from disk."""
    path = _get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)

def get_session_metadata(session_id: str) -> Dict[str, Any]:
    """Returns minimal session metadata without full history."""
    data = get_session_data(session_id)
    if not data:
        return None
        
    return {
        "session_id": data["session_id"],
        "collection_name": data["collection_name"],
        "created_at": data["created_at"],
        "last_active": data["last_active"],
        "turn_count": len(data["messages"]) // 2
    }

def get_session_ids() -> List[str]:
    """Returns all non-expired session IDs from disk."""
    if not os.path.exists(SESSION_DIR):
        return []
        
    session_ids = []
    now = datetime.now()
    
    for filename in os.listdir(SESSION_DIR):
        if not filename.endswith(".json"):
            continue
            
        session_id = filename.replace(".json", "")
        # Check expiry
        data = get_session_data(session_id)
        if data:
            last_active = datetime.fromisoformat(data["last_active"])
            hours_diff = (now - last_active).total_seconds() / 3600
            
            if hours_diff <= SESSION_TTL_HOURS:
                session_ids.append(session_id)
            else:
                # Purge if expired
                delete_session(session_id)
                
    return session_ids

def purge_expired_sessions() -> int:
    """Deletes sessions where last_active > SESSION_TTL_HOURS ago. Returns count."""
    if not os.path.exists(SESSION_DIR):
        return 0
        
    now = datetime.now()
    count = 0
    
    for filename in os.listdir(SESSION_DIR):
        if not filename.endswith(".json"):
            continue
            
        session_id = filename.replace(".json", "")
        data = get_session_data(session_id)
        if data:
            last_active = datetime.fromisoformat(data["last_active"])
            hours_diff = (now - last_active).total_seconds() / 3600
            
            if hours_diff > SESSION_TTL_HOURS:
                delete_session(session_id)
                count += 1
                
    return count
