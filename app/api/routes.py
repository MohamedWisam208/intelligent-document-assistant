from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pydantic
from pydantic import BaseModel
import os
import uuid
import time
import asyncio
from app.pipelines.retrieval import ingest_documents, UPLOAD_DIR
from app.pipelines.memory import (
    create_session, get_session_ids, get_session_metadata, 
    delete_session, purge_expired_sessions, clear_history
)
from app.pipelines.pipeline import invoke_chat, stream_chat

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ClearHistoryRequest(BaseModel):
    session_id: str


@router.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    start_time = time.time()
    
    # Save the file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ingest documents via Phase 1 pipeline
    try:
         stats = ingest_documents()
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

    # Time calculation
    elapsed_time = round(time.time() - start_time, 2)
    
    # Phase 4 integration
    # The spec says POST /api/upload now calls create_session(collection_name)
    # Since we have one collection, we can just use the default
    from app.pipelines.retrieval import COLLECTION_NAME
    session_id = create_session(COLLECTION_NAME)

    return JSONResponse(status_code=200, content={
        "filename": file.filename,
        "num_pages": stats.get("num_pages", 0),
        "num_chunks": stats.get("num_chunks", 0),
        "elapsed_time": elapsed_time,
        "session_id": session_id,
        "message": "Upload & Ingestion Complete"
    })

@router.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Stateless chat endpoint (returns full JSON at once instead of streaming)."""
    try:
        response_data = invoke_chat(request.session_id, request.question)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint using SSE."""
    async def event_generator():
        # Wrap the synchronous generator in an async generator
        for token in stream_chat(request.session_id, request.question):
            # Format as SSE event
            yield token
            await asyncio.sleep(0.01) # Yield control
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/api/history/clear")
async def clear_chat_history(request: ClearHistoryRequest):
    try:
        clear_history(request.session_id)
        return {"message": "Chat history cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/summarize")
async def summarize_document(request: ClearHistoryRequest):
    """Endpoint triggered by Summarize button using the session."""
    # We just run a simple chat invoking the summarize doc
    question = "Please summarize the main points of this document."
    try:
        response_data = invoke_chat(request.session_id, question, skip_relevance=True)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/sessions")
async def list_sessions():
    """List all active non-expired session IDs."""
    sessions = get_session_ids()
    return {"sessions": sessions}

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Return SessionMetadata for a given session."""
    metadata = get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found.")
    return metadata

@router.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Return the full message history for a given session."""
    from app.pipelines.memory import get_session_data
    data = get_session_data(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"messages": data.get("messages", [])}


@router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete session file from disk."""
    delete_session(session_id)
    return {"message": f"Session {session_id} deleted."}

@router.post("/api/sessions/purge")
async def purge_sessions():
    """Delete all sessions inactive > SESSION_TTL_HOURS."""
    count = purge_expired_sessions()
    return {"purged_count": count, "message": f"Purged {count} expired sessions."}

