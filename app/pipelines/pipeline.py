from typing import Dict, Any, Generator
import os
from .retrieval import get_vectorstore
from .generation import create_generation_chain, get_llm
from .memory import get_history, add_turn
from app.guardrails.guardrails import check_relevance, check_faithfulness

def invoke_chat(session_id: str, question: str, skip_relevance: bool = False) -> Dict[str, Any]:
    """
    Handles a single turn of chat for a given session.
    1. Guardrail 1 (Relevance)
    2. Retrieval
    3. LLM Generation
    4. Guardrail 2 (Faithfulness check)
    5. State Update
    """
    llm = get_llm()
    vectorstore = get_vectorstore()
    
    # 1. Relevance Guardrail (skipped for internal calls like summarization)
    if not skip_relevance:
        relevance = check_relevance(question, vectorstore)
        if not relevance["passed"]:
             return {
                 "answer": "This question doesn't seem related to the uploaded document.",
                 "sources": [],
                 "faithfulness_warning": False
             }
         
    # 2. Retrieval
    # We fetch more context. The relevance used k=3, here we can use k=5
    docs = vectorstore.similarity_search(question, k=5)
    
    # Format context and sources
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Collect unique source references (e.g. page numbers)
    # PyPDFDirectoryLoader adds 'page' to metadata
    sources = set()
    for doc in docs:
        metadata = doc.metadata
        if 'page' in metadata:
            sources.add(f"Page {metadata['page'] + 1}")
        elif 'source' in metadata:
            sources.add(os.path.basename(metadata['source']))
    sources_list = list(sources)
    
    # Optional Phase 2 refinement: Use a history re-writer
    # For simplicity of the baseline, we inject stringified history directly
    history_dicts = get_history(session_id)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_dicts])
    
    # 3. Generation
    chain = create_generation_chain()
    answer = chain.invoke({
        "context": context_text,
        "history": history_text,
        "question": question
    })
    
    # 4. Faithfulness Guardrail
    # Passed context documents to the faithfulness checker
    faithfulness = check_faithfulness(answer, docs, llm)
    
    # 5. Save State
    add_turn(session_id, question, answer)
    
    return {
        "answer": answer,
        "sources": sources_list,
        "faithfulness_warning": not faithfulness["passed"]
    }

def stream_chat(session_id: str, question: str, skip_relevance: bool = False) -> Generator[str, None, None]:
    """
    Generator that yields tokens from the LLM based on the conversation history.
    Since relevance and faithfulness guards require the full string, they are adapted or skipped for true real-time streaming, 
    but the spec calls for SSE stream output. We will just yield tokens for the answer.
    """
    llm = get_llm()
    vectorstore = get_vectorstore()
    
    # Guardrail 1 (skipped for internal calls like summarization)
    if not skip_relevance:
        relevance = check_relevance(question, vectorstore)
        if not relevance["passed"]:
             yield "This question doesn't seem related to the uploaded document."
             return
         
    docs = vectorstore.similarity_search(question, k=5)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    sources = set()
    for doc in docs:
        metadata = doc.metadata
        if 'page' in metadata:
            sources.add(f"Page {metadata['page'] + 1}")
        elif 'source' in metadata:
            sources.add(os.path.basename(metadata['source']))
            
    history_dicts = get_history(session_id)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_dicts])
    
    chain = create_generation_chain()
    
    # Collect streamed answer to run final guardrail and save state
    full_answer = ""
    for chunk in chain.stream({
        "context": context_text,
        "history": history_text,
        "question": question
    }):
        full_answer += chunk
        yield chunk
        
    faithfulness = check_faithfulness(full_answer, docs, llm)
    add_turn(session_id, question, full_answer)
    
    # Append sources and warning as metadata chunks
    sources_str = ", ".join(list(sources))
    yield f"\n\nSOURCES: {sources_str}"
    
    if not faithfulness["passed"]:
         yield f"\n\nWARNING: The generated answer might not be completely faithful to the provided context."

