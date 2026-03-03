from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_core.documents import Document

RELEVANCE_THRESHOLD = 0.35

def check_relevance(question: str, vectorstore: Chroma) -> Dict[str, Any]:
    """
    Checks if the question is topically related to the vectorstore.
    Uses cosine similarity max score from top results.
    Rejects if score < RELEVANCE_THRESHOLD.
    """
    # similarity_search_with_score returns (Document, score) where lower score is better (distance)
    # Note: HuggingFaceEmbeddings uses L2 distance by default in Chroma, 
    # but the instructions specifically mention we want relevant items.
    # To adapt "max cosine similarity", we will do a basic similarity search.
    # We will assume a normalized distance if the model is normalized.
    results = vectorstore.similarity_search_with_relevance_scores(question, k=3)
    
    if not results:
         return {
             "passed": False, 
             "score": 0.0, 
             "reason": "No context found to compute relevance."
         }
         
    # results is list of tuples: (Document, match_score)
    # Higher match_score is better when using relevance scores.
    max_score = max([score for _, score in results])
    passed = max_score >= RELEVANCE_THRESHOLD
    
    reason = "Question is relevant" if passed else f"Question is off-topic (score {max_score:.2f} < {RELEVANCE_THRESHOLD})"
    
    return {
        "passed": passed,
        "score": float(max_score),
        "reason": reason
    }

def check_faithfulness(answer: str, context: List[Document], llm) -> Dict[str, Any]:
    """
    Sends a prompt to the LLM to verify if the answer is completely supported by context.
    Returns: passed=True if YES, False if NO.
    """
    prompt = f"""
    Context:
    {chr(10).join([doc.page_content for doc in context])}

    Answer:
    {answer}

    Is this answer fully supported by the context? Answer YES or NO and one sentence why.
    """
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    passed = content.upper().startswith("YES")
    
    return {
        "passed": passed,
        "score": "YES" if passed else "NO",
        "reason": content
    }
