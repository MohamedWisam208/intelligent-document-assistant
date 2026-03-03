import os
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.pipelines.retrieval import get_vectorstore
from app.pipelines.generation import create_generation_chain
from app.pipelines.memory import create_session, get_history, add_turn, clear_history
from app.pipelines.pipeline import invoke_chat

# Typical test questions format
# test_questions = [
#  {
#  "question": "What is the main contribution of this paper?",
#  "ground_truth": "...expected answer...",
#  },
# ]

def run_evaluation(collection_name: str, test_questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """Runs the full RAGAS evaluation suite on the provided questions."""
    
    # 1. Prepare data for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # We will need a fresh session for each to not taint history
    vectorstore = get_vectorstore()
    
    for item in test_questions:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        # Get Context
        docs = vectorstore.similarity_search(question, k=5)
        context_list = [d.page_content for d in docs]
        
        # We invoke the chain directly to skip the API/memory overhead for pure eval
        # (Or we can use invoke_chat, but RAGAS expects specific inputs)
        chain = create_generation_chain()
        context_text = "\n\n".join(context_list)
        answer = chain.invoke({
            "context": context_text,
            "history": "",
            "question": question
        })
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
    
    # 2. Build HuggingFace dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Evaluate with Ragas
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    print("Running evaluation, this might take a few minutes...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    df = result.to_pandas()
    
    return {
        "scores": result,
        "summary": df
    }

def save_evaluation_report(results: Dict[str, Any], output_path: str):
    """Saves evaluation summary DataFrame to CSV."""
    df = results.get("summary")
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Evaluation report saved to {output_path}")
