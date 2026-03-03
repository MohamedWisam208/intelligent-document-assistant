from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    """Initializes and returns the ChatGroq model instance."""
    return ChatGroq(model_name="llama-3.1-8b-instant", max_retries=2)

def get_rag_prompt():
    """Returns the strict conversational RAG prompt template."""
    # Based on Phase 2 specifications
    template = """
    You are a helpful and intelligent document assistant. Use the following context to answer the user's question.
    If the answer is not contained in the context, say "I don't know based on the provided document." Do not try to make up an answer.
    Always cite your sources by referencing the page numbers if available.

    Context:
    {context}

    History:
    {history}

    Question: 
    {question}

    Answer:
    """
    return PromptTemplate(
        template=template,
        input_variables=["context", "history", "question"]
    )

def create_generation_chain():
    """Creates the base generation chain from Prompt -> LLM -> Parser."""
    prompt = get_rag_prompt()
    llm = get_llm()
    parser = StrOutputParser()
    
    return prompt | llm | parser

