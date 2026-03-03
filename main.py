from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from langserve import add_routes
from app.api.routes import router as api_router
from app.pipelines.memory import purge_expired_sessions
from app.pipelines.retrieval import get_vectorstore
from app.pipelines.generation import create_generation_chain
import os

# Load .env first so GROQ_API_KEY is available regardless of how the app is launched
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm embeddings singleton
    print("Initializing document retrieval store...")
    _ = get_vectorstore()
    
    # Pre-warm LLM singleton 
    print("Initializing generation chain...")
    _ = create_generation_chain()

    # Purge expired sessions
    print("Purging expired sessions...")
    purged_count = purge_expired_sessions()
    print(f"Purged {purged_count} expired sessions.")
    
    yield
    print("Shutting down application...")

app = FastAPI(lifespan=lifespan, title="Intelligent Document Assistant API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Router
app.include_router(api_router)

# Mount raw stateless RAG chain at /chain
# We need to make sure the groq key is set in .env or os env
try:
    chain = create_generation_chain()
    add_routes(
        app,
        chain,
        path="/chain"
    )
except Exception as e:
    print(f"Failed to mount LangServe routes: {e}")

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment!")
        
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
