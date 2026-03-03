# Intelligent Document Assistant (RAG)
A Retrieval-Augmented Generation (RAG) application that lets you upload documents and chat with an AI assistant that answers questions based on your document content.
Built with **FastAPI**, **LangChain**, **ChromaDB**, **Groq LLM**, and **Gradio**.
## Features
- **Document Upload & Ingestion** — Upload PDFs or text files to build a searchable knowledge base
- **RAG-Powered Chat** — Ask questions and get answers grounded in your documents
- **Session Management** — Persistent sessions with conversation history
- **Guardrails** — Input/output validation to keep responses safe and on-topic
- **Evaluation** — Built-in RAG evaluation using Ragas metrics
- **LangServe** — Exposes a stateless RAG chain at `/chain` for programmatic access
- **Gradio UI** — User-friendly web interface for uploading docs and chatting
## Project Structure
```
RAG_Project/
├── main.py                  # FastAPI app entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── app/
│   ├── api/                 # API route definitions
│   ├── pipelines/           # Core RAG pipeline (ingestion, retrieval, generation, memory)
│   ├── guardrails/          # Input/output guardrails
│   └── evaluation/          # RAG evaluation module
├── ui/
│   └── app.py               # Gradio web UI
├── tests/                   # Unit & integration tests
└── data/                    # Runtime data (gitignored)
    ├── uploads/             # Uploaded documents
    ├── sessions/            # Session persistence
    └── vectorstore/         # ChromaDB vector store
```
## Getting Started
### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)
### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RAG_Project.git
cd RAG_Project
# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
# Install dependencies
pip install -r requirements.txt
```
> **⚡ GPU Acceleration (Optional but Recommended)**
> The embedding model runs on CPU by default, which can be slow for large documents. If you have an NVIDIA GPU, you can significantly speed up embedding by installing the CUDA-enabled version of PyTorch.
> Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and use the configurator to get the right install command for your system, then run it **before** installing the other requirements. For example:
> ```bash
> # Example for CUDA 12.1 — check the PyTorch site for your exact version
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> You can verify GPU detection afterwards with `python -c "import torch; print(torch.cuda.is_available())"`.
### Configuration
```bash
# Copy the environment template and add your API key
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```
### Running the App
**Backend API:**
```bash
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```
**Gradio UI:**
```bash
python ui/app.py
# UI available at http://localhost:7860
```
## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document |
| POST | `/chat` | Send a chat message |
| GET | `/sessions` | List active sessions |
| POST | `/chain/invoke` | LangServe stateless RAG |
## Running Tests
```bash
pytest
```
## License
This project is provided as-is for educational and personal use.
