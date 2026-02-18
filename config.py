import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
