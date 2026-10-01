import os

from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present

DATA_DIR = os.getenv("DATA_DIR", "data")

# Embeddings (local, via fastembed). EMBED_DIM must match the model's output size.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
EMBED_CACHE = os.getenv("EMBED_CACHE", ".cache/fastembed")

# Chunking (characters) and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# LLM: any OpenAI-compatible endpoint
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None  # None = OpenAI's default
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")