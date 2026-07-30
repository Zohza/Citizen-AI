"""Configuration loader for the Citizen AI backend.

Loads settings from environment variables (with .env file support via python-dotenv).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq (LLM provider) ──
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
)
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── NVIDIA NIM (embeddings only) ──
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "nvidia/nv-embedqa-e5-v5",
)

# ── Vector store ──
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./data/chroma",
)

# ── Memory ──
MEMORY_DB_PATH: str = os.getenv(
    "MEMORY_DB_PATH",
    "./data/memory.sqlite",
)

# ── Ingestion ──
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
