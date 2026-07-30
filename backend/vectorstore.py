"""Persistent ChromaDB vector store for the Citizen AI knowledge base.

Creates / loads a single collection named ``citizen_ai_kb``.
"""

from __future__ import annotations

import chromadb
from chromadb import Collection
from typing import Optional

from config import CHROMA_PERSIST_DIR

_client: chromadb.PersistentClient | None = None
_collection: Collection | None = None


def get_client() -> chromadb.PersistentClient:
    """Return the singleton persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_collection() -> Collection:
    """Return (creating if necessary) the ``citizen_ai_kb`` collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name="citizen_ai_kb",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
