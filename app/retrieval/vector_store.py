"""
vector_store.py — LangChain Chroma wrapper + HuggingFace local embeddings.

Two Chroma collections (both under the same persist_directory):
    "sessions"  — one document per SessionChunk
    "messages"  — one document per individual ParsedMessage

Both use HuggingFaceEmbeddings with all-MiniLM-L6-v2 (local, no paid API).

Module-level singletons ensure the embedding model is loaded only once per
process (the first call to get_embeddings() triggers the ~90 MB download on
the first ever run; subsequent starts use the local cache).

Public API:
    get_embeddings()          -> HuggingFaceEmbeddings
    get_sessions_store()      -> Chroma
    get_messages_store()      -> Chroma
    upsert_chunks(chunks)     -> None
    upsert_messages(msgs, hash_to_chunk) -> None
    collection_counts()       -> dict[str, int]   (for debug / persistence tests)
"""

from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.core.models import ParsedMessage, SessionChunk

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None
_sessions_store: Chroma | None = None
_messages_store: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and lazily initialise) the shared embedding model instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    return _embeddings


def get_sessions_store() -> Chroma:
    """Return the 'sessions' Chroma collection (lazily initialised)."""
    global _sessions_store
    if _sessions_store is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _sessions_store = Chroma(
            collection_name="sessions",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _sessions_store


def get_messages_store() -> Chroma:
    """Return the 'messages' Chroma collection (lazily initialised)."""
    global _messages_store
    if _messages_store is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _messages_store = Chroma(
            collection_name="messages",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _messages_store


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[SessionChunk]) -> None:
    """Embed and store SessionChunk objects in the 'sessions' collection.

    Uses chunk_id as the Chroma document ID — safe to re-call since dedup
    guarantees these chunks haven't been stored before.
    """
    if not chunks:
        return

    # Deduplicate by chunk_id before sending to Chroma.
    # (Defensive: chunker should already produce unique IDs, but guard anyway.)
    seen_ids: set[str] = set()
    unique_chunks: list[SessionChunk] = []
    for chunk in chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            unique_chunks.append(chunk)

    docs = [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "start_ts": chunk.start_ts.isoformat(),
                "end_ts": chunk.end_ts.isoformat(),
                "participants": ",".join(chunk.participants),
                "message_count": chunk.message_count,
            },
        )
        for chunk in unique_chunks
    ]
    get_sessions_store().add_documents(docs, ids=[c.chunk_id for c in unique_chunks])


def upsert_messages(
    messages: list[ParsedMessage],
    hash_to_chunk: dict[str, str],
) -> None:
    """Embed and store individual ParsedMessage objects in 'messages' collection.

    Args:
        messages:       New (deduplicated) messages to store.
        hash_to_chunk:  Mapping content_hash → chunk_id so each message doc
                        carries its parent session's ID as metadata.

    Note: WhatsApp exports occasionally contain duplicate messages (identical
    timestamp + sender + text).  We deduplicate by content_hash here so Chroma
    never sees repeated IDs in a single batch.
    """
    if not messages:
        return

    # Deduplicate by content_hash — keeps the first occurrence.
    seen_ids: set[str] = set()
    unique_messages: list[ParsedMessage] = []
    for m in messages:
        if m.content_hash not in seen_ids:
            seen_ids.add(m.content_hash)
            unique_messages.append(m)

    docs = [
        Document(
            page_content=(
                f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}"
            ),
            metadata={
                "content_hash": m.content_hash,
                "sender": m.sender,
                "timestamp": m.timestamp.isoformat(),
                "chunk_id": hash_to_chunk.get(m.content_hash, ""),
            },
        )
        for m in unique_messages
    ]
    get_messages_store().add_documents(docs, ids=[m.content_hash for m in unique_messages])


# ---------------------------------------------------------------------------
# Debug / persistence verification
# ---------------------------------------------------------------------------

def collection_counts() -> dict[str, int]:
    """Return document counts for both collections without loading embeddings.

    Uses the chromadb PersistentClient directly so this is fast and safe to
    call before the embedding model is initialised.  Returns 0 for a
    collection that does not yet exist.
    """
    import chromadb  # local import to keep top-level imports lean

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    counts: dict[str, int] = {}
    for name in ("sessions", "messages"):
        try:
            counts[name] = client.get_collection(name).count()
        except Exception:
            counts[name] = 0
    return counts
