"""
vector_store.py — Chroma storage + local HuggingFace embeddings.

Two Chroma collections share one on-disk PersistentClient:
    "sessions"  — one document per SessionChunk (conversation context)
    "messages"  — one document per ParsedMessage (pinpoint search + citations)

Embeddings: all-MiniLM-L6-v2, run locally on CPU (no key, no cost), with
normalised vectors so distance behaves like cosine similarity.

Chroma metadata only supports scalar values and numeric range operators, so:
    * timestamps are stored twice: ISO string (readable) + epoch int (filterable)
    * session participants are stored as a comma string (readable) plus one
      boolean flag per participant, `p_<hash>` (filterable)

Public API:
    get_embeddings(), get_sessions_store(), get_messages_store()
    upsert_chunks(chunks), upsert_messages(messages, hash_to_chunk)
    search_sessions(query, senders, start, end, k) -> list[dict]
    search_messages(query, senders, start, end, k) -> list[dict]
    messages_in_range(start, end)                  -> list[dict]
    collection_counts(), ensure_data_version()
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import os
import re
import threading
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.core import dedup
from app.core.models import ParsedMessage, SessionChunk

log = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Bump when the stored metadata layout changes: old data is wiped on startup
# (the user re-uploads), instead of silently breaking filters.
DATA_VERSION = "2"
_BATCH = 256
_TS_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ", re.MULTILINE)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_embeddings: HuggingFaceEmbeddings | None = None
_client = None
_stores: dict[str, Chroma] = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and lazily load) the shared embedding model."""
    global _embeddings
    with _lock:
        if _embeddings is None:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        return _embeddings


def get_client():
    """Return the shared chromadb PersistentClient."""
    global _client
    with _lock:
        if _client is None:
            import chromadb

            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        return _client


def _store(name: str) -> Chroma:
    with _lock:
        if name not in _stores:
            _stores[name] = Chroma(
                client=get_client(),
                collection_name=name,
                embedding_function=get_embeddings(),
            )
        return _stores[name]


def get_sessions_store() -> Chroma:
    return _store("sessions")


def get_messages_store() -> Chroma:
    return _store("messages")


def reset_singletons() -> None:
    """Forget cached clients/stores (tests point settings at a temp dir)."""
    global _client
    with _lock:
        _client = None
        _stores.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_epoch(dt: datetime) -> int:
    """Naive export time → int seconds (treated as UTC; only ordering matters)."""
    return calendar.timegm(dt.timetuple())


def participant_key(name: str) -> str:
    return "p_" + hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:12]


def _and(conds: list[dict]) -> dict | None:
    conds = [c for c in conds if c]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _or(conds: list[dict]) -> dict | None:
    conds = [c for c in conds if c]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$or": conds}


def _range_bounds(start: datetime | None, end: datetime | None) -> tuple[int | None, int | None]:
    return (to_epoch(start) if start else None, to_epoch(end) if end else None)


def _message_from(meta: dict) -> dict:
    return {
        "content_hash": meta.get("content_hash", ""),
        "timestamp": meta.get("timestamp", ""),
        "sender": meta.get("sender", ""),
        "text": meta.get("text", ""),
        "chunk_id": meta.get("chunk_id", ""),
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[SessionChunk]) -> None:
    """Embed and store SessionChunks (idempotent: chunk_id is the Chroma ID)."""
    unique = list({c.chunk_id: c for c in chunks}.values())
    store = get_sessions_store()
    for i in range(0, len(unique), _BATCH):
        batch = unique[i : i + _BATCH]
        docs = []
        for c in batch:
            meta = {
                "chunk_id": c.chunk_id,
                "start_ts": c.start_ts.isoformat(),
                "end_ts": c.end_ts.isoformat(),
                "start_epoch": to_epoch(c.start_ts),
                "end_epoch": to_epoch(c.end_ts),
                "participants": ",".join(c.participants),
                "message_count": c.message_count,
            }
            meta.update({participant_key(p): True for p in c.participants})
            # Embed "Sender: text" lines; the timestamps would only add noise.
            docs.append(Document(page_content=_TS_PREFIX_RE.sub("", c.text), metadata=meta))
        store.add_documents(docs, ids=[c.chunk_id for c in batch])


def upsert_messages(messages: list[ParsedMessage], hash_to_chunk: dict[str, str]) -> None:
    """Embed and store individual messages (idempotent: content_hash is the ID)."""
    unique = list({m.content_hash: m for m in messages}.values())
    store = get_messages_store()
    for i in range(0, len(unique), _BATCH):
        batch = unique[i : i + _BATCH]
        docs = [
            Document(
                page_content=f"{m.sender}: {m.text}",
                metadata={
                    "content_hash": m.content_hash,
                    "sender": m.sender,
                    "text": m.text,
                    "timestamp": m.timestamp.isoformat(),
                    "ts_epoch": to_epoch(m.timestamp),
                    "chunk_id": hash_to_chunk.get(m.content_hash, ""),
                },
            )
            for m in batch
        ]
        store.add_documents(docs, ids=[m.content_hash for m in batch])


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def messages_for_chunks(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """chunk_id → its messages in chronological order."""
    if not chunk_ids:
        return {}
    res = get_messages_store()._collection.get(
        where={"chunk_id": {"$in": list(chunk_ids)}}, include=["metadatas"]
    )
    grouped: dict[str, list[dict]] = {cid: [] for cid in chunk_ids}
    for meta in res.get("metadatas") or []:
        grouped.setdefault(meta.get("chunk_id", ""), []).append(_message_from(meta))
    for msgs in grouped.values():
        msgs.sort(key=lambda m: m["timestamp"])
    return grouped


def search_sessions(
    query: str,
    senders: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    k: int = 5,
) -> list[dict]:
    """Semantic search over sessions with optional participant / date filters.

    A session matches the date range if it overlaps it.  Returns sessions
    (best first) each with its full message list."""
    lo, hi = _range_bounds(start, end)
    where = _and([
        _or([{participant_key(s): True} for s in senders or []]),
        {"end_epoch": {"$gte": lo}} if lo is not None else None,
        {"start_epoch": {"$lte": hi}} if hi is not None else None,
    ])
    store = get_sessions_store()
    if store._collection.count() == 0:
        return []
    docs = store.similarity_search_with_score(query, k=k, filter=where)
    chunk_ids = [d.metadata["chunk_id"] for d, _ in docs]
    msgs = messages_for_chunks(chunk_ids)
    return [
        {
            "chunk_id": d.metadata["chunk_id"],
            "start_ts": d.metadata.get("start_ts"),
            "end_ts": d.metadata.get("end_ts"),
            "participants": d.metadata.get("participants", "").split(","),
            "score": round(float(score), 4),
            "messages": msgs.get(d.metadata["chunk_id"], []),
        }
        for d, score in docs
    ]


def search_messages(
    query: str,
    senders: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    k: int = 5,
) -> list[dict]:
    """Semantic search over individual messages with optional filters."""
    lo, hi = _range_bounds(start, end)
    where = _and([
        _or([{"sender": s} for s in senders or []]),
        {"ts_epoch": {"$gte": lo}} if lo is not None else None,
        {"ts_epoch": {"$lte": hi}} if hi is not None else None,
    ])
    store = get_messages_store()
    if store._collection.count() == 0:
        return []
    docs = store.similarity_search_with_score(query, k=k, filter=where)
    return [{**_message_from(d.metadata), "score": round(float(s), 4)} for d, s in docs]


def message_window(message: dict, before: int = 1, after: int = 1, minutes: int = 30) -> list[dict]:
    """The message plus its chronological neighbours (within `minutes`),
    regardless of which chunk they were stored in."""
    t = to_epoch(datetime.fromisoformat(message["timestamp"]))
    res = get_messages_store()._collection.get(
        where={"$and": [{"ts_epoch": {"$gte": t - minutes * 60}}, {"ts_epoch": {"$lte": t + minutes * 60}}]},
        include=["metadatas"],
    )
    near = sorted((_message_from(m) for m in res.get("metadatas") or []), key=lambda m: (m["timestamp"], m["content_hash"]))
    idx = next((i for i, m in enumerate(near) if m["content_hash"] == message["content_hash"]), None)
    if idx is None:
        return [message]
    return near[max(idx - before, 0) : idx + after + 1]


def messages_in_range(start: datetime, end: datetime) -> list[dict]:
    """Every stored message in [start, end], chronological (no vector search)."""
    lo, hi = _range_bounds(start, end)
    res = get_messages_store()._collection.get(
        where={"$and": [{"ts_epoch": {"$gte": lo}}, {"ts_epoch": {"$lte": hi}}]},
        include=["metadatas"],
    )
    msgs = [_message_from(m) for m in res.get("metadatas") or []]
    msgs.sort(key=lambda m: m["timestamp"])
    return msgs


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def collection_counts() -> dict[str, int]:
    """Document counts per collection, without loading the embedding model."""
    client = get_client()
    counts: dict[str, int] = {}
    for name in ("sessions", "messages"):
        try:
            counts[name] = client.get_collection(name).count()
        except Exception:
            counts[name] = 0
    return counts


def reset_collections() -> None:
    client = get_client()
    with _lock:
        for name in ("sessions", "messages"):
            try:
                client.delete_collection(name)
            except Exception:
                pass
        _stores.clear()


def ensure_data_version() -> None:
    """Wipe stored data written by an older, incompatible version of the app."""
    if dedup.get_meta("data_version") == DATA_VERSION:
        return
    log.warning("Stored data format changed (→ v%s); resetting the index. Re-upload your chat.", DATA_VERSION)
    reset_collections()
    dedup.reset_index()
    dedup.set_meta("data_version", DATA_VERSION)
