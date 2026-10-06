"""
Shared fixtures.

`store` points Chroma + SQLite at a fresh temp directory and swaps the real
embedding model for a deterministic fake one, so storage/retrieval tests run
fast, offline, and never touch app/data.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import settings
from app.core.models import ParsedMessage
from app.retrieval import vector_store


def make_msg(ts: str, sender: str, text: str) -> ParsedMessage:
    t = datetime.fromisoformat(ts)
    h = hashlib.sha256(f"{t.isoformat()}|{sender}|{text}".encode()).hexdigest()
    return ParsedMessage(timestamp=t, sender=sender, text=text, content_hash=h)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "dedup.db"))
    vector_store.reset_singletons()
    monkeypatch.setattr(vector_store, "_embeddings", DeterministicFakeEmbedding(size=32))
    yield vector_store
    vector_store.reset_singletons()
