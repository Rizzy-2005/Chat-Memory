"""
debug.py — temporary debug endpoints for development and persistence verification.

GET /debug/counts  — returns document counts in both Chroma collections and
                     the number of rows in the SQLite dedup table.

Use this endpoint to verify that data survives container restarts:
    1. Upload a .zip → note the counts.
    2. docker compose restart (or docker compose down && docker compose up)
    3. GET /debug/counts again → should return identical numbers.
"""
import sqlite3

from fastapi import APIRouter

from app.config import settings
from app.retrieval.vector_store import collection_counts

router = APIRouter()


@router.get("/debug/counts")
async def debug_counts():
    """Return document counts for persistence verification.

    Does NOT require the embedding model to be loaded.
    """
    chroma = collection_counts()

    # Count rows in the SQLite dedup table (may not exist yet → 0)
    try:
        conn = sqlite3.connect(settings.sqlite_path)
        (dedup_rows,) = conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()
        conn.close()
    except Exception:
        dedup_rows = 0

    return {
        "chroma_sessions": chroma["sessions"],
        "chroma_messages": chroma["messages"],
        "sqlite_dedup_rows": dedup_rows,
    }
