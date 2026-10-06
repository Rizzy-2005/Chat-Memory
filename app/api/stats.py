"""
stats.py — GET /stats: what is currently stored (drives the sidebar and empty states).
"""
from fastapi import APIRouter

from app.agent.llm import model_name
from app.core import dedup
from app.retrieval.vector_store import collection_counts

router = APIRouter()


@router.get("/stats")
def get_stats():
    """Message / session counts, participants and the chat's date span.
    Does not load the embedding model, so it is fast even on a cold start."""
    stats = dedup.get_stats()
    return {
        "total_messages": stats["total_messages"],
        "total_sessions": collection_counts()["sessions"],
        "participants": stats["participants"],
        "first_message": stats["first_message"],
        "last_message": stats["last_message"],
        "llm": model_name(),
    }
