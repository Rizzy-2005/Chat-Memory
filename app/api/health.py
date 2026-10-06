"""
health.py — liveness endpoint used by Docker, Render and the frontend's wake-up check.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/", include_in_schema=False)
def root():
    """Friendly landing response: this service is API-only (the UI is separate)."""
    return {"service": "Chat Memory API", "status": "ok", "docs": "/docs", "health": "/health", "stats": "/stats"}


@router.get("/health")
def health_check():
    return {"status": "ok"}
