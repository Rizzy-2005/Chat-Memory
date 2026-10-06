"""
health.py — liveness endpoint used by Docker, Render and the frontend's wake-up check.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
