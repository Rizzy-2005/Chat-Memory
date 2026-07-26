"""
health.py — liveness / readiness endpoint.
Returns a real 200 OK so docker-compose healthchecks and curl smoke tests work.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}
