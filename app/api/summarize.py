"""
summarize.py — date-range summaries with a decisions/plans section.
Stage 1 stub: returns 'not implemented' until the summary flow is wired in.
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/summarize")
async def summarize_range():
    # TODO (Stage 5+): accept date range + optional participant, return structured summary.
    return {"status": "not implemented"}
