"""
summarize.py — POST /summarize: narrative + decisions/plans for a date range.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.summarize import summarize_range
from app.api.errors import llm_errors

router = APIRouter()


class SummarizeRequest(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD (inclusive)")
    end_date: str = Field(..., description="YYYY-MM-DD (inclusive)")


class DecisionPlanItem(BaseModel):
    description: str
    timestamp: str
    sender: str
    excerpt: str = ""


class SummarizeResponse(BaseModel):
    narrative_summary: str
    decisions_and_plans: list[DecisionPlanItem] = Field(default_factory=list)
    message_count: int = 0
    found: bool = True


def _parse(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a date in YYYY-MM-DD format.") from exc


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if not request.start_date.strip() or not request.end_date.strip():
        raise HTTPException(status_code=400, detail="start_date and end_date cannot be empty.")
    start, end = _parse(request.start_date, "start_date"), _parse(request.end_date, "end_date")
    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be on or before end_date.")
    with llm_errors("Summarizing"):
        return summarize_range(start, end)
