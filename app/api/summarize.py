"""
summarize.py — date-range summaries with a decisions/plans section.
Stage 6: POST /summarize endpoint.
"""
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.tools import date_range_lookup, resolve_relative_dates_in_context
from app.config import settings

router = APIRouter()


class SummarizeRequest(BaseModel):
    start_date: str = Field(..., description="ISO date string YYYY-MM-DD")
    end_date: str = Field(..., description="ISO date string YYYY-MM-DD")


class DecisionPlanItem(BaseModel):
    description: str = Field(..., description="Description of the decision, plan, or agreement made")
    timestamp: str = Field(..., description="Timestamp of the message where this was decided/planned")
    sender: str = Field(..., description="Sender who proposed or agreed to the decision/plan")


class SummarizeResponse(BaseModel):
    narrative_summary: str = Field(
        ..., description="Comprehensive narrative paragraph summarizing the conversation in the date range"
    )
    decisions_and_plans: list[DecisionPlanItem] = Field(
        default_factory=list,
        description="Separate list of decisions, plans, or scheduled events",
    )


def _get_llm():
    """Return configured LLM instance (Groq or Gemini)."""
    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    elif settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
    else:
        raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in settings.")


_SUMMARIZE_SYSTEM = """\
You are an expert assistant summarizing WhatsApp chat conversations for a specific date range.

You will be given the retrieved chat messages from that date range along with any resolved relative date references.

STRICT INSTRUCTIONS:
1. Provide a clear, comprehensive narrative summary of what was discussed during this period in `narrative_summary`.
2. Extract all concrete decisions, plans, agreements, scheduled events, or commitments into a separate list `decisions_and_plans`.
3. Each item in `decisions_and_plans` MUST include:
   - `description`: a concise explanation of what was decided, planned, or agreed.
   - `timestamp`: the exact timestamp string from the chat message (e.g., '2026-05-09 17:37').
   - `sender`: the name of the participant who stated/proposed/agreed to it.
4. If no clear decisions or plans were made in the retrieved text, return an empty list for `decisions_and_plans`.
5. NEVER fabricate facts, dates, or senders not explicitly present in the provided context.
"""


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_range(request: SummarizeRequest):
    """Summarize WhatsApp chat conversations for a specific date range.

    Returns:
        {
            "narrative_summary": str,
            "decisions_and_plans": [
                {
                    "description": str,
                    "timestamp": str,
                    "sender": str
                }
            ]
        }
    """
    if not request.start_date.strip() or not request.end_date.strip():
        raise HTTPException(status_code=400, detail="start_date and end_date cannot be empty.")

    # 1. Retrieve session chunks for the date range
    raw_results = date_range_lookup.invoke({
        "start_date": request.start_date.strip(),
        "end_date": request.end_date.strip(),
    })

    # Check if results indicate no sessions found or an error
    try:
        parsed_json = json.loads(raw_results)
        if isinstance(parsed_json, dict) and ("message" in parsed_json or "error" in parsed_json):
            return SummarizeResponse(
                narrative_summary=f"No chat activity found between {request.start_date} and {request.end_date}.",
                decisions_and_plans=[],
            )
    except Exception:
        pass

    # 2. Enrich with relative date resolutions
    date_info = resolve_relative_dates_in_context(raw_results)

    # 3. Call LLM with structured output
    try:
        llm = _get_llm().with_structured_output(SummarizeResponse)
        prompt = (
            f"{_SUMMARIZE_SYSTEM}\n\n"
            f"Date Range: {request.start_date} to {request.end_date}\n\n"
            f"Retrieved Context:\n{raw_results}{date_info}\n\n"
            "Generate the structured summary now."
        )
        response: SummarizeResponse = llm.invoke(prompt)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization error: {exc}") from exc
