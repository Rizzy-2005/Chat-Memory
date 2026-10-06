"""
query.py — POST /query: a natural-language question through the LangGraph agent.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import run_query
from app.api.errors import llm_errors

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)


@router.post("/query")
def query_chat(request: QueryRequest):
    """Returns {answer, citations[{timestamp, sender, excerpt}], mode_used, filters,
    tool_calls, found, date_resolutions, trace}.  When nothing relevant is found,
    found is false, citations is empty and answer is the fixed not-found sentence."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    with llm_errors("Answering"):
        return run_query(question)
