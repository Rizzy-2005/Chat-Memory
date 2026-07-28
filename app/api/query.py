"""
query.py — handles natural-language questions over the stored chat.
Stage 4: runs the full LangGraph agent (router → run_tool → composer).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.graph import get_graph

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
async def query_chat(request: QueryRequest):
    """Run a natural-language question through the LangGraph agent.

    Returns a grounded answer with per-claim citations (timestamp + sender)
    and the retrieval mode that was used.

    If nothing relevant is found the answer field will contain an explicit
    'not found' message and citations will be empty — never hallucinated output.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    graph = get_graph()
    try:
        result = graph.invoke({"question": request.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return {
        "answer": result.get("answer", "No answer produced."),
        "citations": result.get("citations", []),
        "mode_used": result.get("mode_used", "unknown"),
    }
