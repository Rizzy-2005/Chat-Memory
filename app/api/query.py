"""
query.py — handles natural-language questions over the stored chat.
Stage 1 stub: returns 'not implemented' until the agent graph is wired in.
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/query")
async def query_chat():
    # TODO (Stage 4+): accept QueryRequest, run LangGraph agent, return cited answer.
    return {"status": "not implemented"}
