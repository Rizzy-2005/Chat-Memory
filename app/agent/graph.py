"""
graph.py — LangGraph StateGraph: router → run_tool → composer.

Flow:
    1. router   — one Gemini call with all 4 tools bound; decides which tool and args.
    2. run_tool — executes the chosen tool, stores raw JSON results in state.
    3. composer — second Gemini call that reads the raw results and emits a
                  structured { answer, citations, mode_used } response.

Hard rules enforced in the composer system prompt:
  - Every claim must cite a retrieved message (timestamp + sender).
  - If nothing relevant is in the retrieved context → "I couldn't find anything
    in the chat about that." with empty citations.  Never fabricate.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.tools import (
    date_range_lookup,
    message_pinpoint,
    participant_filtered_lookup,
    plain_rag_lookup,
)
from app.config import settings

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Typed state bag threaded through every node in the graph.

    Only 'question' is required at graph invocation time.
    Each subsequent node fills in its own keys.
    """
    question: str          # the user's natural-language question
    tool_name: str         # set by router: which tool was selected
    tool_args: dict[str, Any]   # set by router: arguments to pass to the tool
    raw_results: str       # set by run_tool: JSON string returned by the tool
    answer: str            # set by composer: final prose answer
    citations: list[dict]  # set by composer: [{timestamp, sender}, ...]
    mode_used: str         # set by composer: tool name echoed for transparency


# ---------------------------------------------------------------------------
# Pydantic schema for structured composer output
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    timestamp: str
    sender: str


class ComposerOutput(BaseModel):
    answer: str
    citations: list[Citation]
    mode_used: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOOLS = [plain_rag_lookup, participant_filtered_lookup, message_pinpoint, date_range_lookup]

_TOOL_MAP = {
    "plain_rag_lookup": plain_rag_lookup,
    "participant_filtered_lookup": participant_filtered_lookup,
    "message_pinpoint": message_pinpoint,
    "date_range_lookup": date_range_lookup,
}

_ROUTER_SYSTEM = """\
You are the router for a WhatsApp chat question-answering system.
Given a user question, call exactly one of the four available tools:

  plain_rag_lookup            — general questions without a specific sender or date
  participant_filtered_lookup — questions about what a specific person said/did
  message_pinpoint            — looking for one precise quote or specific message
  date_range_lookup           — questions about events on specific dates or periods

Always call exactly one tool.  Do NOT attempt to answer the question yourself."""

_COMPOSER_SYSTEM = """\
You are answering questions about a WhatsApp group chat.
You have been given retrieved context from the chat.

STRICT RULES — follow every one without exception:
1. ONLY use information explicitly present in the retrieved context below.
2. Every factual claim in your answer MUST be backed by a citation listing the exact
   timestamp and sender from the retrieved context.
3. If the retrieved context does not contain information relevant to the question,
   set answer to exactly: "I couldn't find anything in the chat about that."
   and return an empty citations list.
4. NEVER fabricate facts, names, dates, or quotes.
5. NEVER use general knowledge — only what is in the context.
6. For mode_used, return the tool name that retrieved the context."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    """Return a Groq LLM instance (free tier, high RPM, supports tool-calling)."""
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


from typing import Any, Literal, TypedDict


class RouterSelection(BaseModel):
    tool_name: Literal[
        "plain_rag_lookup",
        "participant_filtered_lookup",
        "message_pinpoint",
        "date_range_lookup",
    ]
    question: str = ""
    sender: str = ""
    start_date: str = ""
    end_date: str = ""


def router_node(state: AgentState) -> dict:
    """Decide which retrieval tool to call and with what arguments using structured output."""
    llm = _get_llm().with_structured_output(RouterSelection)
    prompt = (
        f"{_ROUTER_SYSTEM}\n\n"
        f"Question: {state['question']}\n\n"
        "Select the appropriate tool and supply its arguments."
    )
    try:
        sel: RouterSelection = llm.invoke(prompt)
        tool_args: dict[str, Any] = {}
        tool_name = sel.tool_name

        if tool_name == "date_range_lookup" and (not sel.start_date or not sel.end_date):
            # If model selected date_range_lookup without valid dates, default to plain_rag_lookup
            tool_name = "plain_rag_lookup"

        if tool_name in ("plain_rag_lookup", "message_pinpoint"):
            tool_args = {"question": sel.question or state["question"]}
        elif tool_name == "participant_filtered_lookup":
            tool_args = {
                "question": sel.question or state["question"],
                "sender": sel.sender,
            }
        elif tool_name == "date_range_lookup":
            tool_args = {
                "start_date": sel.start_date,
                "end_date": sel.end_date,
            }
        return {"tool_name": tool_name, "tool_args": tool_args}
    except Exception:
        return {
            "tool_name": "plain_rag_lookup",
            "tool_args": {"question": state["question"]},
        }


def run_tool_node(state: AgentState) -> dict:
    """Execute the tool the router selected and store the raw JSON result."""
    tool_fn = _TOOL_MAP.get(state.get("tool_name", "plain_rag_lookup"), plain_rag_lookup)
    result = tool_fn.invoke(state.get("tool_args", {"question": state["question"]}))
    return {"raw_results": str(result)}


def composer_node(state: AgentState) -> dict:
    """Turn raw retrieval results into a grounded, cited answer."""
    llm = _get_llm().with_structured_output(ComposerOutput)
    prompt = (
        f"{_COMPOSER_SYSTEM}\n\n"
        f"Tool used: {state.get('tool_name', 'unknown')}\n\n"
        f"Retrieved context:\n{state.get('raw_results', 'No context retrieved.')}\n\n"
        f"Question: {state['question']}\n\n"
        "Produce your structured answer now."
    )
    try:
        result: ComposerOutput = llm.invoke(prompt)
        return {
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "mode_used": state.get("tool_name", "unknown"),
        }
    except Exception as exc:
        return {
            "answer": f"Error generating answer: {exc}",
            "citations": [],
            "mode_used": state.get("tool_name", "unknown"),
        }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("run_tool", run_tool_node)
    g.add_node("composer", composer_node)

    g.set_entry_point("router")
    g.add_edge("router", "run_tool")
    g.add_edge("run_tool", "composer")
    g.add_edge("composer", END)

    return g.compile()


# Module-level singleton — built once per process
_graph = None


def get_graph():
    """Return (and lazily build) the compiled LangGraph agent."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
