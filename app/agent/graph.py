"""
graph.py — the LangGraph agent: agent ⇄ tools → composer.

    START → agent ──(reply has tool_calls?)──yes──▶ tools ──▶ agent  (loop)
                  └─────────────no──────────────▶ composer ──▶ END

* agent    — the chat model with the three tools bound. Its reply either
             contains tool_calls (its own choice of tool + arguments) or not;
             routing is just checking that.  Rounds are capped at
             MAX_TOOL_ROUNDS so a confused model cannot loop forever.
* tools    — runs every requested tool and appends ToolMessages (with
             artifacts holding the real retrieved messages).
* composer — a separate structured-output call that writes the answer ONLY
             from the retrieved messages and cites them by number.  Citations
             are mapped back to real messages in code, never trusted as text.
             Nothing retrieved / nothing relevant → the fixed not-found answer.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.agent import context as ctx
from app.agent.llm import (
    LLMNotConfigured,
    LLMRateLimited,
    invoke_structured_budgeted,
    invoke_with_tools,
)
from app.agent.tools import RETRIEVAL_TOOLS, TOOL_MAP, TOOLS
from app.config import settings
from app.core import dedup

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find anything in the chat about that."


# ---------------------------------------------------------------------------
# State and structured output
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]  # running conversation with the agent
    tool_rounds: int
    forced_search: bool    # the agent model failed / skipped tools → plain search, then compose
    # filled by the composer
    answer: str
    citations: list[dict]
    found: bool
    mode_used: str
    filters: dict
    tool_calls: list[str]
    trace: list[dict]
    date_resolutions: list[dict]


class ComposerOutput(BaseModel):
    """Structured answer written from the numbered retrieved messages."""

    found: bool = Field(description="True only if the numbered messages actually answer the question.")
    answer: str = Field(description="The answer, concise and natural. No [n] markers in the text.")
    citation_ids: list[int] = Field(
        default_factory=list,
        description="Numbers of the messages that support the answer (e.g. [3, 7]). Empty if found is false.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_AGENT_PROMPT = """\
You are the retrieval planner for a question-answering app over ONE exported WhatsApp chat.
Today is {today_weekday}, {today}.
The chat has {total} messages from {first} to {last}.
Participants (exact names): {participants}

Your job is to call tools that fetch the chat messages needed to answer the user's question.
You do NOT write the final answer; a later step does that from what you retrieve.

How to choose:
- search_sessions — the default. Set `sender` only if the question is about what a specific
  person said or did (use the exact participant name). Set start_date/end_date (YYYY-MM-DD)
  only if the question names a date or period: convert "in June", "on 12 May", "last week",
  "between 1 and 15 July" to absolute dates using today's date and the chat's date span
  (a month without a year means the year in which the chat covers that month).
  Sender and dates can be combined in one call.
  Search is by meaning, not by time — so for "latest", "most recent", "last message" or
  "recently" questions, set start_date to 7 days before the chat's last message date and
  end_date to the last message date.
- message_pinpoint — only when the user wants one particular message or quote
  ("find the message where…", "who sent the link to…", "who said '…'").
- resolve_date_reference — when a retrieved message contains a relative date ("next Friday",
  "tomorrow") that matters for the question, call it with that phrase and that message's
  timestamp. Never do date arithmetic yourself.

One search is usually enough: the results are only previews, and the answer is written later
from the full messages. Search again ONLY if the results contain nothing relevant (then retry
once with different wording or fewer filters). Never repeat a search with near-identical wording.
When you have what you need, reply with the single word "done" and no tool call."""

_COMPOSER_PROMPT = """\
You answer questions about a WhatsApp chat using ONLY the numbered messages below.

Hard rules:
1. Use only what the numbered messages say. Never use general knowledge or guess.
2. Every factual claim must be supported by messages you list in citation_ids.
3. If the messages do not contain the answer, set found=false, answer exactly
   "{not_found}" and return an empty citation_ids list. If they are related and answer
   part of the question, give that partial answer (found=true) and say what is missing.
4. Never invent names, dates, numbers or quotes.
5. When a date matters, use the resolved dates provided (e.g. "Fri 15 May 2026"). If the
   question asks whether something is upcoming, still on, or past, compare the event's date
   with today ({today}) and say so explicitly.
6. Write a direct, natural answer (1–5 sentences or a short list). Mention who said what.
   Do not put [n] markers in the answer text.

Question: {question}

Retrieved messages:
{messages}

{date_notes}"""


def _agent_system_prompt() -> str:
    stats = dedup.get_stats()
    participants = stats["participants"]
    shown = ", ".join(participants[:60]) + (" …" if len(participants) > 60 else "")
    today = date.today()
    return _AGENT_PROMPT.format(
        today=today.isoformat(),
        today_weekday=today.strftime("%A"),
        total=stats["total_messages"],
        first=stats["first_message"] or "n/a",
        last=stats["last_message"] or "n/a",
        participants=shown or "(none yet)",
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _retrieval_done(messages: list[AnyMessage]) -> bool:
    return any(isinstance(m, ToolMessage) and m.name in RETRIEVAL_TOOLS for m in messages)


def _fallback_call(question: str, n: int) -> AIMessage:
    """Force a plain session search (used when the model skips tools or errors)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_sessions", "args": {"question": question}, "id": f"fallback_{n}", "type": "tool_call"}],
    )


def _call_key(call: dict) -> str:
    args = {k: " ".join(str(v).lower().split()) for k, v in (call.get("args") or {}).items() if v not in (None, "")}
    return f"{call['name']}|{sorted(args.items())}"


def _agent_view(history: list[AnyMessage]) -> list[AnyMessage]:
    """What the agent model sees.  Tool results from earlier rounds are shrunk
    to a one-line note (the composer still gets everything via artifacts),
    which keeps each call well inside free-tier tokens-per-minute limits."""
    last_ai = max((i for i, m in enumerate(history) if isinstance(m, AIMessage)), default=-1)
    view = []
    for i, m in enumerate(history):
        if isinstance(m, ToolMessage) and i < last_ai:
            n = len((m.artifact or {}).get("messages", [])) if isinstance(m.artifact, dict) else 0
            m = m.model_copy(update={"content": f"(earlier result, {n} messages — already passed to the answer writer)"})
        view.append(m)
    return view


def agent_node(state: AgentState) -> dict:
    rounds = state.get("tool_rounds", 0)
    if rounds >= settings.max_tool_rounds or state.get("forced_search"):
        return {}  # loop limit reached / fallback used → router sends us to the composer

    history = state.get("messages", [])
    try:
        reply = invoke_with_tools(TOOLS, [SystemMessage(_agent_system_prompt()), *_agent_view(history)])
    except LLMNotConfigured:
        raise
    except Exception as exc:
        if _retrieval_done(history):
            log.warning("Agent failed (%s); composing from what was already retrieved.", exc)
            return {}
        if isinstance(exc, LLMRateLimited):
            raise  # nothing retrieved yet and the composer would fail the same way
        log.warning("Agent failed on every model (%s); falling back to a plain search.", exc)
        reply = None

    if reply is not None and reply.tool_calls:
        # Drop exact repeats of earlier calls; if nothing new is asked, compose.
        done = {_call_key(c) for m in history if isinstance(m, AIMessage) for c in m.tool_calls}
        fresh = [c for c in reply.tool_calls if _call_key(c) not in done]
        if not fresh and _retrieval_done(history):
            return {}
        if len(fresh) != len(reply.tool_calls):
            reply = reply.model_copy(update={"tool_calls": fresh})

    if reply is None or (not reply.tool_calls and not _retrieval_done(history)):
        # A RAG answer must be grounded: never skip retrieval entirely.  The
        # synthetic call is never shown to a model again (Gemini rejects tool
        # calls it did not sign), so after it we go straight to the composer.
        return {"messages": [_fallback_call(state["question"], rounds)], "tool_rounds": rounds + 1, "forced_search": True}
    if reply.tool_calls:
        return {"messages": [reply], "tool_rounds": rounds + 1}
    return {"messages": [reply]}


def tools_node(state: AgentState) -> dict:
    """Run every tool call in the last AI message; errors become ToolMessages."""
    last = state["messages"][-1]
    results: list[ToolMessage] = []
    for call in last.tool_calls:
        tool = TOOL_MAP.get(call["name"])
        if tool is None:
            results.append(ToolMessage(content=f"Unknown tool '{call['name']}'.", tool_call_id=call["id"], name=call["name"], status="error"))
            continue
        try:
            results.append(tool.invoke({**call, "type": "tool_call"}))
        except Exception as exc:  # bad arguments, storage error, …
            log.warning("Tool %s failed: %s", call["name"], exc)
            results.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=call["id"], name=call["name"], status="error"))
    return {"messages": results}


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "composer"


def _collect(state: AgentState) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """From the ToolMessages: retrieved messages, agent-requested date results,
    a per-call trace for the UI, and the ordered list of tool names called."""
    retrieved: list[dict] = []
    agent_dates: list[dict] = []
    trace: list[dict] = []
    calls: list[str] = []
    args_by_id = {
        c["id"]: c["args"]
        for m in state.get("messages", [])
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    for m in state.get("messages", []):
        if not isinstance(m, ToolMessage):
            continue
        calls.append(m.name)
        art = m.artifact if isinstance(m.artifact, dict) else {}
        entry = {"tool": m.name, "args": args_by_id.get(m.tool_call_id, {}), "ok": m.status != "error"}
        if art.get("kind") in RETRIEVAL_TOOLS:
            retrieved.extend(art.get("messages", []))
            entry["filters"] = art.get("filters")
            entry["results"] = len(art.get("messages", []))
        elif art.get("kind") == "date":
            res = art.get("result", {})
            entry["result"] = res
            if res.get("status") != "error":
                agent_dates.append(res)
        trace.append(entry)
    return retrieved, agent_dates, trace, calls


def composer_node(state: AgentState) -> dict:
    retrieved, agent_dates, trace, calls = _collect(state)
    retrieval_entries = [t for t in trace if t["tool"] in RETRIEVAL_TOOLS]
    first = next((t for t in retrieval_entries if t.get("results")), retrieval_entries[0] if retrieval_entries else None)
    meta = {
        "mode_used": first["tool"] if first else "none",
        "filters": (first or {}).get("filters") or {"sender": None, "start_date": None, "end_date": None},
        "tool_calls": calls,
        "trace": trace,
    }
    not_found = {**meta, "answer": NOT_FOUND, "citations": [], "found": False, "date_resolutions": []}

    pool = ctx.dedupe_messages(retrieved)
    if not pool:
        return not_found

    def build(budget: int):
        # Numbering depends on the model's context budget, so the exact list
        # numbered in the prompt travels back with the answer.
        msgs = sorted(ctx.fit_budget(pool, budget), key=lambda m: m["timestamp"])
        dates = ctx.auto_date_resolutions(msgs)
        prompt = _COMPOSER_PROMPT.format(
            not_found=NOT_FOUND,
            today=date.today().isoformat(),
            question=state["question"],
            messages=ctx.numbered_lines(msgs),
            date_notes=ctx.date_notes_text(dates, agent_dates),
        )
        return prompt, (msgs, dates)

    out, (messages, auto_dates) = invoke_structured_budgeted(ComposerOutput, build)

    ids = ctx.valid_ids(out.citation_ids, len(messages))
    if not out.found or not ids or out.answer.strip() == NOT_FOUND:
        return not_found

    cited = sorted(ids, key=lambda i: messages[i - 1]["timestamp"])
    # Date callouts: what the agent explicitly resolved + phrases inside cited messages.
    resolutions = list(agent_dates) + [r for i in cited for r in auto_dates.get(i, [])]
    unique, seen = [], set()
    for r in resolutions:
        key = (r["reference_text"].lower(), r["reference_timestamp"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return {
        **meta,
        "answer": out.answer.strip(),
        "citations": [ctx.citation(messages[i - 1]) for i in cited],
        "found": True,
        "date_resolutions": unique,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("composer", composer_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "composer": "composer"})
    g.add_edge("tools", "agent")
    g.add_edge("composer", END)
    return g.compile()


_graph = None


def get_graph():
    """Return (and lazily build) the compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(question: str) -> dict[str, Any]:
    """Run one question through the agent and return the API-shaped result."""
    result = get_graph().invoke(
        {"question": question, "messages": [HumanMessage(question)], "tool_rounds": 0},
        config={"recursion_limit": 4 * settings.max_tool_rounds + 6},
    )
    return {
        "answer": result.get("answer", NOT_FOUND),
        "citations": result.get("citations", []),
        "mode_used": result.get("mode_used", "none"),
        "filters": result.get("filters", {"sender": None, "start_date": None, "end_date": None}),
        "tool_calls": result.get("tool_calls", []),
        "found": result.get("found", False),
        "date_resolutions": result.get("date_resolutions", []),
        "trace": result.get("trace", []),
    }
