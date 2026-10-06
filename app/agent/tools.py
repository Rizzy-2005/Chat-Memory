"""
tools.py — the three LangChain tools the agent can call.

    search_sessions          — semantic search over conversation sessions, with
                               optional sender and date-range filters (combined)
    message_pinpoint         — semantic search over single messages
    resolve_date_reference   — deterministic relative-date resolution

Each tool uses response_format="content_and_artifact":
    content  → compact text the agent model reads to decide what to do next
    artifact → structured data (messages / date result) the composer uses to
               build citations, so citations always come from real messages.

The docstrings are what the model reads to choose a tool — keep them precise.
"""

from __future__ import annotations

import difflib
import json
from datetime import date, datetime, time
from typing import Optional

from langchain_core.tools import tool

from app.core import dedup
from app.core.dates import resolve_date_reference_impl
from app.retrieval import vector_store

RETRIEVAL_TOOLS = ("search_sessions", "message_pinpoint")
_NULLISH = {"", "null", "none", "n/a", "na", "any", "all", "anyone", "everyone", "unknown"}


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def clean_arg(value) -> str | None:
    """Models sometimes send 'null' / '' / 'None' for an omitted optional arg."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in _NULLISH else s


def resolve_senders(name: str, participants: list[str]) -> list[str]:
    """Map a name from the question onto the exact participant names.

    exact (case-insensitive) → substring → all words present → fuzzy.
    May return several names (e.g. 'Arathi' when two Arathis exist)."""
    n = " ".join(name.lower().split())
    if not n or not participants:
        return []
    lower = {p: p.lower() for p in participants}
    exact = [p for p, pl in lower.items() if pl == n]
    if exact:
        return exact
    contains = [p for p, pl in lower.items() if n in pl or pl in n]
    if contains:
        return contains
    words = n.split()
    all_words = [p for p, pl in lower.items() if all(w in pl.split() for w in words)]
    if all_words:
        return all_words
    # Fuzzy: compare against full names and against each name's first word.
    first_words = {p: pl.split()[0] for p, pl in lower.items() if pl.split()}
    close = difflib.get_close_matches(n, list(lower.values()), n=3, cutoff=0.8)
    close_first = difflib.get_close_matches(words[0], list(first_words.values()), n=3, cutoff=0.8)
    return [p for p in participants if lower[p] in close or first_words.get(p) in close_first]


def parse_date_arg(value: str | None, end_of_day: bool = False) -> datetime | None:
    """'YYYY-MM-DD' (or anything dateparser understands, day-first) → datetime."""
    s = clean_arg(value)
    if not s:
        return None
    d: date | None = None
    try:
        d = date.fromisoformat(s[:10])
    except ValueError:
        import dateparser

        parsed = dateparser.parse(s, settings={"DATE_ORDER": "DMY"})
        d = parsed.date() if parsed else None
    if d is None:
        return None
    return datetime.combine(d, time(23, 59, 59) if end_of_day else time(0, 0))


_PREVIEW_CHARS = 220


def _line(m: dict) -> str:
    """One-line preview for the agent (the composer later sees full text)."""
    text = " ".join(m["text"].split())
    if len(text) > _PREVIEW_CHARS:
        text = text[: _PREVIEW_CHARS - 1] + "…"
    return f"[{m['timestamp'][:16].replace('T', ' ')}] {m['sender']}: {text}"


def _participants() -> list[str]:
    return dedup.get_stats()["participants"]


# ---------------------------------------------------------------------------
# Tool 1 — session search (plain / participant / date-range in one tool)
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def search_sessions(
    question: str,
    sender: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Search the chat's conversation sessions (bursts of back-and-forth messages).

    Use this for most questions: general questions, "what did X say about Y",
    "what was decided in June", "what happened between 1 and 15 July".
    Filters can be combined. Fill them ONLY if the question names them:
      sender: a participant's name (use the exact name from the participant list).
      start_date / end_date: YYYY-MM-DD, inclusive. For a single day use the same date twice.
    Returns the best matching sessions with every message's timestamp and sender.
    """
    filters = {"sender": None, "start_date": None, "end_date": None}
    senders: list[str] = []
    sender = clean_arg(sender)
    if sender:
        participants = _participants()
        senders = resolve_senders(sender, participants)
        if not senders:
            note = (
                f"No participant matches '{sender}'. Participants are: {', '.join(participants)}. "
                "Retry with one of these exact names, or without a sender filter."
            )
            return note, {"kind": "search_sessions", "filters": {**filters, "sender": sender}, "messages": [], "note": note}
        filters["sender"] = ", ".join(senders)

    start = parse_date_arg(start_date)
    end = parse_date_arg(end_date, end_of_day=True)
    if start and end and start > end:
        start, end = datetime.combine(end.date(), time(0, 0)), datetime.combine(start.date(), time(23, 59, 59))
    filters["start_date"] = start.date().isoformat() if start else None
    filters["end_date"] = end.date().isoformat() if end else None

    sessions = vector_store.search_sessions(question, senders or None, start, end, k=5)
    messages = [m for s in sessions for m in s["messages"]]
    artifact = {"kind": "search_sessions", "filters": filters, "messages": messages}
    if not messages:
        return "No sessions matched this search and these filters.", artifact

    blocks = []
    for i, s in enumerate(sessions, 1):
        header = f"Session {i} ({(s['start_ts'] or '')[:10]}; {', '.join(s['participants'])}):"
        blocks.append(header + "\n" + "\n".join(_line(m) for m in s["messages"]))
    return "\n\n".join(blocks), artifact


# ---------------------------------------------------------------------------
# Tool 2 — message pinpoint
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def message_pinpoint(question: str, sender: Optional[str] = None):
    """Find ONE specific message. Use only when the user wants to locate a
    particular message or exact quote, e.g. "find the message where Priya shared
    the link" or "who said 'see you at 6'". Optionally restrict to a sender
    (exact participant name). Returns the top matching individual messages.
    """
    senders: list[str] = []
    sender = clean_arg(sender)
    if sender:
        senders = resolve_senders(sender, _participants())
    hits = vector_store.search_messages(question, senders or None, k=5)
    filters = {"sender": ", ".join(senders) or None, "start_date": None, "end_date": None}
    if not hits:
        return "No matching messages found.", {"kind": "message_pinpoint", "filters": filters, "messages": []}

    # A message often only makes sense with its neighbour (a bare link followed
    # by "Cognizant drive 👆"), so each hit comes with the message before/after.
    blocks, messages = [], []
    for i, hit in enumerate(hits, 1):
        window = vector_store.message_window(hit)
        lines = [("> " if m["content_hash"] == hit["content_hash"] else "  ") + _line(m) for m in window]
        blocks.append(f"Match {i}:\n" + "\n".join(lines))
        messages.extend([hit] + [m for m in window if m["content_hash"] != hit["content_hash"]])
    return "\n\n".join(blocks), {"kind": "message_pinpoint", "filters": filters, "messages": messages}


# ---------------------------------------------------------------------------
# Tool 3 — deterministic date resolution
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def resolve_date_reference(reference_text: str, message_timestamp: str):
    """Turn a relative date like "next Friday" or "tomorrow" into a calendar date,
    relative to when the message containing it was SENT, then compare it with
    today: status is upcoming, today or past. Never do this arithmetic yourself.
      reference_text: the phrase exactly as written, e.g. "next Friday".
      message_timestamp: the message's timestamp, e.g. "2026-05-09 17:37".
    """
    res = resolve_date_reference_impl(reference_text, message_timestamp)
    return json.dumps(res), {"kind": "date", "result": res}


TOOLS = [search_sessions, message_pinpoint, resolve_date_reference]
TOOL_MAP = {t.name: t for t in TOOLS}
