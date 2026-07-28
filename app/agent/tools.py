"""
tools.py — four LangChain @tool-decorated retrieval functions.

plain_rag_lookup          — semantic search over 'sessions' collection, top 5
participant_filtered_lookup — semantic search + participant post-filter, top 5
message_pinpoint          — semantic search over 'messages' collection, top 3
date_range_lookup         — metadata filter on 'sessions' by date range (no vector search)

All tools return a JSON string so the LLM composer can parse and cite them.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.retrieval.vector_store import get_messages_store, get_sessions_store


def _docs_to_json(docs) -> str:
    """Serialize a list of LangChain Documents to a JSON string."""
    return json.dumps(
        [{"text": d.page_content, "metadata": d.metadata} for d in docs],
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------------------------
# Tool 1 — General semantic search over sessions
# ---------------------------------------------------------------------------

@tool
def plain_rag_lookup(question: str) -> str:
    """Search the WhatsApp chat sessions for content semantically related to the question.

    Use this for general questions that do not mention a specific sender or a
    specific date/time period.  Returns the top 5 most relevant session chunks.
    """
    store = get_sessions_store()
    docs = store.similarity_search(question, k=5)
    if not docs:
        return json.dumps({"message": "No relevant sessions found."})
    return _docs_to_json(docs)


# ---------------------------------------------------------------------------
# Tool 2 — Participant-filtered semantic search
# ---------------------------------------------------------------------------

@tool
def participant_filtered_lookup(question: str, sender: str) -> str:
    """Search the WhatsApp chat for sessions that involve a specific participant.

    Use this when the question asks about what a specific person said or did.
    'sender' should be the contact name as it appears in the chat.

    NOTE: Chroma does not support $contains substring filtering on string
    metadata fields.  We therefore fetch the top-20 sessions by semantic
    similarity and post-filter in Python to keep only chunks whose 'participants'
    field contains the requested sender (case-insensitive).
    """
    store = get_sessions_store()
    # Fetch a wider net to account for the post-filter loss
    docs = store.similarity_search(question, k=20)
    filtered = [
        d for d in docs
        if sender.lower() in d.metadata.get("participants", "").lower()
    ]
    if not filtered:
        return json.dumps(
            {"message": f"No sessions found involving participant '{sender}'."}
        )
    return _docs_to_json(filtered[:5])


# ---------------------------------------------------------------------------
# Tool 3 — Message-level pinpoint search
# ---------------------------------------------------------------------------

@tool
def message_pinpoint(question: str) -> str:
    """Find the single most precise individual WhatsApp message relevant to the question.

    Use this when the user is looking for one specific message, quote, or fact
    rather than a whole conversation thread.  Returns the top 3 matching messages
    from the 'messages' collection with their exact timestamp and sender.
    """
    store = get_messages_store()
    docs = store.similarity_search(question, k=3)
    if not docs:
        return json.dumps({"message": "No matching messages found."})
    return _docs_to_json(docs)


# ---------------------------------------------------------------------------
# Tool 4 — Date-range session retrieval
# ---------------------------------------------------------------------------

@tool
def date_range_lookup(start_date: str, end_date: str) -> str:
    """Retrieve all WhatsApp conversation sessions that started within a date range.

    Use this when the question mentions specific dates, days, or a time period
    (e.g. 'last Monday', 'between 9 May and 12 May').

    Args:
        start_date: ISO date string, YYYY-MM-DD (inclusive).
        end_date:   ISO date string, YYYY-MM-DD (inclusive).

    Uses Chroma metadata filtering on 'start_ts' (ISO datetime strings are
    lexicographically comparable, so $gte / $lte work correctly).
    """
    store = get_sessions_store()
    try:
        # Access the underlying chromadb Collection directly for a pure
        # metadata-filter fetch (no embedding / similarity score needed here).
        results = store._collection.get(
            where={
                "$and": [
                    {"start_ts": {"$gte": start_date}},
                    {"start_ts": {"$lte": end_date + "T23:59:59"}},
                ]
            },
            include=["documents", "metadatas"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents:
            return json.dumps(
                {"message": f"No sessions found between {start_date} and {end_date}."}
            )
        combined = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(documents, metadatas)
        ]
        return json.dumps(combined, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool 5 — Relative Date Resolution
# ---------------------------------------------------------------------------

import re
from datetime import datetime
import dateparser


def resolve_date_reference_impl(reference_text: str, message_timestamp: str) -> dict:
    """Pure deterministic helper for resolving relative date references."""
    # Parse base timestamp from the message
    base_dt = None
    if isinstance(message_timestamp, str):
        # Try standard formats
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                base_dt = datetime.strptime(message_timestamp.strip(), fmt)
                break
            except ValueError:
                continue

    if base_dt is None:
        try:
            base_dt = datetime.fromisoformat(str(message_timestamp).strip())
        except Exception:
            base_dt = dateparser.parse(str(message_timestamp)) or datetime.now()

    # Parse relative phrase with RELATIVE_BASE = base_dt
    clean_text = reference_text.strip()
    prefer = "past" if re.search(r"\blast\b", clean_text, re.IGNORECASE) else "future"
    parsed_text = re.sub(r"\b(next|this|last)\b\s*", "", clean_text, flags=re.IGNORECASE)

    resolved_dt = dateparser.parse(
        parsed_text if parsed_text else clean_text,
        settings={"RELATIVE_BASE": base_dt, "PREFER_DATES_FROM": prefer},
    )
    if not resolved_dt:
        resolved_dt = dateparser.parse(
            clean_text,
            settings={"RELATIVE_BASE": base_dt, "PREFER_DATES_FROM": prefer},
        )

    if not resolved_dt:
        return {
            "status": "error",
            "error": f"Could not resolve relative date '{reference_text}'",
        }

    now = datetime.now()
    today_date = now.date()
    resolved_date = resolved_dt.date()
    delta_days = (resolved_date - today_date).days

    if delta_days > 0:
        status = "upcoming"
        delta_description = f"{delta_days} day{'s' if delta_days > 1 else ''} from now"
    elif delta_days < 0:
        status = "past"
        abs_days = abs(delta_days)
        delta_description = f"{abs_days} day{'s' if abs_days > 1 else ''} ago"
    else:
        status = "today"
        delta_description = "today"

    return {
        "resolved_date": resolved_date.strftime("%Y-%m-%d"),
        "reference_timestamp": base_dt.strftime("%Y-%m-%d %H:%M"),
        "today": today_date.strftime("%Y-%m-%d"),
        "status": status,
        "delta_description": delta_description,
    }


_RELATIVE_DATE_PAT = re.compile(
    r"\b(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)\b|\b(tomorrow|yesterday)\b",
    re.IGNORECASE,
)


def resolve_relative_dates_in_context(raw_results: str) -> str:
    """Scan retrieved context for candidate relative-date phrases and resolve them."""
    if not raw_results:
        return ""

    resolved_notes: list[str] = []

    lines = raw_results.splitlines()
    for line in lines:
        matches = _RELATIVE_DATE_PAT.findall(line)
        if not matches:
            continue

        # Look for timestamp in format [YYYY-MM-DD HH:MM] or ISO string
        ts_match = re.search(r"\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]?", line)
        ts_str = ts_match.group(1) if ts_match else None

        if not ts_str:
            iso_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
            ts_str = iso_match.group(1) if iso_match else None

        if not ts_str:
            continue

        for m in matches:
            phrase = " ".join([part for part in m if part]).strip()
            if not phrase:
                continue

            res = resolve_date_reference_impl(phrase, ts_str)
            if res.get("status") != "error":
                note = (
                    f"- In message at '{ts_str}', relative phrase '{phrase}' resolves to "
                    f"absolute date {res['resolved_date']} (Status: {res['status']}, {res['delta_description']} relative to today {res['today']})."
                )
                if note not in resolved_notes:
                    resolved_notes.append(note)

    if resolved_notes:
        return "\n\nResolved Date Information:\n" + "\n".join(resolved_notes)
    return ""


@tool
def resolve_date_reference(reference_text: str, message_timestamp: str) -> dict:
    """Resolve a relative date phrase (e.g. 'next Friday', 'tomorrow', 'last Tuesday')
    relative to a specific message timestamp into an absolute calendar date (YYYY-MM-DD)
    and compute whether it is upcoming, today, or in the past relative to today's date.

    Args:
        reference_text: The relative time phrase, e.g. 'next Friday'.
        message_timestamp: The timestamp of the message, e.g. '2026-05-09 17:37'.

    Returns:
        dict with keys: resolved_date, reference_timestamp, today, status, delta_description.
    """
    return resolve_date_reference_impl(reference_text, message_timestamp)


