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
