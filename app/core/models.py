"""
models.py — shared Pydantic / dataclass models used across the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ParsedMessage:
    """One WhatsApp message after parsing and hashing."""

    timestamp: datetime    # parsed from the export header line
    sender: str            # contact name exactly as it appears in the export
    text: str              # full message body (multi-line messages joined with '\n')
    content_hash: str      # sha256(f"{timestamp.isoformat()}|{sender}|{text}")


@dataclass
class SessionChunk:
    """A burst of conversation where no two consecutive messages are more than
    SESSION_GAP_HOURS apart.  This is the unit stored in the 'sessions' Chroma
    collection."""

    chunk_id: str               # sha256 of sorted(message_hashes) — deterministic
    text: str                   # readable "[YYYY-MM-DD HH:MM] Sender: body" lines joined by '\n'
    participants: list[str]     # sorted list of unique senders in this session
    start_ts: datetime          # timestamp of the first message
    end_ts: datetime            # timestamp of the last message
    message_count: int          # number of messages in this session
    message_hashes: list[str]   # content_hash of every message, in chronological order
