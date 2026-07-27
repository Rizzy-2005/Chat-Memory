"""
models.py — shared Pydantic / dataclass models used across the app.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParsedMessage:
    """One WhatsApp message after parsing and hashing."""

    timestamp: datetime   # parsed from the export header line
    sender: str           # contact name exactly as it appears in the export
    text: str             # full message body (multi-line messages joined with '\n')
    content_hash: str     # sha256(f"{timestamp.isoformat()}|{sender}|{text}")
