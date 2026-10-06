"""
context.py — helpers shared by the Q&A composer and the summarizer.

The LLM never invents a citation: retrieved messages are numbered [1], [2], …
in the prompt, the model returns the numbers it relied on, and the real
timestamp / sender / text are looked up from those numbers here.
"""

from __future__ import annotations

from app.core.dates import find_relative_phrases, resolve_date_reference_impl

EXCERPT_CHARS = 300


def dedupe_messages(messages: list[dict]) -> list[dict]:
    """Drop repeats (by content_hash), keeping first-retrieved order."""
    seen: set[str] = set()
    out = []
    for m in messages:
        key = m.get("content_hash") or f"{m.get('timestamp')}|{m.get('sender')}|{m.get('text')}"
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def fit_budget(messages: list[dict], max_chars: int) -> list[dict]:
    """Keep messages (in the given priority order) until the char budget is used."""
    out, used = [], 0
    for m in messages:
        size = len(m.get("text", "")) + len(m.get("sender", "")) + 24
        if out and used + size > max_chars:
            break
        out.append(m)
        used += size
    return out


def numbered_lines(messages: list[dict]) -> str:
    """'[n] YYYY-MM-DD HH:MM | Sender: text' lines; blank line between days."""
    lines, prev_day = [], None
    for i, m in enumerate(messages, 1):
        ts = m["timestamp"][:16].replace("T", " ")
        if prev_day and ts[:10] != prev_day:
            lines.append("")
        prev_day = ts[:10]
        text = m["text"].replace("\n", "\n    ")
        lines.append(f"[{i}] {ts} | {m['sender']}: {text}")
    return "\n".join(lines)


def auto_date_resolutions(messages: list[dict], limit: int = 15) -> dict[int, list[dict]]:
    """Deterministically resolve relative-date phrases in each numbered message.
    Returns {message_number: [resolution, ...]}."""
    out: dict[int, list[dict]] = {}
    count = 0
    for i, m in enumerate(messages, 1):
        for phrase in find_relative_phrases(m["text"]):
            res = resolve_date_reference_impl(phrase, m["timestamp"])
            if res.get("status") == "error":
                continue
            out.setdefault(i, []).append(res)
            count += 1
            if count >= limit:
                return out
    return out


def date_notes_text(resolutions: dict[int, list[dict]], extra: list[dict] | None = None) -> str:
    lines = []
    for i, items in sorted(resolutions.items()):
        for r in items:
            lines.append(
                f"- In [{i}] (sent {r['reference_timestamp'][:10]}), \"{r['reference_text']}\" = "
                f"{r['weekday']} {r['resolved_date']} ({r['status']}, {r['delta_description']})."
            )
    for r in extra or []:
        lines.append(
            f"- \"{r['reference_text']}\" sent {r['reference_timestamp']} = "
            f"{r['weekday']} {r['resolved_date']} ({r['status']}, {r['delta_description']})."
        )
    if not lines:
        return ""
    today = (next(iter(resolutions.values()), [None])[0] or (extra or [{}])[0]).get("today", "")
    return (
        f"Resolved dates (computed by a deterministic tool; today is {today}; trust these, "
        "do not redo the arithmetic):\n" + "\n".join(lines)
    )


def citation(m: dict) -> dict:
    text = m["text"]
    excerpt = text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1].rstrip() + "…"
    return {"timestamp": m["timestamp"], "sender": m["sender"], "excerpt": excerpt}


def valid_ids(ids, n: int) -> list[int]:
    """Keep in-range, unique ids in their given order."""
    out: list[int] = []
    for i in ids or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out
