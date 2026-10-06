"""
dates.py — deterministic relative-date resolution (no LLM involved).

"next Friday" only means something relative to when the message was SENT.
resolve_date_reference_impl() anchors the phrase on the message timestamp,
then compares the resolved date with today: upcoming / today / past.

Rules for weekday phrases (base = the day the message was sent):
    "Friday" / "this Friday" → the coming Friday, or the base day itself if it is a Friday
    "next Friday" / "coming Friday" → the first Friday strictly after the base day
    "last Friday" → the most recent Friday strictly before the base day
Other phrases: today/tonight, tomorrow (tmrw/tmr), day after tomorrow,
yesterday, day before yesterday, next/last/this week|month|year.  Anything
else falls back to dateparser with RELATIVE_BASE set to the message time.

Public API:
    resolve_date_reference_impl(reference_text, message_timestamp, today=None) -> dict
    find_relative_phrases(text) -> list[str]
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_MON = "|".join(m[:3] + r"[a-z]*" for m in _MONTHS)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WD = "|".join(_WEEKDAYS)

_ALIASES = [
    (re.compile(r"\b(tmrw|tmrow|tmr|tommorow|tommorrow|tomorow)\b"), "tomorrow"),
    (re.compile(r"\bupcoming\b"), "coming"),
]

_PHRASE_RE = re.compile(
    rf"\b(?:day\s+after\s+(?:tomorrow|tmrw|tmr)"
    rf"|day\s+before\s+yesterday"
    rf"|tomorrow|tmrw|tmr|yesterday"
    rf"|(?:next|this|coming|last)\s+(?:week|month|year)"
    rf"|(?:(?:next|this|coming|last)\s+)?(?:{_WD}))\b",
    re.IGNORECASE,
)


def _parse_base(message_timestamp: str) -> datetime | None:
    s = str(message_timestamp).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year, month = d.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _next_day_of_month(base: date, day: int, month: int | None, year: int | None) -> date | None:
    """The first date on/after `base` with that day (and month/year if given)."""
    if not 1 <= day <= 31:
        return None
    if year and month:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    for offset in range(0, 24):  # search forward up to two years
        y, mo = base.year + (base.month - 1 + offset) // 12, (base.month - 1 + offset) % 12 + 1
        if month and mo != month:
            continue
        if day <= calendar.monthrange(y, mo)[1]:
            candidate = date(y, mo, day)
            if candidate >= base:
                return candidate
    return None


def _resolve(text: str, base: date) -> date | None:
    """Rule-based resolution; returns None when no rule applies."""
    if text in ("today", "tonight"):
        return base
    if text == "tomorrow":
        return base + timedelta(days=1)
    if text == "day after tomorrow":
        return base + timedelta(days=2)
    if text == "yesterday":
        return base - timedelta(days=1)
    if text == "day before yesterday":
        return base - timedelta(days=2)

    m = re.fullmatch(rf"(?:(next|this|coming|last)\s+)?(?:on\s+)?({_WD})", text)
    if m:
        modifier, target = m.group(1), _WEEKDAYS.index(m.group(2))
        if modifier == "last":
            back = (base.weekday() - target) % 7 or 7
            return base - timedelta(days=back)
        ahead = (target - base.weekday()) % 7
        if modifier in ("next", "coming") and ahead == 0:
            ahead = 7
        return base + timedelta(days=ahead)

    # Day of month: "28th", "the 5th", "28 july", "july 28", "28th and 29th" (first date wins).
    m = re.match(rf"^(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(?:of\s+)?({_MON}))?(?:\s+(\d{{4}}))?\b", text) or re.match(
        rf"^({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", text)
    if m and not re.match(r"^\d{1,2}\s*(?:am|pm|:|days?|weeks?|months?|hours?|mins?)", text):
        if m.group(1).isdigit():
            day, month_name, year = int(m.group(1)), m.group(2), m.group(3)
        else:
            month_name, day, year = m.group(1), int(m.group(2)), m.group(3)
        month = next((i + 1 for i, name in enumerate(_MONTHS) if month_name and name.startswith(month_name[:3])), None)
        return _next_day_of_month(base, day, month, int(year) if year else None)

    m = re.fullmatch(r"(next|this|coming|last)\s+(week|month|year)", text)
    if m:
        step = {"next": 1, "coming": 1, "this": 0, "last": -1}[m.group(1)]
        unit = m.group(2)
        if unit == "week":
            return base + timedelta(weeks=step)
        if unit == "month":
            return _add_months(base, step)
        return _add_months(base, 12 * step)
    return None


def _dateparser_fallback(text: str, base_dt: datetime) -> date | None:
    import dateparser  # local import: heavy module, rarely needed

    parsed = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": base_dt,
            "PREFER_DATES_FROM": "past" if re.search(r"\b(last|ago|before)\b", text) else "future",
            "DATE_ORDER": "DMY",
        },
    )
    if not parsed:
        return None
    # Chat phrases refer to the near past/future; anything wildly far away is a
    # misparse (e.g. "28th and 29th" → year 2029), so refuse rather than guess.
    if abs((parsed.date() - base_dt.date()).days) > 366:
        return None
    return parsed.date()


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def resolve_date_reference_impl(
    reference_text: str,
    message_timestamp: str,
    today: date | None = None,
) -> dict:
    """Resolve a relative date phrase anchored at the message's send time.

    Returns {reference_text, resolved_date, weekday, reference_timestamp,
    today, status, delta_days, delta_description} or {status: "error", error}.
    """
    base_dt = _parse_base(message_timestamp)
    if base_dt is None:
        return {"status": "error", "error": f"Could not read message timestamp '{message_timestamp}'"}

    text = re.sub(r"\s+", " ", reference_text.strip().lower()).strip(" .,!?")
    for pattern, replacement in _ALIASES:
        text = pattern.sub(replacement, text)
    text = re.sub(r"^on\s+", "", text)

    resolved = _resolve(text, base_dt.date()) or _dateparser_fallback(text, base_dt)
    if resolved is None:
        return {"status": "error", "error": f"Could not resolve relative date '{reference_text}'"}

    today = today or date.today()
    delta = (resolved - today).days
    if delta > 0:
        status, desc = "upcoming", f"{_plural(delta, 'day')} from now"
    elif delta < 0:
        status, desc = "past", f"{_plural(-delta, 'day')} ago"
    else:
        status, desc = "today", "today"

    return {
        "reference_text": reference_text.strip(),
        "resolved_date": resolved.isoformat(),
        "weekday": resolved.strftime("%A"),
        "reference_timestamp": base_dt.strftime("%Y-%m-%d %H:%M"),
        "today": today.isoformat(),
        "status": status,
        "delta_days": delta,
        "delta_description": desc,
    }


def find_relative_phrases(text: str) -> list[str]:
    """Return the distinct relative-date phrases in a message, in order."""
    seen: list[str] = []
    for m in _PHRASE_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", m.group(0).strip())
        if phrase.lower() not in (p.lower() for p in seen):
            seen.append(phrase)
    return seen
