"""
tests/test_date_resolution.py — deterministic relative-date resolution
(app/core/dates.py), the LangChain tool wrapper, and context annotation.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.agent.context import auto_date_resolutions
from app.agent.tools import resolve_date_reference
from app.core.dates import find_relative_phrases, resolve_date_reference_impl

SAT = "2026-05-09 17:37"  # Saturday
TODAY = date(2026, 10, 6)


@pytest.mark.parametrize(
    "phrase, base, expected",
    [
        ("next Friday", SAT, "2026-05-15"),
        ("Friday", SAT, "2026-05-15"),
        ("this Saturday", SAT, "2026-05-09"),       # same weekday → today
        ("next Saturday", SAT, "2026-05-16"),       # same weekday → a week later
        ("coming Monday", SAT, "2026-05-11"),
        ("last Friday", SAT, "2026-05-08"),
        ("last Saturday", SAT, "2026-05-02"),
        ("tomorrow", SAT, "2026-05-10"),
        ("tmrw", SAT, "2026-05-10"),
        ("day after tomorrow", SAT, "2026-05-11"),
        ("yesterday", SAT, "2026-05-08"),
        ("next week", SAT, "2026-05-16"),
        ("next month", "2026-01-31 10:00", "2026-02-28"),
        ("in 3 days", SAT, "2026-05-12"),            # dateparser fallback
        ("tomorrow", "2026-05-09T23:50:00", "2026-05-10"),
    ],
)
def test_resolution(phrase, base, expected):
    assert resolve_date_reference_impl(phrase, base, today=TODAY)["resolved_date"] == expected


def test_status_and_delta():
    past = resolve_date_reference_impl("next Friday", SAT, today=TODAY)
    assert past["status"] == "past"
    assert past["delta_description"] == "144 days ago"
    assert past["weekday"] == "Friday"

    upcoming = resolve_date_reference_impl("tomorrow", "2026-10-06 09:00", today=TODAY)
    assert upcoming["status"] == "upcoming"
    assert upcoming["delta_description"] == "1 day from now"

    today = resolve_date_reference_impl("tomorrow", "2026-10-05 09:00", today=TODAY)
    assert today["status"] == "today"


def test_bad_inputs_return_error():
    assert resolve_date_reference_impl("next Friday", "not a timestamp")["status"] == "error"
    assert resolve_date_reference_impl("blorptastic", SAT)["status"] == "error"


def test_tool_wrapper():
    out = resolve_date_reference.invoke({"reference_text": "next Friday", "message_timestamp": SAT})
    assert '"resolved_date": "2026-05-15"' in out


def test_find_relative_phrases():
    text = "Let's meet next Friday, or tmrw if possible. Last week was busy. Fridays are fine."
    assert find_relative_phrases(text) == ["next Friday", "tmrw", "Last week"]


def test_auto_resolution_uses_each_messages_own_timestamp():
    """Regression: the phrase must be anchored on the message that contains it."""
    msgs = [
        {"timestamp": "2026-05-01T10:00:00", "sender": "A", "text": "hi"},
        {"timestamp": "2026-05-09T17:37:00", "sender": "B", "text": "meet next friday"},
    ]
    res = auto_date_resolutions(msgs)
    assert list(res) == [2]
    assert res[2][0]["resolved_date"] == "2026-05-15"


@pytest.mark.parametrize(
    "phrase, base, expected",
    [
        ("28th and 29th", "2026-07-23 21:47", "2026-07-28"),  # regression: was 2029-07-28
        ("28th", "2026-07-23 21:47", "2026-07-28"),
        ("the 5th", "2026-07-23 21:47", "2026-08-05"),        # next occurrence
        ("15 May 2026", "2026-05-09 10:00", "2026-05-15"),
        ("14th july 2026", "2026-06-02 21:43", "2026-07-14"),
        ("july 14", "2026-06-02 21:43", "2026-07-14"),
        ("june 4th", "2026-05-25 20:36", "2026-06-04"),
    ],
)
def test_day_of_month(phrase, base, expected):
    assert resolve_date_reference_impl(phrase, base, today=TODAY)["resolved_date"] == expected


def test_far_off_misparse_refused():
    assert resolve_date_reference_impl("in 900 days", SAT, today=TODAY)["status"] == "error"
