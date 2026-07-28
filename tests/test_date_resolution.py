"""
tests/test_date_resolution.py — unit tests for relative date resolution tool and composer integration.

Covers:
    - resolve_date_reference tool with relative phrase (e.g. 'next Friday')
    - relative_base set to message_timestamp
    - status ('past', 'upcoming', 'today') and delta_description computation
    - date resolution extraction from context lines in graph composer
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.agent.tools import (
    resolve_date_reference,
    resolve_date_reference_impl,
    resolve_relative_dates_in_context,
)


class TestResolveDateReference:

    def test_next_friday_resolution(self):
        """'next Friday' relative to Saturday 2026-05-09 17:37 resolves to 2026-05-15."""
        res = resolve_date_reference_impl(
            reference_text="next Friday",
            message_timestamp="2026-05-09 17:37",
        )

        assert res["resolved_date"] == "2026-05-15"
        assert "2026-05-09" in res["reference_timestamp"]
        assert "status" in res
        assert "delta_description" in res
        assert res["status"] in ("past", "upcoming", "today")

    def test_tomorrow_resolution(self):
        """'tomorrow' relative to 2026-05-09 10:00 resolves to 2026-05-10."""
        res = resolve_date_reference_impl(
            reference_text="tomorrow",
            message_timestamp="2026-05-09 10:00",
        )
        assert res["resolved_date"] == "2026-05-10"

    def test_past_status_computation(self):
        """A past resolved date relative to today's real date reports status 'past'."""
        # 2000-01-01 + 1 day = 2000-01-02, which is in the past
        res = resolve_date_reference_impl(
            reference_text="tomorrow",
            message_timestamp="2000-01-01 12:00",
        )
        assert res["resolved_date"] == "2000-01-02"
        assert res["status"] == "past"
        assert "ago" in res["delta_description"]

    def test_upcoming_status_computation(self):
        """A future resolved date relative to today reports status 'upcoming'."""
        # 2099-01-01 + 1 day = 2099-01-02, which is in the future
        res = resolve_date_reference_impl(
            reference_text="tomorrow",
            message_timestamp="2099-01-01 12:00",
        )
        assert res["resolved_date"] == "2099-01-02"
        assert res["status"] == "upcoming"
        assert "from now" in res["delta_description"]

    def test_tool_invocation(self):
        """LangChain @tool wrapper resolve_date_reference returns expected dict."""
        res = resolve_date_reference.invoke({
            "reference_text": "next Friday",
            "message_timestamp": "2026-05-09 17:37",
        })
        assert isinstance(res, dict)
        assert res["resolved_date"] == "2026-05-15"


class TestContextDateResolution:

    def test_context_resolution_extractor(self):
        """resolve_relative_dates_in_context extracts relative phrase and appends info."""
        raw = '[2026-05-09 17:37] Arathi TKM CSE: let\'s meet next Friday for placement prep.'
        info = resolve_relative_dates_in_context(raw)

        assert "Resolved Date Information:" in info
        assert "2026-05-15" in info
        assert "next Friday" in info
