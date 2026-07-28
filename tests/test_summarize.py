"""
tests/test_summarize.py — unit tests for POST /summarize endpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.summarize import DecisionPlanItem, SummarizeResponse
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.summarize import DecisionPlanItem, SummarizeResponse, router

app_for_testing = FastAPI()
app_for_testing.include_router(router)
client = TestClient(app_for_testing)


class TestSummarizeEndpoint:

    def test_summarize_empty_dates_400(self):
        """Requesting /summarize with blank dates returns 400 Bad Request."""
        res = client.post("/summarize", json={"start_date": "", "end_date": ""})
        assert res.status_code == 400
        assert "cannot be empty" in res.json()["detail"]

    @patch("app.api.summarize.date_range_lookup")
    def test_summarize_no_sessions_found(self, mock_lookup):
        """When date_range_lookup finds no sessions, returns polite empty response."""
        mock_lookup.invoke.return_value = '{"message": "No sessions found between 2026-01-01 and 2026-01-05."}'

        res = client.post(
            "/summarize",
            json={"start_date": "2026-01-01", "end_date": "2026-01-05"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "No chat activity found" in data["narrative_summary"]
        assert data["decisions_and_plans"] == []

    @patch("app.api.summarize._get_llm")
    @patch("app.api.summarize.date_range_lookup")
    def test_summarize_success_flow(self, mock_lookup, mock_get_llm):
        """When date_range_lookup finds sessions, returns narrative and decisions/plans list."""
        mock_lookup.invoke.return_value = (
            '[{"text": "[2026-05-09 17:37] Arathi: let\'s meet next Friday for placement prep.", '
            '"metadata": {"start_ts": "2026-05-09T17:37:00"}}]'
        )

        mock_llm_instance = MagicMock()
        mock_structured_llm = MagicMock()
        mock_get_llm.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        mock_structured_llm.invoke.return_value = SummarizeResponse(
            narrative_summary="Arathi proposed a meeting for placement preparation.",
            decisions_and_plans=[
                DecisionPlanItem(
                    description="Placement prep meeting scheduled for next Friday",
                    timestamp="2026-05-09 17:37",
                    sender="Arathi",
                )
            ],
        )

        res = client.post(
            "/summarize",
            json={"start_date": "2026-05-01", "end_date": "2026-05-31"},
        )
        assert res.status_code == 200
        data = res.json()

        assert "narrative_summary" in data
        assert "decisions_and_plans" in data
        assert len(data["decisions_and_plans"]) == 1
        assert data["decisions_and_plans"][0]["sender"] == "Arathi"
        assert data["decisions_and_plans"][0]["description"] == "Placement prep meeting scheduled for next Friday"
