"""
tests/test_graph.py — unit tests for app/agent/graph.py and app/agent/tools.py

Strategy: mock the LangChain LLM (_get_llm) so tests run instantly without
hitting the Gemini API or requiring Chroma/SQLite to be populated.

Covers:
    - router picks plain_rag_lookup for a general question
    - router picks participant_filtered_lookup when a name is mentioned
    - router picks message_pinpoint for a "find the message" question
    - router picks date_range_lookup when a date is mentioned
    - router falls back to plain_rag_lookup when LLM returns no tool call
    - run_tool_node correctly dispatches to the chosen tool
    - composer_node returns the expected structure
    - composer_node handles a 'not found' scenario correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.graph import (
    AgentState,
    ComposerOutput,
    Citation,
    composer_node,
    router_node,
    run_tool_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tool_call(name: str, args: dict) -> MagicMock:
    """Build a mock LLM response that contains one tool call."""
    response = MagicMock()
    response.tool_calls = [{"name": name, "args": args}]
    return response


def _mock_no_tool_call() -> MagicMock:
    """Build a mock LLM response with NO tool call (fallback case)."""
    response = MagicMock()
    response.tool_calls = []
    return response


# ---------------------------------------------------------------------------
# Router node tests
# ---------------------------------------------------------------------------

class TestRouterNode:
    """Test that the router picks the expected tool for each question type."""

    def _run_router(self, llm_response: MagicMock, question: str) -> dict:
        """Patch _get_llm and run the router_node."""
        with patch("app.agent.graph._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke.return_value = llm_response
            mock_get_llm.return_value = mock_llm

            state: AgentState = {"question": question}
            return router_node(state)

    def test_general_question_picks_plain_rag(self):
        """Generic question → plain_rag_lookup."""
        result = self._run_router(
            _mock_tool_call("plain_rag_lookup", {"question": "what did we discuss?"}),
            "what did we discuss?",
        )
        assert result["tool_name"] == "plain_rag_lookup"
        assert "question" in result["tool_args"]

    def test_sender_question_picks_participant_filter(self):
        """Question mentioning a specific person → participant_filtered_lookup."""
        result = self._run_router(
            _mock_tool_call(
                "participant_filtered_lookup",
                {"question": "what did Arathi say?", "sender": "Arathi TKM CSE"},
            ),
            "what did Arathi say about TCS?",
        )
        assert result["tool_name"] == "participant_filtered_lookup"
        assert result["tool_args"].get("sender") == "Arathi TKM CSE"

    def test_pinpoint_question_picks_message_pinpoint(self):
        """'Find the exact message' type → message_pinpoint."""
        result = self._run_router(
            _mock_tool_call("message_pinpoint", {"question": "find the message about NQT"}),
            "find the specific message about NQT registration",
        )
        assert result["tool_name"] == "message_pinpoint"

    def test_date_question_picks_date_range(self):
        """Question mentioning a date range → date_range_lookup."""
        result = self._run_router(
            _mock_tool_call(
                "date_range_lookup",
                {"start_date": "2026-05-09", "end_date": "2026-05-12"},
            ),
            "what happened between 9 May and 12 May?",
        )
        assert result["tool_name"] == "date_range_lookup"
        assert "start_date" in result["tool_args"]
        assert "end_date" in result["tool_args"]

    def test_fallback_when_no_tool_call(self):
        """If the LLM returns no tool call, fall back to plain_rag_lookup."""
        result = self._run_router(
            _mock_no_tool_call(),
            "what is going on?",
        )
        assert result["tool_name"] == "plain_rag_lookup"
        assert result["tool_args"] == {"question": "what is going on?"}


# ---------------------------------------------------------------------------
# run_tool_node tests
# ---------------------------------------------------------------------------

class TestRunToolNode:

    def test_dispatches_to_correct_tool(self):
        """run_tool_node calls the right tool and stores its output in raw_results.

        Must patch _TOOL_MAP (not the name plain_rag_lookup) because run_tool_node
        looks up tools from the dict that was already built at import time.
        """
        fake_result = '[{"text": "Hello world", "metadata": {}}]'
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = fake_result

        with patch.dict("app.agent.graph._TOOL_MAP", {"plain_rag_lookup": mock_tool}):
            state: AgentState = {
                "question": "test?",
                "tool_name": "plain_rag_lookup",
                "tool_args": {"question": "test?"},
            }
            result = run_tool_node(state)

        assert result["raw_results"] == fake_result
        mock_tool.invoke.assert_called_once_with({"question": "test?"})

    def test_unknown_tool_falls_back_to_plain_rag(self):
        """An unrecognised tool_name falls back to plain_rag_lookup without crashing."""
        with patch("app.agent.graph.plain_rag_lookup") as mock_tool:
            mock_tool.invoke.return_value = "[]"

            state: AgentState = {
                "question": "anything?",
                "tool_name": "nonexistent_tool",
                "tool_args": {"question": "anything?"},
            }
            result = run_tool_node(state)

        assert "raw_results" in result


# ---------------------------------------------------------------------------
# composer_node tests
# ---------------------------------------------------------------------------

class TestComposerNode:

    def _make_composer_output(self, answer: str, citations: list[Citation]) -> ComposerOutput:
        return ComposerOutput(answer=answer, citations=citations, mode_used="plain_rag_lookup")

    def test_returns_answer_and_citations(self):
        """Composer returns answer, citation list, and mode_used."""
        composer_result = self._make_composer_output(
            answer="The TCS NQT registration deadline is 15 May 2026.",
            citations=[Citation(timestamp="2026-05-09T17:37:00", sender="Arathi TKM CSE")],
        )
        with patch("app.agent.graph._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = composer_result
            mock_get_llm.return_value = mock_llm

            state: AgentState = {
                "question": "When is TCS NQT registration deadline?",
                "tool_name": "plain_rag_lookup",
                "raw_results": '[{"text": "[2026-05-09 17:37] Arathi TKM CSE: TCS NQT last date 15 May", "metadata": {}}]',
            }
            result = composer_node(state)

        assert result["answer"] == "The TCS NQT registration deadline is 15 May 2026."
        assert len(result["citations"]) == 1
        assert result["citations"][0]["sender"] == "Arathi TKM CSE"
        assert result["mode_used"] == "plain_rag_lookup"

    def test_not_found_returns_empty_citations(self):
        """When nothing relevant is found, answer is explicit and citations are empty."""
        not_found = self._make_composer_output(
            answer="I couldn't find anything in the chat about that.",
            citations=[],
        )
        with patch("app.agent.graph._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = not_found
            mock_get_llm.return_value = mock_llm

            state: AgentState = {
                "question": "What is the capital of France?",
                "tool_name": "plain_rag_lookup",
                "raw_results": '{"message": "No relevant sessions found."}',
            }
            result = composer_node(state)

        assert "couldn't find" in result["answer"].lower()
        assert result["citations"] == []

    def test_composer_handles_llm_exception_gracefully(self):
        """If the LLM throws, composer returns a safe error message — no crash."""
        with patch("app.agent.graph._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API error")
            mock_get_llm.return_value = mock_llm

            state: AgentState = {
                "question": "test",
                "tool_name": "plain_rag_lookup",
                "raw_results": "[]",
            }
            result = composer_node(state)

        assert "Error" in result["answer"]
        assert result["citations"] == []
