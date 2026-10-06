"""
llm.py — builds the chat models and calls them with automatic failover.

Free-tier models come and go (retired, overloaded, quota hit), so every call
walks an ordered chain of candidates until one succeeds:

    LLM_PROVIDER=auto   → GEMINI_MODEL, GEMINI_FALLBACK_MODELS…, then GROQ_MODEL
    LLM_PROVIDER=gemini → Gemini models only
    LLM_PROVIDER=groq   → Groq only

(each provider only when its API key is set).  All models sit behind
LangChain's ChatModel interface, so the rest of the code never knows which
one answered.  Each candidate carries its own context budget: Groq's free tier
caps tokens per minute far lower than Gemini's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T")


class LLMNotConfigured(RuntimeError):
    """No usable API key / model is configured."""


class LLMRateLimited(RuntimeError):
    """Every candidate model hit its free-tier quota or rate limit."""


@dataclass
class Candidate:
    name: str          # e.g. "gemini:gemini-3.5-flash"
    budget: int        # max characters of chat text per call
    make: Callable[[], BaseChatModel]


def _gemini(model: str, retries: int) -> Candidate:
    def make() -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_retries=retries,
            timeout=settings.llm_timeout_s,
        )

    return Candidate(f"gemini:{model}", 30000, make)


def _groq(model: str, retries: int) -> Candidate:
    def make() -> BaseChatModel:
        from langchain_groq import ChatGroq

        extra = {}
        if model.startswith(("openai/gpt-oss", "qwen/")):
            # Reasoning models: keep thinking short (latency + tokens-per-minute cap).
            extra = {"reasoning_effort": "low"}
        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_retries=retries,
            timeout=settings.llm_timeout_s,
            **extra,
        )

    return Candidate(f"groq:{model}", 12000, make)


def candidates() -> list[Candidate]:
    choice = settings.llm_provider.strip().lower()
    specs: list[tuple[Callable[[str, int], Candidate], str]] = []
    if choice in ("auto", "gemini") and settings.gemini_api_key:
        models = [settings.gemini_model] + [
            m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()
        ]
        specs += [(_gemini, m) for m in dict.fromkeys(models)]
    if choice in ("auto", "groq") and settings.groq_api_key:
        specs.append((_groq, settings.groq_model))
    if not specs:
        raise LLMNotConfigured(
            "No LLM API key configured. Set GEMINI_API_KEY (or GROQ_API_KEY) in .env."
        )
    # Only the last model retries (waiting out a rate limit); the others fail
    # over immediately, which is much faster than sleeping on a 429/503.
    return [factory(model, 2 if i == len(specs) - 1 else 0) for i, (factory, model) in enumerate(specs)]


def model_name() -> str:
    """The preferred model (shown in /stats)."""
    try:
        return candidates()[0].name
    except LLMNotConfigured:
        return "none"


def context_budget() -> int:
    """Character budget of the preferred model (MAX_CONTEXT_CHARS overrides)."""
    if settings.max_context_chars:
        return settings.max_context_chars
    try:
        return candidates()[0].budget
    except LLMNotConfigured:
        return 12000


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _chain(exc: BaseException | None):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_config_error(exc: BaseException) -> bool:
    """Wrong key, unknown/retired model or no access: retrying will not help."""
    for e in _chain(exc):
        if isinstance(e, LLMNotConfigured):
            return True
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        text = str(e).lower()
        if status in (401, 403, 404) or any(
            s in text
            for s in ("model_not_found", "invalid api key", "api_key_invalid", "permission_denied",
                      "no longer available", "is not found for api version", "does not exist")
        ):
            return True
    return False


def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection across providers (429 / quota / rate limit)."""
    for e in _chain(exc):
        if isinstance(e, LLMRateLimited):
            return True
        name = type(e).__name__.lower()
        text = str(e).lower()
        if "ratelimit" in name or "resourceexhausted" in name:
            return True
        if getattr(e, "status_code", None) == 429 or getattr(e, "code", None) == 429:
            return True
        if any(s in text for s in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted", "resource exhausted")):
            return True
    return False


def _final_error(errors: list[tuple[str, Exception]]) -> Exception:
    summary = "; ".join(f"{name}: {str(e)[:160]}" for name, e in errors)
    if any(is_rate_limit_error(e) for _, e in errors):
        return LLMRateLimited(summary)
    if errors and all(is_config_error(e) for _, e in errors):
        return LLMNotConfigured(f"No configured model is usable — {summary}")
    return RuntimeError(f"All models failed — {summary}")


# ---------------------------------------------------------------------------
# Calling with failover
# ---------------------------------------------------------------------------

def invoke_with_tools(tools: list, messages: list) -> Any:
    """Call the first working model with the tools bound; returns its AIMessage."""
    errors: list[tuple[str, Exception]] = []
    for cand in candidates():
        try:
            return cand.make().bind_tools(tools).invoke(messages)
        except Exception as exc:
            log.warning("%s failed (%s); trying the next model.", cand.name, str(exc)[:200])
            errors.append((cand.name, exc))
    raise _final_error(errors)


def invoke_structured_budgeted(schema, build: Callable[[int], tuple[str, T]]) -> tuple[Any, T]:
    """Structured-output call with failover.  `build(budget)` returns the prompt
    for that model's context budget plus a payload (e.g. the exact messages it
    numbered), which is returned with the result so citations line up."""
    errors: list[tuple[str, Exception]] = []
    for cand in candidates():
        budget = settings.max_context_chars or cand.budget
        prompt, payload = build(budget)
        try:
            out = cand.make().with_structured_output(schema).invoke(prompt)
            if out is None:
                raise ValueError("model returned no structured output")
            return out, payload
        except Exception as exc:
            log.warning("%s failed (%s); trying the next model.", cand.name, str(exc)[:200])
            errors.append((cand.name, exc))
    raise _final_error(errors)


def invoke_structured(schema, prompt: str) -> Any:
    out, _ = invoke_structured_budgeted(schema, lambda _budget: (prompt, None))
    return out
