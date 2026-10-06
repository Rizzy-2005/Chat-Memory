"""
errors.py — maps LLM / agent failures to clean HTTP errors (never a stack trace).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from fastapi import HTTPException

from app.agent.llm import LLMNotConfigured, LLMRateLimited, is_rate_limit_error

log = logging.getLogger(__name__)

RATE_LIMIT_DETAIL = "The free AI quota is used up for now. Try again in a minute."


@contextmanager
def llm_errors(action: str):
    """Translate exceptions raised inside the block into HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMRateLimited as exc:
        raise HTTPException(status_code=429, detail=RATE_LIMIT_DETAIL) from exc
    except Exception as exc:
        if is_rate_limit_error(exc):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_DETAIL) from exc
        log.exception("%s failed", action)
        raise HTTPException(status_code=500, detail=f"{action} failed: {type(exc).__name__}: {exc}"[:500]) from exc
