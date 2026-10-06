"""
config.py — single source of truth for all environment variables.
Every env var used by the backend is declared here via pydantic-settings.
"""
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    # "auto": try Gemini models first (if GEMINI_API_KEY is set), then Groq.
    # "gemini" / "groq": use only that provider.
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")

    # Google Gemini (primary) — free key at aistudio.google.com/app/apikey
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    # Tried in order when the main model is overloaded, retired or out of quota.
    gemini_fallback_models: str = Field(default="gemini-flash-lite-latest", alias="GEMINI_FALLBACK_MODELS")

    # Groq (fallback) — free key at console.groq.com
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")

    # ── Storage ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="app/data/chroma", alias="CHROMA_PERSIST_DIR")
    sqlite_path: str = Field(default="app/data/dedup.db", alias="SQLITE_PATH")

    # ── Ingestion ────────────────────────────────────────────────────────────
    # Hours of silence between messages before a new session chunk begins.
    session_gap_hours: float = Field(default=2.0, alias="SESSION_GAP_HOURS")
    # Session chunks longer than this are split (MiniLM reads ~256 word-pieces).
    max_chunk_chars: int = Field(default=1000, alias="MAX_CHUNK_CHARS")
    # "auto" detects day-first vs month-first from the export; or force DMY / MDY.
    date_order: str = Field(default="auto", alias="DATE_ORDER")
    max_upload_mb: int = Field(default=50, alias="MAX_UPLOAD_MB")

    # ── Agent ────────────────────────────────────────────────────────────────
    max_tool_rounds: int = Field(default=3, alias="MAX_TOOL_ROUNDS")
    # Seconds before an LLM call is abandoned (and the next model is tried).
    llm_timeout_s: float = Field(default=45, alias="LLM_TIMEOUT_S")
    # Character budget for retrieved text sent to the LLM in one call.
    # 0 = auto per model: 30000 for Gemini, 12000 for Groq (its free tier
    # caps tokens per minute much lower).
    max_context_chars: int = Field(default=0, alias="MAX_CONTEXT_CHARS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
