"""
config.py — single source of truth for all environment variables.
Every env var used by the app is declared here via pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Google Gemini API key (kept for reference, currently not used as active LLM).
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Gemini model name (not used when Groq is the active provider).
    gemini_model: str = Field(default="gemini-2.0-flash-lite", alias="GEMINI_MODEL")

    # Groq API key — get a free key at console.groq.com
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # Groq model — llama-3.3-70b-versatile supports structured output reliably.
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Path where Chroma will persist its on-disk vector store.
    chroma_persist_dir: str = Field(
        default="app/data/chroma", alias="CHROMA_PERSIST_DIR"
    )

    # Path to the SQLite deduplication database.
    sqlite_path: str = Field(
        default="app/data/dedup.db", alias="SQLITE_PATH"
    )

    # Gap in hours between messages that triggers a new session chunk.
    session_gap_hours: float = Field(default=2.0, alias="SESSION_GAP_HOURS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
