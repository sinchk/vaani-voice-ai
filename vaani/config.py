"""Central configuration.

All settings come from environment variables (optionally loaded from a ``.env``
file) and every filesystem path is anchored to the project root, so the app
behaves identically regardless of the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- LLM / embeddings -------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("VAANI_CHAT_MODEL", "gemini-3.8-flash"))
    # Tried when the primary model stays overloaded after retries. Empty string disables it.
    fallback_model: str = field(default_factory=lambda: os.getenv("VAANI_FALLBACK_MODEL", "gemini-3.1-flash-lite"))
    embedding_model: str = field(default_factory=lambda: os.getenv("VAANI_EMBEDDING_MODEL", "gemini-embedding-001"))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("VAANI_LLM_MAX_RETRIES", "3")))
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("VAANI_MAX_TOOL_ROUNDS", "4")))

    # --- RAG --------------------------------------------------------------
    knowledge_base_dir: Path = field(
        default_factory=lambda: _path("VAANI_KB_DIR", PROJECT_ROOT / "data" / "knowledge_base")
    )
    chroma_dir: Path = field(default_factory=lambda: _path("PERSIST_DIRECTORY", PROJECT_ROOT / "chroma_db"))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("VAANI_RAG_TOP_K", "3")))
    # Cosine similarity (0-1) a chunk must reach to be used as context. Gemini embeddings are
    # compressed (unrelated text still scores ~0.55), so the floor sits well above that.
    rag_min_score: float = field(default_factory=lambda: float(os.getenv("VAANI_RAG_MIN_SCORE", "0.62")))
    # Also drop chunks scoring more than this below the best match, to cut near-miss noise.
    rag_margin: float = field(default_factory=lambda: float(os.getenv("VAANI_RAG_MARGIN", "0.05")))

    # --- Business data ----------------------------------------------------
    seed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    state_dir: Path = field(default_factory=lambda: _path("VAANI_STATE_DIR", PROJECT_ROOT / ".state"))

    # --- Speech -----------------------------------------------------------
    whisper_model: str = field(default_factory=lambda: os.getenv("VAANI_WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("VAANI_WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("VAANI_WHISPER_COMPUTE", "int8"))
    audio_dir: Path = field(default_factory=lambda: _path("VAANI_AUDIO_DIR", PROJECT_ROOT / ".state" / "audio"))
    audio_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("VAANI_AUDIO_TTL", "3600")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("VAANI_MAX_UPLOAD_MB", "10")) * 1024 * 1024)

    # --- Misc -------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("VAANI_MAX_HISTORY", "20")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
