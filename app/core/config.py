"""Centralised, typed application settings.

All configuration is read from environment variables (or a local `.env` file).
Nothing secret is hardcoded here — only safe defaults.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"  # OpenAI or any OpenAI-compatible server (e.g. Ollama)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "RAG Document Chatbot"
    log_level: str = "INFO"

    # --- Storage ---
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chroma_db_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    metadata_db_path: Path = PROJECT_ROOT / "data" / "documents.sqlite3"
    chroma_collection: str = "document_chunks"

    # --- Upload limits ---
    max_upload_size_mb: float = Field(default=20, gt=0, le=200)
    # Index uploads in a background thread so the browser gets an answer straight away.
    # Tests turn this off so indexing finishes before the upload request returns.
    background_indexing: bool = True

    # --- Chunking ---
    chunk_size: int = Field(default=1000, ge=100, le=8000, description="Characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters shared between neighbouring chunks")

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1)

    # --- Retrieval ---
    top_k: int = Field(default=5, ge=1, le=50)
    # Cosine similarity in [−1, 1]. If even the BEST match scores below this, the question is
    # treated as unrelated and the LLM is not called. Otherwise all top-K chunks go to the LLM.
    min_relevance_score: float = Field(default=0.15, ge=-1, le=1)

    # --- OCR ---
    tesseract_cmd: str | None = None  # e.g. /opt/homebrew/bin/tesseract; None = use PATH
    # Default OCR language(s); the user can pick another per upload. Combine with "+", e.g. "hin+eng".
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    # A PDF page with fewer extracted characters than this is treated as scanned and OCR'd.
    ocr_min_chars_per_page: int = Field(default=50, ge=0)

    # --- Conversation memory ---
    # How many earlier question/answer pairs are sent with a follow-up question.
    # More turns = better memory, but a longer (slower) prompt.
    chat_history_turns: int = Field(default=6, ge=0, le=20)

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-opus-5-5"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None  # only for OpenAI-compatible servers
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_max_tokens: int = Field(default=4096, ge=256)
    # Anthropic only: reasoning effort (low | medium | high | xhigh | max).
    llm_effort: str = "low"
    # Anthropic only: let the API re-run a refused request on a fallback model.
    anthropic_enable_fallbacks: bool = True

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.llm_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("LLM_EFFORT must be one of: low, medium, high, xhigh, max")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def llm_configured(self) -> bool:
        """An OpenAI-compatible local server (base_url set) may not need a key."""
        has_key = self.llm_api_key is not None and self.llm_api_key.get_secret_value().strip() != ""
        if self.llm_provider is LLMProvider.OPENAI and self.llm_base_url:
            return True
        return has_key

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
