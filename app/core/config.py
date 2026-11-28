"""Centralised, typed application settings.

All configuration is read from environment variables (or a local `.env` file).
Nothing secret is hardcoded here — only safe defaults.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"  # OpenAI or any OpenAI-compatible server (e.g. Ollama)


class EmbeddingProvider(str, Enum):
    LOCAL = "local"  # models run on this computer (ONNX Runtime): free, private, ~90 MB per model
    API = "api"  # an OpenAI-compatible embedding API (e.g. Gemini); for hosts with no room for models


class OCRProvider(str, Enum):
    TESSERACT = "tesseract"  # local Tesseract program (free, private)
    LLM = "llm"  # the LLM reads the image (Gemini can see images); 1 LLM request per page


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
    # Postgres (with pgvector) instead of SQLite + ChromaDB + the uploads folder. Needed on hosts
    # without a shared disk, e.g. Vercel (+ Neon, which sets DATABASE_URL). Empty = local files.
    database_url: SecretStr | None = None

    # --- Upload limits ---
    max_upload_size_mb: float = Field(default=20, gt=0, le=200)
    # Index uploads in a background thread so the browser gets an answer straight away.
    # Tests turn this off so indexing finishes before the upload request returns.
    background_indexing: bool = True

    # --- Chunking ---
    chunk_size: int = Field(default=1000, ge=100, le=8000, description="Characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters shared between neighbouring chunks")

    # --- Embeddings ---
    # "api" uses LLM_BASE_URL + LLM_API_KEY (e.g. EMBEDDING_MODEL=gemini-embedding-001) and turns
    # re-ranking off, because no local models are loaded at all. Used for Vercel.
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1)
    # Models the user may pick per upload (comma-separated). EMBEDDING_MODEL is always allowed.
    # Each model gets its own Chroma collection, because vectors from different models
    # have different sizes and meanings and must never be mixed.
    embedding_model_choices: str = (
        "sentence-transformers/all-MiniLM-L6-v2,"
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1,"
        "sentence-transformers/all-mpnet-base-v2,"
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Retrieval ---
    top_k: int = Field(default=5, ge=1, le=50)
    # Cosine similarity in [−1, 1]. If even the BEST match scores below this, the question is
    # treated as unrelated and the LLM is not called. Otherwise all top-K chunks go to the LLM.
    min_relevance_score: float = Field(default=0.15, ge=-1, le=1)

    # --- Re-ranking ---
    # Vector search is fast but rough. Re-ranking fetches more candidates (RERANK_CANDIDATES),
    # lets a cross-encoder read each (question, chunk) pair, and keeps the best TOP_K.
    rerank_enabled: bool = True  # default for each question; the user can switch it per question
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    # --- OCR ---
    ocr_provider: OCRProvider = OCRProvider.TESSERACT
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
    # "Summarize this document": send the document's first N chunks in reading order (not just
    # the TOP_K best matches). 40 chunks x 1000 characters ~ 40k characters.
    summary_max_chunks: int = Field(default=40, ge=1, le=200)

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gemini-3.5-flash-lite"
    # Other models of the same provider the user may pick per question (comma-separated).
    # LLM_MODEL is always allowed and is the default.
    llm_model_choices: str = ""
    llm_api_key: SecretStr | None = None
    # OpenAI-compatible servers only. Default: Google Gemini (free tier). Ollama: http://localhost:11434/v1
    llm_base_url: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_max_tokens: int = Field(default=4096, ge=256)
    # Anthropic only: reasoning effort (low | medium | high | xhigh | max).
    llm_effort: str = "low"
    # Anthropic only: let the API re-run a refused request on a fallback model.
    anthropic_enable_fallbacks: bool = True

    # --- Answer cache: reuse the answer when the same question is asked again (saves LLM quota)
    answer_cache_enabled: bool = True
    # How alike two questions must be (cosine, 0..1) to count as "the same". Keep it high:
    # "store open?" vs "store close?" already scores 0.86 with all-MiniLM-L6-v2.
    answer_cache_min_similarity: float = Field(default=0.95, ge=0.5, le=1)
    answer_cache_max_entries: int = Field(default=1000, ge=1)

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
    def embedding_models(self) -> list[str]:
        """Allowed embedding models, default first."""
        return _choices(self.embedding_model, self.embedding_model_choices)

    @property
    def llm_models(self) -> list[str]:
        """Allowed LLM models, default first."""
        return _choices(self.llm_model, self.llm_model_choices)

    @property
    def llm_configured(self) -> bool:
        """A key is needed, except for a server on this computer (e.g. Ollama), which accepts any."""
        has_key = self.llm_api_key is not None and self.llm_api_key.get_secret_value().strip() != ""
        if self.llm_provider is LLMProvider.OPENAI and self.llm_base_url:
            host = (urlparse(self.llm_base_url).hostname or "").lower()
            if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
                return True
        return has_key

    @property
    def uses_database(self) -> bool:
        return self.database_url is not None and self.database_url.get_secret_value().strip() != ""

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)


def _choices(default: str, extra: str) -> list[str]:
    """Split a comma-separated list, put the default first and drop duplicates."""
    names = [default] + [n.strip() for n in extra.split(",")]
    return list(dict.fromkeys(n for n in names if n))


@lru_cache
def get_settings() -> Settings:
    return Settings()
