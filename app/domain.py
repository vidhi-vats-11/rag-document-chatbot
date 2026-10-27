"""Plain data objects passed between the service and repository layers.

These are internal types (dataclasses). The HTTP-facing shapes live in
`app/schemas/` as Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FileKind(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True)
class ExtractedPage:
    """Text from one page (PDF), one image, or one whole DOCX (page_number=None)."""

    text: str
    page_number: int | None
    ocr_used: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    pages: list[ExtractedPage]
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_used(self) -> bool:
        return any(p.ocr_used for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text.strip()) for p in self.pages)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    page_number: int | None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int | None
    score: float  # cosine similarity, higher = more relevant


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    stored_filename: str
    file_kind: FileKind
    size_bytes: int
    status: DocumentStatus
    created_at: str
    page_count: int = 0
    chunk_count: int = 0
    ocr_used: bool = False
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    ocr_language: str | None = None  # Tesseract code(s) used for OCR, e.g. "hin+eng"


@dataclass(frozen=True)
class ChatTurn:
    """One earlier question and the answer it got (conversation memory)."""

    question: str
    answer: str
