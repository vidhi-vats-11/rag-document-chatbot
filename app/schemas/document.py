"""HTTP request/response shapes for document endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain import DocumentRecord, DocumentStatus, FileKind


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    file_kind: FileKind
    size_bytes: int
    status: DocumentStatus
    created_at: str
    page_count: int
    chunk_count: int
    ocr_used: bool
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    ocr_language: str | None = None
    embedding_model: str | None = None

    @classmethod
    def from_record(cls, r: DocumentRecord) -> "DocumentOut":
        # Note: stored_filename (server path) is deliberately NOT exposed.
        return cls(
            document_id=r.document_id,
            filename=r.filename,
            file_kind=r.file_kind,
            size_bytes=r.size_bytes,
            status=r.status,
            created_at=r.created_at,
            page_count=r.page_count,
            chunk_count=r.chunk_count,
            ocr_used=r.ocr_used,
            error_message=r.error_message,
            warnings=r.warnings,
            ocr_language=r.ocr_language,
            embedding_model=r.embedding_model,
        )


class UploadResponse(BaseModel):
    document: DocumentOut
    already_indexed: bool = Field(
        description="True if this exact file was already indexed (or is being indexed); no work was repeated."
    )


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool = True


class OcrLanguagesResponse(BaseModel):
    languages: list[str] = Field(description='Installed Tesseract codes, e.g. ["eng", "fra", "hin"].')
    default: str
