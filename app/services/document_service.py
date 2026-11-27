"""Document lifecycle: upload -> validate -> extract -> chunk -> embed -> store; list; delete.

This service orchestrates the other services and both repositories.
Route handlers call it and only translate results into HTTP responses.

Indexing (extract -> chunk -> embed -> store) can be slow for big scanned files, so by
default it runs on a background worker thread. `upload()` only validates and saves the
file, then returns a "processing" record at once; the browser polls until it is ready.
One worker means documents are indexed one at a time, which keeps CPU and memory use
predictable (OCR and the embedding model are both heavy).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.core.exceptions import AppError, DocumentBusyError, DocumentNotFoundError, ExtractionError
from app.domain import DocumentRecord, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_spaces import EmbeddingSpaces
from app.services.extraction_service import ExtractionService
from app.utils.file_validation import validate_upload

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        upload_dir: Path,
        max_upload_bytes: int,
        documents: DocumentRepository,
        spaces: EmbeddingSpaces,
        extractor: ExtractionService,
        chunker: ChunkingService,
        background: bool = True,
    ) -> None:
        self.upload_dir = upload_dir.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.documents = documents
        self.spaces = spaces
        self.extractor = extractor
        self.chunker = chunker
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer") if background else None
        self._lock = threading.Lock()  # makes "is it already uploaded?" + "save" one step

    # ------------------------------------------------------------------ upload

    def upload(
        self,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        ocr_language: str | None = None,
        embedding_model: str | None = None,
    ) -> tuple[DocumentRecord, bool]:
        """Validate and store a file, then index it (in the background by default).

        Returns (record, created). `created` is False when an identical file is
        already indexed with the same embedding model, or is being indexed, so no
        work is repeated. Uploading an indexed file with a *different* embedding
        model re-indexes it with the new model.
        Validation problems raise; extraction problems are recorded as a FAILED
        document so the user can see what went wrong.
        """
        validated = validate_upload(filename, content_type, content, self.max_upload_bytes)
        language = self.extractor.normalize_language(ocr_language)
        model = self.spaces.resolve(embedding_model)

        # Stable ID from the file's bytes: same file -> same ID -> no duplicate work.
        document_id = hashlib.sha256(validated.content).hexdigest()[:32]

        with self._lock:
            existing = self._with_model(self.documents.get(document_id))
            if existing and (
                existing.status is DocumentStatus.PROCESSING
                or (existing.status is DocumentStatus.INDEXED and existing.embedding_model == model)
            ):
                logger.info("Document %s already %s; nothing to redo", document_id, existing.status.value)
                return existing, False
            if existing and existing.embedding_model != model:
                # Re-indexing with another model: remove the vectors made by the old one.
                self.spaces.get(existing.embedding_model).store.delete_document(document_id)

            # Server-generated name: never derived from user input, so no path traversal.
            stored_filename = f"{document_id}{validated.extension}"
            path = self._safe_path(stored_filename)
            path.write_bytes(validated.content)

            record = DocumentRecord(
                document_id=document_id,
                filename=validated.original_filename,
                stored_filename=stored_filename,
                file_kind=validated.kind,
                size_bytes=len(validated.content),
                status=DocumentStatus.PROCESSING,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ocr_language=language,
                embedding_model=model,
            )
            self.documents.save(record)

        self._schedule(record, validated.content)
        return record, True

    def resume_unfinished(self) -> int:
        """Restart indexing for documents left "processing" when the server stopped."""
        unfinished = [self._with_model(r) for r in self.documents.list_by_status(DocumentStatus.PROCESSING)]
        for record in unfinished:
            path = self._safe_path(record.stored_filename)
            if path.exists():
                logger.info("Resuming indexing of %s (%s)", record.document_id, record.filename)
                self._schedule(record, path.read_bytes())
            else:
                self._mark_failed(record, "The uploaded file is missing. Please upload it again.")
        return len(unfinished)

    def shutdown(self) -> None:
        """Stop the worker. Unfinished documents stay "processing" and resume on next start."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _schedule(self, record: DocumentRecord, content: bytes) -> None:
        if self._executor is None:
            self._index_safely(record, content)
        else:
            # The worker gets its own copy, so the record we return to the caller is
            # never changed half-way by another thread.
            self._executor.submit(self._index_safely, dataclasses.replace(record), content)

    def _index_safely(self, record: DocumentRecord, content: bytes) -> None:
        """Index and turn every error into a FAILED record (nothing must escape a worker)."""
        try:
            self._index(record, content)
        except ExtractionError as exc:
            self._mark_failed(record, exc.message)
        except AppError as exc:
            self._mark_failed(record, exc.message)
        except Exception:
            logger.exception("Unexpected error while indexing %s", record.document_id)
            self._mark_failed(record, "An unexpected error occurred while indexing this document.")

    def _index(self, record: DocumentRecord, content: bytes) -> None:
        extraction = self.extractor.extract(content, record.file_kind, record.ocr_language)
        record.page_count = len(extraction.pages)
        record.ocr_used = extraction.ocr_used
        record.warnings = list(extraction.warnings)

        if extraction.total_chars == 0:
            hint = (
                " The image or scan may be too blurry, or Tesseract OCR may be unavailable."
                if record.file_kind.value != "docx"
                else ""
            )
            raise ExtractionError("No readable text was found in this document." + hint)

        chunks = self.chunker.chunk_pages(record.document_id, record.filename, extraction.pages)
        space = self.spaces.get(record.embedding_model)
        embeddings = space.embedder.embed_documents([c.text for c in chunks])

        # Remove stale vectors from any previous failed attempt, then write fresh ones.
        space.store.delete_document(record.document_id)
        space.store.upsert_chunks(chunks, embeddings)

        record.chunk_count = len(chunks)
        record.status = DocumentStatus.INDEXED
        record.error_message = None
        self.documents.save(record)
        logger.info(
            "Indexed %s (%s): %d pages, %d chunks, ocr=%s, embedding=%s",
            record.document_id, record.filename, record.page_count, record.chunk_count, record.ocr_used,
            record.embedding_model,
        )

    def _mark_failed(self, record: DocumentRecord, message: str) -> None:
        logger.warning("Indexing failed for %s: %s", record.document_id, message)
        record.status = DocumentStatus.FAILED
        record.error_message = message
        record.chunk_count = 0
        self.spaces.get(record.embedding_model).store.delete_document(record.document_id)
        self.documents.save(record)

    # ------------------------------------------------------------------ read / delete

    def list_documents(self) -> list[DocumentRecord]:
        return [self._with_model(r) for r in self.documents.list_all()]

    def get_document(self, document_id: str) -> DocumentRecord:
        record = self._with_model(self.documents.get(document_id))
        if record is None:
            raise DocumentNotFoundError("Document not found.")
        return record

    def indexed_document_ids(self) -> set[str]:
        return {d.document_id for d in self.documents.list_all() if d.status is DocumentStatus.INDEXED}

    def indexed_by_model(self) -> dict[str, list[str]]:
        """Indexed document IDs grouped by the embedding model that indexed them."""
        groups: dict[str, list[str]] = {}
        for d in self.list_documents():
            if d.status is DocumentStatus.INDEXED:
                groups.setdefault(d.embedding_model, []).append(d.document_id)
        return groups

    def delete(self, document_id: str) -> None:
        """Remove vectors, the stored file, and the record."""
        record = self.get_document(document_id)
        if record.status is DocumentStatus.PROCESSING:
            # Deleting now would race with the worker, which would write the record back.
            raise DocumentBusyError("This document is still being indexed. Wait until it finishes, then delete it.")
        self.spaces.get(record.embedding_model).store.delete_document(document_id)
        path = self._safe_path(record.stored_filename)
        path.unlink(missing_ok=True)
        self.documents.delete(document_id)
        logger.info("Deleted document %s", document_id)

    # ------------------------------------------------------------------ helpers

    def _with_model(self, record: DocumentRecord | None) -> DocumentRecord | None:
        """Documents indexed before models could be chosen used the default model."""
        if record is not None and record.embedding_model is None:
            record.embedding_model = self.spaces.default_model
        return record

    def _safe_path(self, stored_filename: str) -> Path:
        """Resolve a path and refuse anything that escapes the upload directory."""
        path = (self.upload_dir / stored_filename).resolve()
        if not path.is_relative_to(self.upload_dir):
            raise AppError("Invalid storage path.")
        return path
