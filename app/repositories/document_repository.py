"""SQLite storage for document records (filename, status, counts, errors).

ChromaDB stores *chunks*; this table stores one row per *document* so we can
list uploads, show indexing status/errors, and know which file to delete.
SQLite is part of Python's standard library, so no extra service is needed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.domain import DocumentRecord, DocumentStatus, FileKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    file_kind       TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    page_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ocr_used        INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    warnings        TEXT NOT NULL DEFAULT '[]',
    ocr_language    TEXT,
    embedding_model TEXT
)
"""

# Columns added after the first release. Older databases get them via ALTER TABLE.
_ADDED_COLUMNS = {"ocr_language": "TEXT", "embedding_model": "TEXT"}

_COLUMNS = (
    "document_id, filename, stored_filename, file_kind, size_bytes, status, created_at, "
    "page_count, chunk_count, ocr_used, error_message, warnings, ocr_language, embedding_model"
)


class DocumentRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()  # serialise writes from FastAPI's thread pool
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            for name, sql_type in _ADDED_COLUMNS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, record: DocumentRecord) -> None:
        """Insert or replace the full record."""
        values = (
            record.document_id,
            record.filename,
            record.stored_filename,
            record.file_kind.value,
            record.size_bytes,
            record.status.value,
            record.created_at,
            record.page_count,
            record.chunk_count,
            int(record.ocr_used),
            record.error_message,
            json.dumps(record.warnings),
            record.ocr_language,
            record.embedding_model,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO documents ({_COLUMNS}) VALUES ({', '.join('?' * len(values))})",
                values,
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def list_by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE status = ? ORDER BY created_at", (status.value,)
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def delete(self, document_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            stored_filename=row["stored_filename"],
            file_kind=FileKind(row["file_kind"]),
            size_bytes=row["size_bytes"],
            status=DocumentStatus(row["status"]),
            created_at=row["created_at"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            ocr_used=bool(row["ocr_used"]),
            error_message=row["error_message"],
            warnings=json.loads(row["warnings"]),
            ocr_language=row["ocr_language"],
            embedding_model=row["embedding_model"],
        )
