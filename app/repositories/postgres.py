"""Postgres storage (used when DATABASE_URL is set, e.g. on Vercel with Neon).

Why: on Vercel the app can run as several copies at once, each with its own temporary
disk. Files saved by one copy are invisible to the others ("no document uploaded",
"not found"). A shared database fixes that, and its data survives restarts.

The four classes here do exactly what the local ones do, with the same methods:
- PgDocumentRepository   <-> document_repository.DocumentRepository (SQLite)
- PgVectorStore          <-> vector_store.VectorStore (ChromaDB), using pgvector
- PgAnswerCacheRepository <-> answer_cache_repository.AnswerCacheRepository (SQLite)
- PgFileStore            <-> file_store.LocalFileStore (the data/uploads folder)
so the services above don't know or care which one they use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.domain import DocumentRecord, DocumentStatus, FileKind, RetrievedChunk, TextChunk
from app.repositories.answer_cache_repository import CachedAnswer

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    file_kind       TEXT NOT NULL,
    size_bytes      BIGINT NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    page_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ocr_used        BOOLEAN NOT NULL DEFAULT FALSE,
    error_message   TEXT,
    warnings        TEXT NOT NULL DEFAULT '[]',
    ocr_language    TEXT,
    embedding_model TEXT
);

-- One row per chunk. Vectors from different models have different sizes, so the column
-- has no fixed size and every search is limited to one embedding_model.
CREATE TABLE IF NOT EXISTS chunks (
    embedding_model TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    document_id     TEXT NOT NULL,
    filename        TEXT NOT NULL,
    page_number     INTEGER,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    embedding       vector NOT NULL,
    PRIMARY KEY (embedding_model, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (embedding_model, document_id);

CREATE TABLE IF NOT EXISTS answer_cache (
    id            BIGSERIAL PRIMARY KEY,
    lookup_key    TEXT NOT NULL,
    question      TEXT NOT NULL,
    question_norm TEXT NOT NULL,
    embedding     TEXT NOT NULL,
    answer        TEXT NOT NULL,
    cited_ids     TEXT NOT NULL,
    llm_model     TEXT,
    hits          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS answer_cache_lookup ON answer_cache (lookup_key);

CREATE TABLE IF NOT EXISTS stored_files (
    name    TEXT PRIMARY KEY,
    content BYTEA NOT NULL
);
"""

_DOC_COLUMNS = (
    "document_id, filename, stored_filename, file_kind, size_bytes, status, created_at, "
    "page_count, chunk_count, ocr_used, error_message, warnings, ocr_language, embedding_model"
)


class PostgresDatabase:
    """A small pool of connections shared by the four classes below."""

    def __init__(self, url: str, schema: str | None = None) -> None:
        from psycopg_pool import ConnectionPool

        # Tests use their own schema; "public" stays on the path so the `vector` type is found.
        options = f"-c search_path={schema},public" if schema else ""
        self.pool = ConnectionPool(
            url,
            min_size=0,  # serverless: don't hold connections open while idle
            max_size=4,
            kwargs={"options": options, "autocommit": False} if options else {"autocommit": False},
            check=ConnectionPool.check_connection,  # Neon closes idle connections; reconnect quietly
            open=True,
        )
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
            if schema:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            conn.execute(_SCHEMA)

    def close(self) -> None:
        self.pool.close()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


# ---------------------------------------------------------------------- documents


class PgDocumentRepository:
    def __init__(self, db: PostgresDatabase) -> None:
        self._pool = db.pool

    def save(self, record: DocumentRecord) -> None:
        values = (
            record.document_id, record.filename, record.stored_filename, record.file_kind.value,
            record.size_bytes, record.status.value, record.created_at, record.page_count,
            record.chunk_count, bool(record.ocr_used), record.error_message,
            json.dumps(record.warnings), record.ocr_language, record.embedding_model,
        )
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _DOC_COLUMNS.split(", ")[1:])
        with self._pool.connection() as conn:
            conn.execute(
                f"INSERT INTO documents ({_DOC_COLUMNS}) VALUES ({', '.join(['%s'] * len(values))}) "
                f"ON CONFLICT (document_id) DO UPDATE SET {updates}",
                values,
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM documents WHERE document_id = %s", (document_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        with self._pool.connection() as conn:
            rows = conn.execute(f"SELECT {_DOC_COLUMNS} FROM documents ORDER BY created_at DESC").fetchall()
        return [self._to_record(r) for r in rows]

    def list_by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM documents WHERE status = %s ORDER BY created_at", (status.value,)
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def delete(self, document_id: str) -> bool:
        with self._pool.connection() as conn:
            return conn.execute("DELETE FROM documents WHERE document_id = %s", (document_id,)).rowcount > 0

    @staticmethod
    def _to_record(row: tuple) -> DocumentRecord:
        (document_id, filename, stored_filename, file_kind, size_bytes, status, created_at,
         page_count, chunk_count, ocr_used, error_message, warnings, ocr_language, embedding_model) = row
        return DocumentRecord(
            document_id=document_id,
            filename=filename,
            stored_filename=stored_filename,
            file_kind=FileKind(file_kind),
            size_bytes=size_bytes,
            status=DocumentStatus(status),
            created_at=created_at,
            page_count=page_count,
            chunk_count=chunk_count,
            ocr_used=bool(ocr_used),
            error_message=error_message,
            warnings=json.loads(warnings),
            ocr_language=ocr_language,
            embedding_model=embedding_model,
        )


# ---------------------------------------------------------------------- vectors


class PgVectorStore:
    """Chunks + vectors of ONE embedding model (like one Chroma collection)."""

    def __init__(self, db: PostgresDatabase, embedding_model: str) -> None:
        self._pool = db.pool
        self._model = embedding_model

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = [
            (self._model, c.chunk_id, c.document_id, c.filename, c.page_number, c.chunk_index, c.text,
             _vector_literal(e))
            for c, e in zip(chunks, embeddings)
        ]
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (embedding_model, chunk_id, document_id, filename, page_number, "
                "chunk_index, text, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) "
                "ON CONFLICT (embedding_model, chunk_id) DO UPDATE SET document_id = EXCLUDED.document_id, "
                "filename = EXCLUDED.filename, page_number = EXCLUDED.page_number, "
                "chunk_index = EXCLUDED.chunk_index, text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                rows,
            )

    def delete_document(self, document_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE embedding_model = %s AND document_id = %s", (self._model, document_id)
            )

    def search(
        self, query_embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        sql = (
            "SELECT chunk_id, document_id, filename, text, page_number, "
            "1 - (embedding <=> %s::vector) AS score FROM chunks WHERE embedding_model = %s"
        )
        params: list = [_vector_literal(query_embedding), self._model]
        if document_ids:
            sql += " AND document_id = ANY(%s)"
            params.append(list(document_ids))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [_vector_literal(query_embedding), top_k]
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RetrievedChunk(chunk_id=r[0], document_id=r[1], filename=r[2], text=r[3], page_number=r[4], score=float(r[5]))
            for r in rows
        ]

    def document_chunks(self, document_id: str, limit: int) -> list[RetrievedChunk]:
        """The first `limit` chunks of one document, in reading order (used for summaries)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, document_id, filename, text, page_number FROM chunks "
                "WHERE embedding_model = %s AND document_id = %s "
                "ORDER BY page_number NULLS FIRST, chunk_index LIMIT %s",
                (self._model, document_id, limit),
            ).fetchall()
        return [
            RetrievedChunk(chunk_id=r[0], document_id=r[1], filename=r[2], text=r[3], page_number=r[4], score=1.0)
            for r in rows
        ]

    def count_for_document(self, document_id: str) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_model = %s AND document_id = %s",
                (self._model, document_id),
            ).fetchone()[0]

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding_model = %s", (self._model,)).fetchone()[0]


# ---------------------------------------------------------------------- answer cache


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PgAnswerCacheRepository:
    def __init__(self, db: PostgresDatabase) -> None:
        self._pool = db.pool

    def find(self, lookup_key: str) -> list[CachedAnswer]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, question, question_norm, embedding, answer, cited_ids, llm_model "
                "FROM answer_cache WHERE lookup_key = %s",
                (lookup_key,),
            ).fetchall()
        return [
            CachedAnswer(id=r[0], question=r[1], question_norm=r[2], embedding=json.loads(r[3]), answer=r[4],
                         cited_ids=json.loads(r[5]), llm_model=r[6])
            for r in rows
        ]

    def add(self, lookup_key, question, question_norm, embedding, answer, cited_ids, llm_model, max_entries) -> None:
        now = _now()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO answer_cache (lookup_key, question, question_norm, embedding, answer, cited_ids, "
                "llm_model, created_at, last_used_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (lookup_key, question, question_norm, json.dumps(embedding), answer, json.dumps(cited_ids),
                 llm_model, now, now),
            )
            conn.execute(
                "DELETE FROM answer_cache WHERE id NOT IN "
                "(SELECT id FROM answer_cache ORDER BY last_used_at DESC, id DESC LIMIT %s)",
                (max_entries,),
            )

    def mark_used(self, entry_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE answer_cache SET hits = hits + 1, last_used_at = %s WHERE id = %s", (_now(), entry_id)
            )

    def clear(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("DELETE FROM answer_cache").rowcount

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]


# ---------------------------------------------------------------------- uploaded files


class PgFileStore:
    """Keeps the original uploaded files (needed to resume or redo indexing)."""

    def __init__(self, db: PostgresDatabase) -> None:
        self._pool = db.pool

    def save(self, name: str, content: bytes) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO stored_files (name, content) VALUES (%s, %s) "
                "ON CONFLICT (name) DO UPDATE SET content = EXCLUDED.content",
                (name, content),
            )

    def read(self, name: str) -> bytes | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT content FROM stored_files WHERE name = %s", (name,)).fetchone()
        return bytes(row[0]) if row else None

    def delete(self, name: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM stored_files WHERE name = %s", (name,))
