"""SQLite storage for cached answers (the only place that talks to the answer_cache table).

Lives in the same SQLite file as the document records. Each row is one answer the LLM
wrote, plus what is needed to decide later whether a new question can reuse it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lookup_key    TEXT NOT NULL,     -- search scope + LLM model + the exact passages found
    question      TEXT NOT NULL,
    question_norm TEXT NOT NULL,     -- lower-case, no punctuation: for exact matches
    embedding     TEXT NOT NULL,     -- question vector (JSON list): for "same meaning" matches
    answer        TEXT NOT NULL,
    cited_ids     TEXT NOT NULL,     -- chunk IDs the answer cites (JSON list)
    llm_model     TEXT,
    hits          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT NOT NULL
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS answer_cache_lookup ON answer_cache (lookup_key)"


@dataclass(frozen=True)
class CachedAnswer:
    id: int
    question: str
    question_norm: str
    embedding: list[float]
    answer: str
    cited_ids: list[str]
    llm_model: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnswerCacheRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()  # serialise writes from FastAPI's thread pool
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.execute(_INDEX)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def find(self, lookup_key: str) -> list[CachedAnswer]:
        """All cached answers made from exactly this scope, model and set of passages."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question, question_norm, embedding, answer, cited_ids, llm_model "
                "FROM answer_cache WHERE lookup_key = ?",
                (lookup_key,),
            ).fetchall()
        return [
            CachedAnswer(
                id=r["id"],
                question=r["question"],
                question_norm=r["question_norm"],
                embedding=json.loads(r["embedding"]),
                answer=r["answer"],
                cited_ids=json.loads(r["cited_ids"]),
                llm_model=r["llm_model"],
            )
            for r in rows
        ]

    def add(
        self,
        lookup_key: str,
        question: str,
        question_norm: str,
        embedding: list[float],
        answer: str,
        cited_ids: list[str],
        llm_model: str | None,
        max_entries: int,
    ) -> None:
        """Save an answer, then drop the least recently used rows above `max_entries`."""
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO answer_cache (lookup_key, question, question_norm, embedding, answer, "
                "cited_ids, llm_model, created_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (lookup_key, question, question_norm, json.dumps(embedding), answer,
                 json.dumps(cited_ids), llm_model, now, now),
            )
            conn.execute(
                "DELETE FROM answer_cache WHERE id NOT IN "
                "(SELECT id FROM answer_cache ORDER BY last_used_at DESC, id DESC LIMIT ?)",
                (max_entries,),
            )

    def mark_used(self, entry_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE answer_cache SET hits = hits + 1, last_used_at = ? WHERE id = ?", (_now(), entry_id)
            )

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM answer_cache").rowcount

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
