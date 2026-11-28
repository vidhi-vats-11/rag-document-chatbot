"""ChromaDB access. This is the ONLY module that talks to the vector database.

Services pass in already-computed embeddings; this layer just stores and
searches them. Keeping Chroma behind this class means it could be replaced
(pgvector, Qdrant, ...) without touching the services.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import EmbeddingModelMismatchError
from app.domain import RetrievedChunk, TextChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 500


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_model: str) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # we always supply our own embeddings
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedding_model},
        )
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedding_model:
            # Vectors from different models live in different "spaces";
            # mixing them silently would make search results meaningless.
            raise EmbeddingModelMismatchError(
                f"Collection '{collection_name}' was built with embedding model "
                f"'{stored_model}', but EMBEDDING_MODEL is '{embedding_model}'. "
                "Use the original model, choose a new CHROMA_COLLECTION, or re-index."
            )

    # ------------------------------------------------------------------ writes

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[i : i + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[i : i + _UPSERT_BATCH],
                documents=[c.text for c in batch],
                metadatas=[self._metadata_for(c) for c in batch],
            )

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    # ------------------------------------------------------------------ reads

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []

        where: dict[str, Any] | None = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(meta["document_id"]),
                    filename=str(meta.get("filename", "unknown")),
                    text=text or "",
                    page_number=int(meta["page_number"]) if "page_number" in meta else None,
                    score=1.0 - float(distance),  # cosine distance -> cosine similarity
                )
            )
        return hits

    def document_chunks(self, document_id: str, limit: int) -> list[RetrievedChunk]:
        """The first `limit` chunks of one document, in reading order (used for summaries)."""
        result = self._collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        rows = sorted(
            zip(result["ids"], result["documents"], result["metadatas"]),
            key=lambda r: (int(r[2].get("page_number", 0)), int(r[2].get("chunk_index", 0))),
        )[:limit]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=str(meta["document_id"]),
                filename=str(meta.get("filename", "unknown")),
                text=text or "",
                page_number=int(meta["page_number"]) if "page_number" in meta else None,
                score=1.0,
            )
            for chunk_id, text, meta in rows
        ]

    def count_for_document(self, document_id: str) -> int:
        return len(self._collection.get(where={"document_id": document_id}, include=[])["ids"])

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _metadata_for(chunk: TextChunk) -> dict[str, str | int]:
        meta: dict[str, str | int] = {
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
        }
        # Chroma metadata cannot hold None, so we omit page_number when unknown (DOCX).
        if chunk.page_number is not None:
            meta["page_number"] = chunk.page_number
        return meta
