"""Find the chunks most relevant to a question."""

from __future__ import annotations

import logging

from app.domain import RetrievedChunk
from app.repositories.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        default_top_k: int = 5,
        min_relevance_score: float = 0.25,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.min_relevance_score = min_relevance_score

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the question (same model as indexing) and return the top-K chunks.

        The relevance check is applied to the *best* match only. Similarity scores
        from small embedding models are noisy: the passage that actually answers a
        question can score lower than a loosely related one. So if the question
        relates to the documents at all, we pass all top-K chunks to the LLM, which
        is far better at judging relevance and is told to say when evidence is
        missing. Only clearly unrelated questions (best score below the floor)
        return nothing, which skips the LLM call entirely.
        """
        query_vector = self.embedder.embed_query(question)
        hits = self.vector_store.search(
            query_vector, top_k=top_k or self.default_top_k, document_ids=document_ids
        )
        hits.sort(key=lambda h: h.score, reverse=True)
        best = hits[0].score if hits else float("nan")
        related = bool(hits) and best >= self.min_relevance_score
        logger.info(
            "Retrieved %d chunk(s); best=%.3f, floor=%.2f -> %s",
            len(hits), best, self.min_relevance_score, "sending to LLM" if related else "unrelated",
        )
        return hits if related else []

    def retrieve_any(
        self,
        queries: list[str],
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search with several phrasings of one question and keep the best top-K overall.

        Used for follow-up questions: "What is his email?" alone matches nothing, but
        "Who is the manager? What is his email?" finds the right part of the document.
        A chunk found by more than one query keeps its highest score.
        """
        best: dict[str, RetrievedChunk] = {}
        for query in queries:
            for hit in self.retrieve(query, top_k=top_k, document_ids=document_ids):
                if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                    best[hit.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ranked[: top_k or self.default_top_k]
