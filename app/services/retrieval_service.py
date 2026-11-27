"""Find the chunks most relevant to a question.

Two steps:
1. Vector search: embed the question with each embedding model that indexed the
   documents, and search that model's collection. Fast, but rough.
2. Re-ranking (optional): a cross-encoder re-scores the candidates and keeps the best.
"""

from __future__ import annotations

import logging

from app.domain import RetrievedChunk
from app.services.embedding_spaces import EmbeddingSpaces
from app.services.rerank_service import RerankService

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        spaces: EmbeddingSpaces,
        reranker: RerankService | None = None,
        default_top_k: int = 5,
        min_relevance_score: float = 0.25,
        rerank_by_default: bool = True,
        rerank_candidates: int = 20,
    ) -> None:
        self.spaces = spaces
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.min_relevance_score = min_relevance_score
        self.rerank_by_default = rerank_by_default
        self.rerank_candidates = rerank_candidates

    def retrieve(
        self,
        question: str,
        targets: dict[str, list[str] | None],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Search every embedding model in `targets` and return the top-K chunks overall.

        `targets` maps an embedding model to the document IDs to search with it
        (None = every document in that model's collection).

        The relevance check is applied to the *best* match only. Similarity scores
        from small embedding models are noisy: the passage that actually answers a
        question can score lower than a loosely related one. So if the question
        relates to the documents at all, we pass all top-K chunks on. Only clearly
        unrelated questions (best score below the floor) return nothing, which skips
        the LLM call entirely.

        Note: when documents use different embedding models, their cosine scores are
        not truly comparable. Re-ranking fixes this, because one cross-encoder
        scores every candidate on the same scale.
        """
        hits: list[RetrievedChunk] = []
        for model_name, document_ids in targets.items():
            space = self.spaces.get(model_name)
            query_vector = space.embedder.embed_query(question)
            hits.extend(space.store.search(query_vector, top_k=top_k, document_ids=document_ids))
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]
        best = hits[0].score if hits else float("nan")
        related = bool(hits) and best >= self.min_relevance_score
        logger.info(
            "Retrieved %d chunk(s) from %d model(s); best=%.3f, floor=%.2f -> %s",
            len(hits), len(targets), best, self.min_relevance_score, "relevant" if related else "unrelated",
        )
        return hits if related else []

    def retrieve_any(
        self,
        queries: list[str],
        targets: dict[str, list[str] | None],
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Search with several phrasings of one question, then optionally re-rank.

        Several phrasings help follow-up questions: "What is his email?" alone matches
        nothing, but "Who is the manager? What is his email?" finds the right part of
        the document. A chunk found by more than one query keeps its highest score.

        With re-ranking, vector search fetches more candidates than needed and the
        cross-encoder picks the final top-K. It scores against the LAST query, which
        is the one with the most context (previous question + new question).
        """
        top_k = top_k or self.default_top_k
        use_rerank = (self.rerank_by_default if rerank is None else rerank) and self.reranker is not None
        fetch = max(top_k, self.rerank_candidates) if use_rerank else top_k

        best: dict[str, RetrievedChunk] = {}
        for query in queries:
            for hit in self.retrieve(query, targets, top_k=fetch):
                if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                    best[hit.chunk_id] = hit
        candidates = sorted(best.values(), key=lambda h: h.score, reverse=True)[:fetch]

        if use_rerank and candidates:
            return self.reranker.rerank(queries[-1], candidates, top_k)
        return candidates[:top_k]
