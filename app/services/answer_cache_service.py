"""Semantic answer cache: reuse an earlier answer when the same question is asked again.

Why: each LLM call costs quota (Gemini free tier ≈ 20 a day) and takes seconds. A cached
answer costs nothing and comes back at once.

A new question reuses a cached answer only when ALL of these match:
1. The search scope (one document or all) and the LLM model.
2. The exact set of document passages the search found. If documents were added, deleted
   or re-indexed with different content, or the question points somewhere else, the set
   changes and the cache is skipped. So a cached answer is never stale.
3. The question itself: either the same words (ignoring case, spaces and punctuation), or
   the same meaning, i.e. cosine similarity of the two question vectors >= `min_similarity`.

Follow-up questions: the search already uses the previous question, so the passages in
rule 2 reflect the conversation. But a question like "What is his email?" means something
different after "Who is the CEO?" than after "Who is the CTO?", even if the same passages are
found. So when a question looks like it depends on the chat (it uses words like "he", "it",
"that", or is very short), the previous question must match too.

The similarity bar is deliberately high (0.95). Measured with all-MiniLM-L6-v2:
"When does the store open?" vs "...close?" scores 0.86, higher than some true rewordings
("What is the refund period?" vs "How long do I have to get a refund?" = 0.79). A lower bar
would give wrong answers, so the cache only catches near-identical wording.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from app.domain import ChatTurn, RetrievedChunk
from app.repositories.answer_cache_repository import AnswerCacheRepository
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheHit:
    answer: str
    sources: list[RetrievedChunk]  # the cited passages, taken from the current search results
    llm_model: str | None
    matched_question: str
    similarity: float


def normalise(question: str) -> str:
    """'  How long do I have, to get a REFUND?? ' -> 'how long do i have to get a refund'"""
    return " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())


# Words that point back to something said earlier in the chat.
_CONTEXT_WORDS = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "there", "same", "above", "previous", "earlier", "else",
    "also", "another", "former", "latter",
}
_SHORT_QUESTION_WORDS = 3  # "And shipping?", "Why?" only make sense with the chat before them


def depends_on_chat(question: str) -> bool:
    words = normalise(question).split()
    return len(words) <= _SHORT_QUESTION_WORDS or any(w in _CONTEXT_WORDS for w in words)


def chat_context(question: str, history: list[ChatTurn] | None) -> str:
    """'' for a self-contained question; otherwise the previous question it builds on."""
    if not history or not depends_on_chat(question):
        return ""
    return normalise(history[-1].question)


def lookup_key(scope: str | None, llm_model: str | None, chunks: list[RetrievedChunk], context: str = "") -> str:
    passages = "|".join(sorted(c.chunk_id for c in chunks))
    raw = f"{scope or '*'}\n{llm_model or ''}\n{passages}\n{context}"
    return hashlib.sha256(raw.encode()).hexdigest()


class AnswerCacheService:
    def __init__(
        self,
        repository: AnswerCacheRepository,
        embedder: EmbeddingService,
        enabled: bool = True,
        min_similarity: float = 0.95,
        max_entries: int = 1000,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.enabled = enabled
        self.min_similarity = min_similarity
        self.max_entries = max_entries

    def lookup(
        self,
        question: str,
        scope: str | None,
        llm_model: str | None,
        chunks: list[RetrievedChunk],
        history: list[ChatTurn] | None = None,
    ) -> CacheHit | None:
        if not self.enabled or not chunks:
            return None
        key = lookup_key(scope, llm_model, chunks, chat_context(question, history))
        candidates = self.repository.find(key)
        if not candidates:
            return None

        norm = normalise(question)
        best, best_score = None, -1.0
        exact = next((c for c in candidates if c.question_norm == norm), None)
        if exact:
            best, best_score = exact, 1.0
        else:
            vector = self.embedder.embed_query(question)
            for c in candidates:
                score = sum(a * b for a, b in zip(vector, c.embedding))  # vectors are normalised
                if score > best_score:
                    best, best_score = c, score
        if best is None or best_score < self.min_similarity:
            return None

        by_id = {c.chunk_id: c for c in chunks}
        sources = [by_id[i] for i in best.cited_ids if i in by_id]
        self.repository.mark_used(best.id)
        logger.info("Answer cache hit (similarity %.3f): %r ~ %r", best_score, question, best.question)
        return CacheHit(best.answer, sources, best.llm_model, best.question, best_score)

    def store(
        self,
        question: str,
        scope: str | None,
        llm_model: str | None,
        chunks: list[RetrievedChunk],
        answer: str,
        cited: list[RetrievedChunk],
        history: list[ChatTurn] | None = None,
    ) -> None:
        if not self.enabled or not chunks:
            return
        self.repository.add(
            lookup_key=lookup_key(scope, llm_model, chunks, chat_context(question, history)),
            question=question,
            question_norm=normalise(question),
            embedding=self.embedder.embed_query(question),
            answer=answer,
            cited_ids=[c.chunk_id for c in cited],
            llm_model=llm_model,
            max_entries=self.max_entries,
        )

    def clear(self) -> int:
        removed = self.repository.clear()
        logger.info("Answer cache cleared (%d entries)", removed)
        return removed

    def size(self) -> int:
        return self.repository.count()
