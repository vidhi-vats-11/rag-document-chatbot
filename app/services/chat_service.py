"""Answer a question: retrieve -> (if relevant) generate -> attach sources."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.exceptions import DocumentNotFoundError, InvalidModelError, LLMNotConfiguredError
from app.domain import ChatTurn, RetrievedChunk
from app.services.answer_cache_service import AnswerCacheService
from app.services.document_service import DocumentService
from app.services.generation_service import GenerationService, extract_citations
from app.services.retrieval_service import RetrievalService

NO_DOCUMENTS_ANSWER = "No documents have been indexed yet. Upload a document first, then ask your question."
NO_RELEVANT_ANSWER = (
    "I couldn't find information relevant to your question in the uploaded documents, "
    "so I can't answer it from them. Try rephrasing, or check that the right document is selected."
)

# "Summarize this", "give me an overview", "what is this document about?", "key points"...
_SUMMARY_REQUEST = re.compile(
    r"\b(summar(y|ies|i[sz]e|i[sz]ing)|overview|tl;?dr|gist|key (points|takeaways)|main (points|ideas)"
    r"|what('s| is) (this|the) (document|doc|file|pdf|text) about)\b",
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    return bool(_SUMMARY_REQUEST.search(question))


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[RetrievedChunk]
    grounded: bool  # True when the answer was generated from retrieved context
    llm_model: str | None = None  # model that wrote the answer; None when the LLM was not called
    reranked: bool = False  # True when the sources were re-ranked by the cross-encoder
    cached: bool = False  # True when the answer was reused from the answer cache (no LLM call)


class ChatService:
    def __init__(
        self,
        documents: DocumentService,
        retrieval: RetrievalService,
        generation: GenerationService,
        history_turns: int = 6,
        llm_models: list[str] | None = None,
        cache: AnswerCacheService | None = None,
        summary_max_chunks: int = 40,
    ) -> None:
        self.documents = documents
        self.retrieval = retrieval
        self.generation = generation
        self.history_turns = history_turns
        # Allowed models, default first. Empty = no choice offered (use the configured model).
        self.llm_models = llm_models or []
        self.cache = cache
        self.summary_max_chunks = summary_max_chunks

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
        history: list[ChatTurn] | None = None,
        llm_model: str | None = None,
        rerank: bool | None = None,
        use_cache: bool = True,
    ) -> ChatAnswer:
        """Answer `question`. `history` holds earlier turns, oldest first (conversation memory).

        `llm_model` picks one of the allowed LLM models for this answer (default: the first).
        `rerank` switches re-ranking on/off for this question (default: from settings).
        `use_cache` = False always asks the LLM (e.g. to get a fresh answer).
        """
        # Fail fast with a clear message before doing any work.
        if not self.generation.configured:
            raise LLMNotConfiguredError(
                "No LLM is configured. Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env and restart."
            )
        model = self._resolve_llm_model(llm_model)

        by_model = self.documents.indexed_by_model()
        if document_id is not None:
            # Search only the one model that indexed the chosen document.
            targets = {m: [document_id] for m, ids in by_model.items() if document_id in ids}
            if not targets:
                raise DocumentNotFoundError("The selected document does not exist or is not indexed.")
        elif sum(len(ids) for ids in by_model.values()) == 1:
            targets = dict(by_model)  # only one document exists: "all documents" means that one
        else:
            targets = {m: None for m in by_model}
        if not targets:
            return ChatAnswer(NO_DOCUMENTS_ANSWER, [], grounded=False)

        # Keep only the most recent turns: enough to follow the conversation, short prompt.
        history = (history or [])[-self.history_turns :] if self.history_turns else []

        # Search with the question alone AND joined to the previous question, so a
        # follow-up like "What is his email?" still finds the right part of the document.
        queries = [question]
        if history:
            queries.append(f"{history[-1].question}\n{question}")

        # "Summarize this document" matches no particular passage, so searching for it finds
        # little. Instead, give the LLM the whole document in reading order (up to a limit).
        single_doc = self._single_document(targets)
        if single_doc and is_summary_request(question):
            chunks = self.retrieval.document_chunks(*single_doc, limit=self.summary_max_chunks)
        else:
            chunks = self.retrieval.retrieve_any(queries, targets, top_k=top_k, rerank=rerank)
        if not chunks:
            # Don't ask the LLM with no evidence — that invites a made-up answer.
            return ChatAnswer(NO_RELEVANT_ANSWER, [], grounded=False)

        reranked = any(c.rerank_score is not None for c in chunks)

        # Reuse an earlier answer if the same question found the same passages
        # (answer_cache_service.py explains how follow-up questions are handled).
        cacheable = self.cache is not None and use_cache
        if cacheable:
            hit = self.cache.lookup(question, document_id, model, chunks, history)
            if hit:
                return ChatAnswer(
                    hit.answer, hit.sources, grounded=True, llm_model=hit.llm_model, reranked=reranked, cached=True
                )

        answer = self.generation.generate(question, chunks, history, model=model)

        # Show only the excerpts the model actually cited. If it cited nothing (e.g. it said
        # the documents don't cover the question), listing pages would be misleading.
        cited = extract_citations(answer, len(chunks))
        sources = [chunks[i - 1] for i in cited]
        if cacheable:
            self.cache.store(question, document_id, model, chunks, answer, sources, history)
        return ChatAnswer(answer, sources, grounded=True, llm_model=model, reranked=reranked)

    @staticmethod
    def _single_document(targets: dict[str, list[str] | None]) -> tuple[str, str] | None:
        """(embedding model, document id) when exactly one document is being searched."""
        if len(targets) != 1:
            return None
        model, ids = next(iter(targets.items()))
        return (model, ids[0]) if ids and len(ids) == 1 else None

    def _resolve_llm_model(self, requested: str | None) -> str | None:
        """Only models from the allowed list may be used (each one can cost quota or money)."""
        requested = (requested or "").strip()
        if not requested:
            return self.llm_models[0] if self.llm_models else None
        if requested not in self.llm_models:
            raise InvalidModelError(
                f"LLM model '{requested}' is not allowed. Choose one of: {', '.join(self.llm_models)}."
            )
        return requested

    def clear_cache(self) -> int:
        """Delete every cached answer; returns how many were removed."""
        return self.cache.clear() if self.cache else 0
