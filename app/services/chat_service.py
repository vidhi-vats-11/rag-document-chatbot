"""Answer a question: retrieve -> (if relevant) generate -> attach sources."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import DocumentNotFoundError, LLMNotConfiguredError
from app.domain import ChatTurn, RetrievedChunk
from app.services.document_service import DocumentService
from app.services.generation_service import GenerationService, extract_citations
from app.services.retrieval_service import RetrievalService

NO_DOCUMENTS_ANSWER = "No documents have been indexed yet. Upload a document first, then ask your question."
NO_RELEVANT_ANSWER = (
    "I couldn't find information relevant to your question in the uploaded documents, "
    "so I can't answer it from them. Try rephrasing, or check that the right document is selected."
)


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[RetrievedChunk]
    grounded: bool  # True when the answer was generated from retrieved context


class ChatService:
    def __init__(
        self,
        documents: DocumentService,
        retrieval: RetrievalService,
        generation: GenerationService,
        history_turns: int = 6,
    ) -> None:
        self.documents = documents
        self.retrieval = retrieval
        self.generation = generation
        self.history_turns = history_turns

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
        history: list[ChatTurn] | None = None,
    ) -> ChatAnswer:
        """Answer `question`. `history` holds earlier turns, oldest first (conversation memory)."""
        # Fail fast with a clear message before doing any work.
        if not self.generation.configured:
            raise LLMNotConfiguredError(
                "No LLM is configured. Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env and restart."
            )

        indexed = self.documents.indexed_document_ids()
        if document_id is not None and document_id not in indexed:
            raise DocumentNotFoundError("The selected document does not exist or is not indexed.")
        if not indexed:
            return ChatAnswer(NO_DOCUMENTS_ANSWER, [], grounded=False)

        # Keep only the most recent turns: enough to follow the conversation, short prompt.
        history = (history or [])[-self.history_turns :] if self.history_turns else []

        # Search with the question alone AND joined to the previous question, so a
        # follow-up like "What is his email?" still finds the right part of the document.
        queries = [question]
        if history:
            queries.append(f"{history[-1].question}\n{question}")
        chunks = self.retrieval.retrieve_any(
            queries, top_k=top_k, document_ids=[document_id] if document_id else None
        )
        if not chunks:
            # Don't ask the LLM with no evidence — that invites a made-up answer.
            return ChatAnswer(NO_RELEVANT_ANSWER, [], grounded=False)

        answer = self.generation.generate(question, chunks, history)

        # Show only the excerpts the model actually cited. If it cited nothing (e.g. it said
        # the documents don't cover the question), listing pages would be misleading.
        cited = extract_citations(answer, len(chunks))
        sources = [chunks[i - 1] for i in cited]
        return ChatAnswer(answer, sources, grounded=True)
