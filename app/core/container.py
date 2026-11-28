"""Builds and wires every service once, at startup ("composition root").

Routes get services from here through FastAPI dependencies, and tests can
build a container with fake parts (e.g. a mocked LLM, a tiny embedding model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.core.config import EmbeddingProvider, Settings
from app.repositories.answer_cache_repository import AnswerCacheRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_store import VectorStore
from app.services.answer_cache_service import AnswerCacheService
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.embedding_spaces import EmbeddingSpaces
from app.services.extraction_service import ExtractionService
from app.services.generation_service import GenerationService
from app.services.llm_providers import LLMClient, build_llm_client
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService


@dataclass
class Container:
    settings: Settings
    extractor: ExtractionService
    embedder: EmbeddingService  # default embedding model
    vector_store: VectorStore  # collection of the default embedding model
    spaces: EmbeddingSpaces  # every allowed embedding model and its collection
    reranker: RerankService
    document_service: DocumentService
    chat_service: ChatService


def build_container(
    settings: Settings,
    embedder: EmbeddingService | None = None,
    llm: LLMClient | None = None,
    use_configured_llm: bool = True,
    embedder_factory: Callable[[str], EmbeddingService] | None = None,
    reranker: RerankService | None = None,
) -> Container:
    settings.ensure_directories()

    extractor = ExtractionService(
        ocr_language=settings.ocr_language,
        ocr_dpi=settings.ocr_dpi,
        ocr_min_chars_per_page=settings.ocr_min_chars_per_page,
        tesseract_cmd=settings.tesseract_cmd,
    )
    api_mode = settings.embedding_provider is EmbeddingProvider.API
    loader = None
    if api_mode:
        from app.services.api_embeddings import ApiSentenceEncoder

        loader = lambda name: ApiSentenceEncoder(name, settings)  # noqa: E731
    embedder = embedder or EmbeddingService(settings.embedding_model, settings.embedding_batch_size, loader=loader)
    vector_store = VectorStore(settings.chroma_db_dir, settings.chroma_collection, embedder.model_name)
    spaces = EmbeddingSpaces(
        default_embedder=embedder,
        default_store=vector_store,
        persist_dir=settings.chroma_db_dir,
        base_collection=settings.chroma_collection,
        allowed_models=settings.embedding_models,
        embedder_factory=embedder_factory
        or (lambda name: EmbeddingService(name, settings.embedding_batch_size, loader=loader)),
    )
    reranker = reranker or RerankService(settings.rerank_model)
    documents = DocumentRepository(settings.metadata_db_path)

    document_service = DocumentService(
        upload_dir=settings.upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
        documents=documents,
        spaces=spaces,
        extractor=extractor,
        chunker=ChunkingService(settings.chunk_size, settings.chunk_overlap),
        background=settings.background_indexing,
    )

    if llm is None and use_configured_llm:
        llm = build_llm_client(settings)

    chat_service = ChatService(
        documents=document_service,
        retrieval=RetrievalService(
            spaces,
            None if api_mode else reranker,  # API mode loads no local models, so no re-ranker
            default_top_k=settings.top_k,
            min_relevance_score=settings.min_relevance_score,
            rerank_by_default=settings.rerank_enabled,
            rerank_candidates=settings.rerank_candidates,
        ),
        generation=GenerationService(llm),
        history_turns=settings.chat_history_turns,
        llm_models=settings.llm_models,
        cache=AnswerCacheService(
            AnswerCacheRepository(settings.metadata_db_path),
            embedder,
            enabled=settings.answer_cache_enabled,
            min_similarity=settings.answer_cache_min_similarity,
            max_entries=settings.answer_cache_max_entries,
        ),
    )
    return Container(
        settings, extractor, embedder, vector_store, spaces, reranker, document_service, chat_service
    )
