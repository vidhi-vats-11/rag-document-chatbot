"""Builds and wires every service once, at startup ("composition root").

Routes get services from here through FastAPI dependencies, and tests can
build a container with fake parts (e.g. a mocked LLM, a tiny embedding model).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_store import VectorStore
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import ExtractionService
from app.services.generation_service import GenerationService
from app.services.llm_providers import LLMClient, build_llm_client
from app.services.retrieval_service import RetrievalService


@dataclass
class Container:
    settings: Settings
    extractor: ExtractionService
    embedder: EmbeddingService
    vector_store: VectorStore
    document_service: DocumentService
    chat_service: ChatService


def build_container(
    settings: Settings,
    embedder: EmbeddingService | None = None,
    llm: LLMClient | None = None,
    use_configured_llm: bool = True,
) -> Container:
    settings.ensure_directories()

    extractor = ExtractionService(
        ocr_language=settings.ocr_language,
        ocr_dpi=settings.ocr_dpi,
        ocr_min_chars_per_page=settings.ocr_min_chars_per_page,
        tesseract_cmd=settings.tesseract_cmd,
    )
    embedder = embedder or EmbeddingService(settings.embedding_model, settings.embedding_batch_size)
    vector_store = VectorStore(settings.chroma_db_dir, settings.chroma_collection, embedder.model_name)
    documents = DocumentRepository(settings.metadata_db_path)

    document_service = DocumentService(
        upload_dir=settings.upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
        documents=documents,
        vector_store=vector_store,
        extractor=extractor,
        chunker=ChunkingService(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        background=settings.background_indexing,
    )

    if llm is None and use_configured_llm:
        llm = build_llm_client(settings)

    chat_service = ChatService(
        documents=document_service,
        retrieval=RetrievalService(vector_store, embedder, settings.top_k, settings.min_relevance_score),
        generation=GenerationService(llm),
        history_turns=settings.chat_history_turns,
    )
    return Container(settings, extractor, embedder, vector_store, document_service, chat_service)
