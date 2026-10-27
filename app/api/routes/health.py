from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import Container

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm_provider: str
    ocr_available: bool
    embedding_model: str
    indexed_chunks: int


@router.get("/health", response_model=HealthResponse, summary="Service health and configuration status")
def health(container: Annotated[Container, Depends(get_container)]) -> HealthResponse:
    s = container.settings
    return HealthResponse(
        status="ok",
        llm_configured=container.chat_service.generation.configured,
        llm_provider=s.llm_provider.value,
        ocr_available=container.extractor.ocr_available(),
        embedding_model=container.embedder.model_name,
        indexed_chunks=container.vector_store.count(),
    )
