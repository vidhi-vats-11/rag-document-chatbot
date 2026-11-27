"""Which models the user may choose from (the web page fills its dropdowns from this)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.core.container import Container

router = APIRouter(prefix="/models", tags=["models"])


class ModelsResponse(BaseModel):
    embedding_models: list[str] = Field(description="Allowed embedding models for indexing, default first.")
    default_embedding_model: str
    llm_models: list[str] = Field(description="Allowed LLM models for answering, default first.")
    default_llm_model: str
    rerank_by_default: bool
    rerank_model: str


@router.get("", response_model=ModelsResponse, summary="Models available for indexing, answering and re-ranking")
def list_models(container: Annotated[Container, Depends(get_container)]) -> ModelsResponse:
    s = container.settings
    return ModelsResponse(
        embedding_models=container.spaces.allowed_models,
        default_embedding_model=container.spaces.default_model,
        llm_models=container.chat_service.llm_models,
        default_llm_model=container.chat_service.llm_models[0] if container.chat_service.llm_models else s.llm_model,
        rerank_by_default=s.rerank_enabled,
        rerank_model=container.reranker.model_name,
    )
