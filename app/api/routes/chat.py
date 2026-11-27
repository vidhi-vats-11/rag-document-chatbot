"""Chat endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.domain import ChatTurn
from app.schemas.chat import AskRequest, AskResponse, CacheClearedResponse, SourceOut
from app.schemas.common import ErrorResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

_SNIPPET_CHARS = 300


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "LLM model not allowed"},
        404: {"model": ErrorResponse, "description": "Selected document not found"},
        429: {"model": ErrorResponse, "description": "LLM rate limit"},
        502: {"model": ErrorResponse, "description": "LLM provider error"},
        503: {"model": ErrorResponse, "description": "LLM not configured"},
        504: {"model": ErrorResponse, "description": "LLM timeout"},
    },
    summary="Ask a question about the uploaded documents",
)
def ask(body: AskRequest, service: Annotated[ChatService, Depends(get_chat_service)]) -> AskResponse:
    history = [ChatTurn(t.question, t.answer) for t in body.history]
    result = service.ask(
        body.question,
        document_id=body.document_id,
        top_k=body.top_k,
        history=history,
        llm_model=body.llm_model,
        rerank=body.rerank,
        use_cache=body.use_cache,
    )
    return AskResponse(
        answer=result.answer,
        grounded=result.grounded,
        llm_model=result.llm_model,
        reranked=result.reranked,
        cached=result.cached,
        sources=[
            SourceOut(
                document_id=s.document_id,
                filename=s.filename,
                page_number=s.page_number,
                chunk_id=s.chunk_id,
                score=round(s.score, 4),
                rerank_score=round(s.rerank_score, 4) if s.rerank_score is not None else None,
                snippet=s.text[:_SNIPPET_CHARS] + ("…" if len(s.text) > _SNIPPET_CHARS else ""),
            )
            for s in result.sources
        ],
    )


@router.delete("/cache", response_model=CacheClearedResponse, summary="Delete all cached answers")
def clear_cache(service: Annotated[ChatService, Depends(get_chat_service)]) -> CacheClearedResponse:
    return CacheClearedResponse(removed=service.clear_cache())
