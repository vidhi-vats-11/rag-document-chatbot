"""Chat endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.domain import ChatTurn
from app.schemas.chat import AskRequest, AskResponse, SourceOut
from app.schemas.common import ErrorResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

_SNIPPET_CHARS = 300


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
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
    result = service.ask(body.question, document_id=body.document_id, top_k=body.top_k, history=history)
    return AskResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[
            SourceOut(
                document_id=s.document_id,
                filename=s.filename,
                page_number=s.page_number,
                chunk_id=s.chunk_id,
                score=round(s.score, 4),
                snippet=s.text[:_SNIPPET_CHARS] + ("…" if len(s.text) > _SNIPPET_CHARS else ""),
            )
            for s in result.sources
        ],
    )
