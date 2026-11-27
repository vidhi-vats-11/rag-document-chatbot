"""HTTP request/response shapes for the chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatTurnIn(BaseModel):
    """One earlier question/answer pair, sent back by the browser for follow-up questions."""

    question: str = Field(max_length=2000)
    answer: str = Field(max_length=10000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What is the refund policy?"])
    document_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{32}$",
        description="Restrict the search to one document. Omit to search all indexed documents.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve (default from settings).")
    llm_model: str | None = Field(
        default=None,
        max_length=100,
        description="LLM model for this answer; must be one listed by GET /api/v1/models. Omit for the default.",
    )
    rerank: bool | None = Field(
        default=None, description="Re-rank the retrieved chunks with a cross-encoder. Omit for the server default."
    )
    history: list[ChatTurnIn] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier turns of this conversation, oldest first. Lets follow-ups like 'what about him?' work.",
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceOut(BaseModel):
    document_id: str
    filename: str
    page_number: int | None = Field(description="1-based page; null for formats without pages (DOCX).")
    chunk_id: str
    score: float = Field(description="Cosine similarity between question and chunk (higher = more relevant).")
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder score in [0, 1] (higher = more relevant); null if not re-ranked."
    )
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    grounded: bool = Field(description="False when no relevant context was found and the LLM was not called.")
    llm_model: str | None = Field(default=None, description="Model that wrote the answer; null if the LLM was not called.")
    reranked: bool = Field(default=False, description="True when the chunks were re-ranked before answering.")
