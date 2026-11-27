"""Shared test fixtures.

Tests never call a real LLM and, except for tests marked `slow`, never load the
real embedding model. Instead they use:
- FakeEncoder: a deterministic bag-of-words "embedding" (texts sharing words are similar)
- FakeLLM: records prompts and returns a canned answer (or raises a chosen error)
- FakeReranker: scores a (question, chunk) pair by how many question words the chunk contains
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.core.config import Settings
from app.core.container import Container, build_container
from app.main import create_app
from app.services.embedding_service import EmbeddingService
from app.services.rerank_service import RerankService

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
requires_tesseract = pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract OCR not installed")

_DIM = 256
_STOPWORDS = {"the", "a", "an", "is", "of", "and", "to", "in", "what", "for", "on", "are", "does", "do", "how"}


class FakeEncoder:
    """Hashes each word into one of 256 buckets; returns L2-normalised vectors."""

    def encode(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word in _STOPWORDS:
                    continue
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm == 0:
                out[row, 0] = 1.0  # avoid zero vectors (undefined cosine)
            else:
                out[row] /= norm
        return out


class FakeLLM:
    def __init__(self, answer: str = "The refund window is 30 days [1].", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.models: list[str | None] = []  # model asked for in each call

    def complete(self, system_prompt: str, user_message: str, model: str | None = None) -> str:
        self.calls.append((system_prompt, user_message))
        self.models.append(model)
        if self.error:
            raise self.error
        return self.answer


class FakeReranker:
    """Share of the query's (non-stopword) words that appear in the chunk, in [0, 1]."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        pairs = list(inputs)
        self.calls.append(pairs)
        scores = []
        for query, text in pairs:
            q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS}
            t = set(re.findall(r"[a-z0-9]+", text.lower()))
            scores.append(len(q & t) / len(q) if q else 0.0)
        return np.array(scores, dtype=np.float32)


def fake_embedder_factory(name: str) -> EmbeddingService:
    return EmbeddingService(name, model=FakeEncoder())


# --------------------------------------------------------------------- settings / app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        upload_dir=tmp_path / "uploads",
        chroma_db_dir=tmp_path / "chroma",
        metadata_db_path=tmp_path / "documents.sqlite3",
        chroma_collection="test_chunks",
        embedding_model="fake-encoder",
        embedding_model_choices="fake-encoder-two",
        llm_model="fake-llm",
        llm_model_choices="fake-llm-two",
        rerank_enabled=False,  # tests that want it pass rerank=True per question
        chunk_size=300,
        chunk_overlap=50,
        max_upload_size_mb=2,
        llm_api_key=None,
        min_relevance_score=0.2,
        background_indexing=False,  # index before upload returns; one test turns it on
    )


@pytest.fixture
def embedder() -> EmbeddingService:
    return EmbeddingService("fake-encoder", model=FakeEncoder())


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


def make_container(
    settings: Settings, embedder: EmbeddingService, llm: FakeLLM | None, reranker: FakeReranker | None = None
) -> Container:
    return build_container(
        settings,
        embedder=embedder,
        llm=llm,
        use_configured_llm=False,
        embedder_factory=fake_embedder_factory,
        reranker=RerankService("fake-reranker", model=reranker or FakeReranker()),
    )


@pytest.fixture
def container(settings, embedder, fake_llm, fake_reranker) -> Container:
    return make_container(settings, embedder, fake_llm, fake_reranker)


@pytest.fixture
def client(container) -> TestClient:
    with TestClient(create_app(container=container)) as c:
        yield c


@pytest.fixture
def client_without_llm(settings, embedder) -> TestClient:
    with TestClient(create_app(container=make_container(settings, embedder, None))) as c:
        yield c


# --------------------------------------------------------------------- sample files

REFUND_TEXT = (
    "Refund Policy. Customers may request a refund within 30 days of purchase. "
    "Refunds are issued to the original payment method within 5 business days."
)
SHIPPING_TEXT = (
    "Shipping Information. Orders ship from our Berlin warehouse. "
    "Standard delivery takes 3 to 7 working days across Europe."
)


def make_text_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def render_text_image(text: str, fmt: str = "PNG") -> bytes:
    font = ImageFont.load_default(size=36)
    lines = _wrap(text, 40)
    img = Image.new("RGB", (1600, 80 + 55 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 55 * i), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_scanned_pdf(text: str) -> bytes:
    """A PDF whose only page is an image of text — no embedded text layer."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=render_text_image(text))
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def upload(client: TestClient, filename: str, content: bytes, content_type: str):
    return client.post("/api/v1/documents/upload", files={"file": (filename, content, content_type)})


PDF = "application/pdf"
PNG = "image/png"
JPEG = "image/jpeg"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
