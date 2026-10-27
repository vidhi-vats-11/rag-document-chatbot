import math

import pytest

from app.core.exceptions import EmbeddingModelMismatchError
from app.domain import TextChunk
from app.repositories.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from tests.conftest import FakeEncoder


def _chunk(doc: str, idx: int, text: str, page: int | None = 1) -> TextChunk:
    return TextChunk(f"{doc}:p{page or 0}:c{idx}", doc, f"{doc}.pdf", text, idx, page)


# ------------------------------------------------------------------ embeddings


def test_embeddings_are_normalised_and_batched():
    service = EmbeddingService("fake", model=FakeEncoder())
    vectors = service.embed_documents(["refund policy", "shipping times", "refund window"])
    assert len(vectors) == 3
    assert all(len(v) == 256 for v in vectors)
    assert all(math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-5) for v in vectors)


def test_embed_empty_list():
    assert EmbeddingService("fake", model=FakeEncoder()).embed_documents([]) == []


def test_query_and_document_embeddings_match():
    service = EmbeddingService("fake", model=FakeEncoder())
    assert service.embed_query("refund policy") == service.embed_documents(["refund policy"])[0]


@pytest.mark.slow
def test_real_model_semantic_similarity():
    """Loads the real Sentence Transformers model (downloaded on first run)."""
    try:
        service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
        q, related, unrelated = service.embed_documents(
            ["How long do I have to return a product?",
             "Customers may request a refund within 30 days of purchase.",
             "The cat sat on the warm windowsill."]
        )
    except OSError as exc:  # offline and not cached
        pytest.skip(f"embedding model unavailable: {exc}")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert len(q) == 384
    assert dot(q, related) > dot(q, unrelated)


# ------------------------------------------------------------------ vector store


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "unit_test", "fake")


@pytest.fixture
def encoder() -> EmbeddingService:
    return EmbeddingService("fake", model=FakeEncoder())


def test_upsert_and_search_returns_metadata(store, encoder):
    chunks = [
        _chunk("docA", 0, "refund within thirty days", page=3),
        _chunk("docB", 0, "shipping from berlin warehouse", page=None),
    ]
    store.upsert_chunks(chunks, encoder.embed_documents([c.text for c in chunks]))

    hits = store.search(encoder.embed_query("refund thirty days"), top_k=2)
    assert hits[0].document_id == "docA"
    assert hits[0].page_number == 3
    assert hits[0].filename == "docA.pdf"
    assert hits[0].score > hits[1].score
    assert hits[1].page_number is None  # DOCX-style chunk without a page


def test_upsert_is_idempotent(store, encoder):
    chunks = [_chunk("docA", 0, "hello world")]
    vectors = encoder.embed_documents(["hello world"])
    store.upsert_chunks(chunks, vectors)
    store.upsert_chunks(chunks, vectors)
    assert store.count() == 1


def test_filter_by_document(store, encoder):
    chunks = [_chunk("docA", 0, "refund policy text"), _chunk("docB", 0, "refund policy text too")]
    store.upsert_chunks(chunks, encoder.embed_documents([c.text for c in chunks]))
    hits = store.search(encoder.embed_query("refund"), top_k=5, document_ids=["docB"])
    assert {h.document_id for h in hits} == {"docB"}


def test_search_empty_store(store, encoder):
    assert store.search(encoder.embed_query("anything"), top_k=5) == []


def test_delete_document_removes_only_its_vectors(store, encoder):
    chunks = [_chunk("docA", 0, "a one"), _chunk("docA", 1, "a two"), _chunk("docB", 0, "b one")]
    store.upsert_chunks(chunks, encoder.embed_documents([c.text for c in chunks]))
    store.delete_document("docA")
    assert store.count_for_document("docA") == 0
    assert store.count_for_document("docB") == 1


def test_model_mismatch_is_detected(tmp_path):
    VectorStore(tmp_path / "chroma", "shared", "model-one")
    with pytest.raises(EmbeddingModelMismatchError):
        VectorStore(tmp_path / "chroma", "shared", "model-two")
