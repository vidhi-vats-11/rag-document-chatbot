"""Choosing the embedding model per upload, the LLM model per question, and re-ranking."""

from __future__ import annotations

from app.domain import RetrievedChunk
from app.repositories.document_repository import DocumentRepository
from app.services.rerank_service import RerankService
from tests.conftest import PDF, REFUND_TEXT, SHIPPING_TEXT, FakeReranker, make_text_pdf


def upload_with_model(client, filename, content, model=None):
    data = {"embedding_model": model} if model else {}
    return client.post(
        "/api/v1/documents/upload", files={"file": (filename, content, PDF)}, data=data
    )


def ask(client, question, **extra):
    return client.post("/api/v1/chat/ask", json={"question": question, **extra})


# ------------------------------------------------------------------ models endpoint


def test_models_endpoint_lists_choices_default_first(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["embedding_models"] == ["fake-encoder", "fake-encoder-two"]
    assert body["default_embedding_model"] == "fake-encoder"
    assert body["llm_models"] == ["fake-llm", "fake-llm-two"]
    assert body["default_llm_model"] == "fake-llm"
    assert body["rerank_by_default"] is False


# ------------------------------------------------------------------ embedding model per upload


def test_upload_uses_default_embedding_model(client):
    doc = upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT])).json()["document"]
    assert doc["embedding_model"] == "fake-encoder"


def test_upload_with_chosen_model_stores_vectors_in_its_own_collection(client, container):
    doc = upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]), "fake-encoder-two").json()["document"]
    assert doc["status"] == "indexed"
    assert doc["embedding_model"] == "fake-encoder-two"

    other = container.spaces.get("fake-encoder-two").store
    assert other.count_for_document(doc["document_id"]) == doc["chunk_count"]
    assert container.vector_store.count_for_document(doc["document_id"]) == 0  # default collection untouched


def test_unknown_embedding_model_is_rejected(client):
    res = upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]), "evil/model")
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_model"


def test_reupload_with_other_model_reindexes(client, container):
    content = make_text_pdf([REFUND_TEXT])
    first = upload_with_model(client, "refund.pdf", content).json()["document"]
    same = upload_with_model(client, "refund.pdf", content)
    assert same.json()["already_indexed"] is True

    res = upload_with_model(client, "refund.pdf", content, "fake-encoder-two")
    assert res.json()["already_indexed"] is False
    assert res.json()["document"]["embedding_model"] == "fake-encoder-two"
    doc_id = first["document_id"]
    assert container.vector_store.count_for_document(doc_id) == 0  # old vectors removed
    assert container.spaces.get("fake-encoder-two").store.count_for_document(doc_id) > 0


def test_question_searches_documents_of_every_model(client, fake_llm):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    upload_with_model(client, "shipping.pdf", make_text_pdf([SHIPPING_TEXT]), "fake-encoder-two")

    res = ask(client, "How long does standard delivery take from the Berlin warehouse?")
    assert res.status_code == 200
    assert "Berlin warehouse" in fake_llm.calls[0][1]  # found via the second model's collection


def test_question_scoped_to_one_document_uses_its_model(client, fake_llm):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    ship = upload_with_model(client, "shipping.pdf", make_text_pdf([SHIPPING_TEXT]), "fake-encoder-two")
    doc_id = ship.json()["document"]["document_id"]

    ask(client, "delivery refund warehouse", document_id=doc_id)
    prompt = fake_llm.calls[0][1]
    assert "Berlin warehouse" in prompt
    assert "Refund Policy" not in prompt


def test_delete_removes_vectors_from_the_documents_model(client, container):
    doc = upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]), "fake-encoder-two").json()["document"]
    assert client.delete(f"/api/v1/documents/{doc['document_id']}").status_code == 200
    assert container.spaces.get("fake-encoder-two").store.count_for_document(doc["document_id"]) == 0


def test_old_records_without_model_use_the_default(settings, container):
    repo = DocumentRepository(settings.metadata_db_path)
    record = container.document_service.upload("refund.pdf", PDF, make_text_pdf([REFUND_TEXT]))[0]
    record.embedding_model = None  # as saved by an older version of the app
    repo.save(record)
    assert container.document_service.get_document(record.document_id).embedding_model == "fake-encoder"


# ------------------------------------------------------------------ LLM model per question


def test_default_llm_model_is_used_and_reported(client, fake_llm):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    res = ask(client, "What is the refund window?")
    assert res.json()["llm_model"] == "fake-llm"
    assert fake_llm.models == ["fake-llm"]


def test_chosen_llm_model_is_passed_to_the_llm(client, fake_llm):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    res = ask(client, "What is the refund window?", llm_model="fake-llm-two")
    assert res.status_code == 200
    assert res.json()["llm_model"] == "fake-llm-two"
    assert fake_llm.models == ["fake-llm-two"]


def test_unknown_llm_model_is_rejected_before_calling_llm(client, fake_llm):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    res = ask(client, "What is the refund window?", llm_model="gpt-expensive")
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_model"
    assert fake_llm.calls == []


# ------------------------------------------------------------------ re-ranking


def _chunk(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(cid, "doc", "doc.pdf", text, 1, score)


def test_rerank_service_reorders_and_trims():
    service = RerankService("fake", model=FakeReranker())
    chunks = [
        _chunk("a", "shipping from berlin", 0.9),  # high vector score, but off-topic
        _chunk("b", "refund within thirty days", 0.4),
        _chunk("c", "refund policy", 0.5),
    ]
    ranked = service.rerank("refund thirty days", chunks, top_k=2)
    assert [c.chunk_id for c in ranked] == ["b", "c"]
    assert ranked[0].rerank_score == 1.0
    assert ranked[0].score == 0.4  # the original cosine score is kept


def test_rerank_service_handles_no_chunks():
    assert RerankService("fake", model=FakeReranker()).rerank("q", [], top_k=5) == []


def test_ask_with_rerank_returns_rerank_scores(client, fake_reranker):
    upload_with_model(client, "policies.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]))
    res = ask(client, "How long is the refund window?", rerank=True)
    body = res.json()
    assert body["reranked"] is True
    assert body["sources"] and body["sources"][0]["rerank_score"] is not None
    assert len(fake_reranker.calls) == 1


def test_ask_without_rerank_skips_the_reranker(client, fake_reranker):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    body = ask(client, "How long is the refund window?", rerank=False).json()
    assert body["reranked"] is False
    assert body["sources"][0]["rerank_score"] is None
    assert fake_reranker.calls == []


def test_rerank_considers_more_candidates_than_top_k(client, container, fake_reranker):
    pages = [f"Section {i}. Refund note number {i} about store credit." for i in range(8)]
    upload_with_model(client, "many.pdf", make_text_pdf(pages))
    ask(client, "refund store credit", rerank=True, top_k=2)
    assert len(fake_reranker.calls[0]) > 2  # candidates scored before trimming to top_k


def test_unrelated_question_skips_rerank_and_llm(client, fake_llm, fake_reranker):
    upload_with_model(client, "refund.pdf", make_text_pdf([REFUND_TEXT]))
    body = ask(client, "Explain quantum chromodynamics gluon confinement", rerank=True).json()
    assert body["grounded"] is False
    assert fake_reranker.calls == []
    assert fake_llm.calls == []
