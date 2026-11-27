"""Answer cache: a repeated question reuses the earlier answer instead of calling the LLM again."""

from __future__ import annotations

from app.core.exceptions import LLMTimeoutError
from app.domain import ChatTurn, RetrievedChunk
from app.repositories.answer_cache_repository import AnswerCacheRepository
from app.services.answer_cache_service import AnswerCacheService, depends_on_chat, normalise
from app.services.embedding_service import EmbeddingService
from tests.conftest import PDF, REFUND_TEXT, SHIPPING_TEXT, FakeEncoder, make_text_pdf, upload


def ask(client, question, **extra):
    return client.post("/api/v1/chat/ask", json={"question": question, **extra})


def _index_policy(client) -> str:
    res = upload(client, "policy.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), PDF)
    return res.json()["document"]["document_id"]


# ------------------------------------------------------------------ through the API


def test_same_question_again_is_answered_from_cache(client, fake_llm):
    _index_policy(client)
    first = ask(client, "What is the refund window?").json()
    second = ask(client, "What is the refund window?").json()

    assert len(fake_llm.calls) == 1  # the LLM was asked only once
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["llm_model"] == first["llm_model"]
    assert [s["chunk_id"] for s in second["sources"]] == [s["chunk_id"] for s in first["sources"]]
    assert second["sources"]  # sources come back too, from the current search results


def test_case_spacing_and_punctuation_do_not_matter(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    res = ask(client, "  what is the REFUND window ").json()
    assert res["cached"] is True
    assert len(fake_llm.calls) == 1


def test_different_question_is_not_served_from_cache(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    res = ask(client, "How much does shipping cost?").json()
    assert res["cached"] is False
    assert len(fake_llm.calls) == 2


def test_other_llm_model_does_not_reuse_answer(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    res = ask(client, "What is the refund window?", llm_model="fake-llm-two").json()
    assert res["cached"] is False and res["llm_model"] == "fake-llm-two"
    assert len(fake_llm.calls) == 2


def test_other_search_scope_does_not_reuse_answer(client, fake_llm):
    doc_id = _index_policy(client)
    ask(client, "What is the refund window?")
    res = ask(client, "What is the refund window?", document_id=doc_id).json()
    assert res["cached"] is False
    assert len(fake_llm.calls) == 2


def test_same_question_again_in_the_same_chat_is_cached(client, fake_llm):
    # The web page sends earlier turns with every question, so repeats happen mid-chat.
    _index_policy(client)
    first = ask(client, "What is the refund window?").json()
    history = [{"question": "What is the refund window?", "answer": first["answer"]}]
    res = ask(client, "What is the refund window?", history=history).json()
    assert res["cached"] is True
    assert len(fake_llm.calls) == 1


def test_chat_dependent_follow_up_is_cached_only_after_the_same_question(client, fake_llm):
    _index_policy(client)
    after_refund = [{"question": "What is the refund window?", "answer": "30 days [1]."}]
    ask(client, "Does that refund go to the original payment method?", history=after_refund)
    assert ask(client, "Does that refund go to the original payment method?", history=after_refund).json()["cached"] is True
    assert len(fake_llm.calls) == 1


def test_depends_on_chat():
    assert depends_on_chat("What is his email?")
    assert depends_on_chat("And shipping?")  # very short
    assert depends_on_chat("Is that still valid?")
    assert not depends_on_chat("What is the refund window?")


def test_chat_dependent_question_needs_the_same_previous_question(tmp_path):
    cache = _service(tmp_path)
    chunks = [_chunk("a"), _chunk("b")]
    after_ceo = [ChatTurn("Who is the CEO?", "Asha [1].")]
    after_cto = [ChatTurn("Who is the CTO?", "Ravi [2].")]
    cache.store("What is his email?", None, "m", chunks, "asha@example.com [1].", [chunks[0]], after_ceo)

    assert cache.lookup("What is his email?", None, "m", chunks, after_ceo).answer == "asha@example.com [1]."
    # Same words, same passages, but "his" now means someone else: must not reuse.
    assert cache.lookup("What is his email?", None, "m", chunks, after_cto) is None
    assert cache.lookup("What is his email?", None, "m", chunks) is None


def test_self_contained_question_ignores_the_chat_before_it(tmp_path):
    cache = _service(tmp_path)
    chunks = [_chunk("a")]
    cache.store("What is the refund window?", None, "m", chunks, "30 days.", [])
    earlier = [ChatTurn("How much does shipping cost?", "5 dollars.")]
    assert cache.lookup("What is the refund window?", None, "m", chunks, earlier).answer == "30 days."


def test_use_cache_false_always_asks_the_llm(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    res = ask(client, "What is the refund window?", use_cache=False).json()
    assert res["cached"] is False
    assert len(fake_llm.calls) == 2


def test_new_relevant_document_makes_old_answer_stale(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    upload(client, "update.pdf", make_text_pdf(["Refund window update: the refund window is now 60 days."]), PDF)
    res = ask(client, "What is the refund window?").json()
    assert res["cached"] is False  # the search found different passages, so the cache is skipped
    assert len(fake_llm.calls) == 2


def test_deleted_document_answer_is_not_reused(client, fake_llm):
    doc_id = _index_policy(client)
    ask(client, "What is the refund window?")
    client.delete(f"/api/v1/documents/{doc_id}")
    upload(client, "other.pdf", make_text_pdf(["Refund window: returns are accepted for 14 days."]), PDF)
    res = ask(client, "What is the refund window?").json()
    assert res["cached"] is False
    assert len(fake_llm.calls) == 2


def test_failed_llm_calls_are_not_cached(client, fake_llm):
    _index_policy(client)
    fake_llm.error = LLMTimeoutError("slow")
    assert ask(client, "What is the refund window?").status_code == 504
    fake_llm.error = None
    res = ask(client, "What is the refund window?").json()
    assert res["cached"] is False
    assert len(fake_llm.calls) == 2


def test_clear_cache_endpoint(client, fake_llm):
    _index_policy(client)
    ask(client, "What is the refund window?")
    ask(client, "How much does shipping cost?")
    res = client.delete("/api/v1/chat/cache")
    assert res.status_code == 200 and res.json() == {"removed": 2}
    assert ask(client, "What is the refund window?").json()["cached"] is False


def test_cache_can_be_switched_off(settings, embedder, fake_llm):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_container

    settings.answer_cache_enabled = False
    with TestClient(create_app(container=make_container(settings, embedder, fake_llm))) as client:
        _index_policy(client)
        ask(client, "What is the refund window?")
        assert ask(client, "What is the refund window?").json()["cached"] is False
    assert len(fake_llm.calls) == 2


# ------------------------------------------------------------------ service and repository


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id="d", filename="f.pdf", text="t", page_number=1, score=0.5)


def _service(tmp_path, **kwargs) -> AnswerCacheService:
    repo = AnswerCacheRepository(tmp_path / "cache.sqlite3")
    return AnswerCacheService(repo, EmbeddingService("fake", model=FakeEncoder()), **kwargs)


def test_normalise():
    assert normalise("  How long do I have, to get a REFUND?? ") == "how long do i have to get a refund"


def test_same_meaning_with_different_words_order_is_a_hit(tmp_path):
    # FakeEncoder is bag-of-words, so these two get identical vectors (similarity 1.0).
    cache = _service(tmp_path)
    chunks = [_chunk("a"), _chunk("b")]
    cache.store("refund window length", None, "m", chunks, "30 days [1].", [chunks[0]])
    hit = cache.lookup("length of refund window", None, "m", list(reversed(chunks)))  # order doesn't matter
    assert hit is not None and hit.answer == "30 days [1]." and hit.similarity > 0.99
    assert [s.chunk_id for s in hit.sources] == ["a"]


def test_similarity_below_threshold_is_a_miss(tmp_path):
    cache = _service(tmp_path, min_similarity=0.95)
    chunks = [_chunk("a")]
    cache.store("refund window length", None, "m", chunks, "30 days.", [])
    assert cache.lookup("refund window length policy details", None, "m", chunks) is None


def test_oldest_entries_are_dropped_above_max(tmp_path):
    cache = _service(tmp_path, max_entries=2)
    for i in range(3):
        cache.store(f"question number {i}", None, "m", [_chunk("a")], f"answer {i}", [])
    assert cache.size() == 2
    assert cache.lookup("question number 0", None, "m", [_chunk("a")]) is None
    assert cache.lookup("question number 2", None, "m", [_chunk("a")]).answer == "answer 2"
