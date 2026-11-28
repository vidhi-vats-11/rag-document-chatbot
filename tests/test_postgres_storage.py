"""Postgres storage (DATABASE_URL), tested against a real database.

Runs only when TEST_DATABASE_URL is set. Each run uses its own temporary schema (a separate
area inside the database) and drops it afterwards, so real data is never touched.

The key test starts TWO copies of the app on the same database, like Vercel does: a file
uploaded through one copy must be visible to, answerable by, and deletable through the other.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import create_app
from tests.conftest import PDF, REFUND_TEXT, SHIPPING_TEXT, FakeLLM, make_container, make_text_pdf, upload

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def pg_settings(settings):
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    joiner = "&" if "?" in DB_URL else "?"
    settings.database_url = SecretStr(f"{DB_URL}{joiner}options={quote(f'-c search_path={schema},public')}")
    yield settings
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _client(settings, embedder, llm) -> TestClient:
    return TestClient(create_app(container=make_container(settings, embedder, llm)))


def ask(client, question, **extra):
    return client.post("/api/v1/chat/ask", json={"question": question, **extra})


def test_two_copies_of_the_app_share_documents(pg_settings, embedder):
    llm_one, llm_two = FakeLLM(), FakeLLM()
    with _client(pg_settings, embedder, llm_one) as one, _client(pg_settings, embedder, llm_two) as two:
        res = upload(one, "policy.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), PDF)
        assert res.status_code == 201
        doc_id = res.json()["document"]["document_id"]

        # Copy two sees the upload made through copy one...
        assert [d["document_id"] for d in two.get("/api/v1/documents").json()["documents"]] == [doc_id]
        # ...can answer from it, with sources...
        answer = ask(two, "What is the refund window?", document_id=doc_id).json()
        assert answer["grounded"] is True and answer["sources"]
        # ...shares the answer cache with copy one...
        assert ask(one, "What is the refund window?", document_id=doc_id).json()["cached"] is True
        assert len(llm_one.calls) == 0 and len(llm_two.calls) == 1
        # ...and can delete it, after which it is gone for both.
        assert two.delete(f"/api/v1/documents/{doc_id}").status_code == 200
        assert one.get("/api/v1/documents").json()["documents"] == []
        assert one.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_data_survives_a_restart(pg_settings, embedder, fake_llm):
    with _client(pg_settings, embedder, fake_llm) as first:
        doc_id = upload(first, "policy.pdf", make_text_pdf([REFUND_TEXT]), PDF).json()["document"]["document_id"]
    with _client(pg_settings, embedder, fake_llm) as restarted:
        assert restarted.get(f"/api/v1/documents/{doc_id}").json()["status"] == "indexed"
        assert ask(restarted, "What is the refund window?").json()["grounded"] is True


def test_summary_reads_the_whole_document_in_order(pg_settings, embedder, fake_llm):
    pages = [f"Page {i} topic: {word} details." for i, word in enumerate(["alpha", "bravo", "charlie"], 1)]
    with _client(pg_settings, embedder, fake_llm) as client:
        doc_id = upload(client, "three.pdf", make_text_pdf(pages), PDF).json()["document"]["document_id"]
        assert ask(client, "Summarize this document", document_id=doc_id).json()["grounded"] is True
    prompt = fake_llm.calls[-1][1]
    assert prompt.index("alpha") < prompt.index("bravo") < prompt.index("charlie")


def test_search_scores_and_document_filter(pg_settings, embedder, fake_llm):
    with _client(pg_settings, embedder, fake_llm) as client:
        refund = upload(client, "refund.pdf", make_text_pdf([REFUND_TEXT]), PDF).json()["document"]["document_id"]
        upload(client, "shipping.pdf", make_text_pdf([SHIPPING_TEXT]), PDF)
        res = ask(client, "What is the refund window?", document_id=refund).json()
    assert {s["document_id"] for s in res["sources"]} == {refund}
    assert all(0 < s["score"] <= 1 for s in res["sources"])
