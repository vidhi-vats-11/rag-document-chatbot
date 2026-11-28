"""Background indexing, OCR language choice, and the database upgrade."""

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.domain import DocumentStatus
from app.main import create_app
from app.repositories.document_repository import DocumentRepository
from tests.conftest import (
    PDF,
    PNG,
    REFUND_TEXT,
    make_container,
    make_text_pdf,
    render_text_image,
    requires_tesseract,
    upload,
)


def _wait_until_done(client, document_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = client.get(f"/api/v1/documents/{document_id}").json()
        if doc["status"] != "processing":
            return doc
        time.sleep(0.05)
    raise AssertionError("indexing did not finish in time")


@pytest.fixture
def background(settings, embedder, fake_llm):
    """A client whose uploads are indexed on the background worker, plus a gate that
    holds the worker inside extraction until the test releases it."""
    container = make_container(settings.model_copy(update={"background_indexing": True}), embedder, fake_llm)
    gate = threading.Event()
    real_extract = container.extractor.extract

    def slow_extract(*args, **kwargs):
        gate.wait(timeout=10)
        return real_extract(*args, **kwargs)

    container.extractor.extract = slow_extract
    with TestClient(create_app(container=container)) as client:
        yield client, gate
        gate.set()


# ------------------------------------------------------------------ background indexing


def test_upload_returns_immediately_then_finishes_in_background(background):
    client, gate = background
    res = upload(client, "big.pdf", make_text_pdf([REFUND_TEXT]), PDF)
    assert res.status_code == 202
    doc = res.json()["document"]
    assert doc["status"] == "processing"  # the browser is not kept waiting

    gate.set()
    done = _wait_until_done(client, doc["document_id"])
    assert done["status"] == "indexed" and done["chunk_count"] >= 1


def test_same_file_while_processing_is_not_started_twice(background):
    client, gate = background
    content = make_text_pdf([REFUND_TEXT])
    first = upload(client, "a.pdf", content, PDF).json()["document"]
    again = upload(client, "a.pdf", content, PDF)
    assert again.status_code == 200
    assert again.json()["already_indexed"] is True
    assert again.json()["document"]["document_id"] == first["document_id"]
    gate.set()


def test_delete_while_processing_is_409(background):
    client, gate = background
    doc = upload(client, "a.pdf", make_text_pdf([REFUND_TEXT]), PDF).json()["document"]
    res = client.delete(f"/api/v1/documents/{doc['document_id']}")
    assert res.status_code == 409
    assert res.json()["error"] == "document_busy"

    gate.set()
    _wait_until_done(client, doc["document_id"])
    assert client.delete(f"/api/v1/documents/{doc['document_id']}").status_code == 200


def test_unfinished_documents_resume_after_restart(container):
    service = container.document_service
    record, _ = service.upload("a.pdf", PDF, make_text_pdf([REFUND_TEXT]))
    # Pretend the server stopped half-way through indexing.
    record.status = DocumentStatus.PROCESSING
    record.chunk_count = 0
    service.documents.save(record)

    assert service.resume_unfinished() == 1
    assert service.get_document(record.document_id).status is DocumentStatus.INDEXED


def test_resume_marks_failed_when_file_is_gone(container):
    service = container.document_service
    record, _ = service.upload("a.pdf", PDF, make_text_pdf([REFUND_TEXT]))
    record.status = DocumentStatus.PROCESSING
    service.documents.save(record)
    service.files.delete(record.stored_filename)

    service.resume_unfinished()
    after = service.get_document(record.document_id)
    assert after.status is DocumentStatus.FAILED
    assert "missing" in after.error_message


def test_get_document_and_404(client):
    doc = upload(client, "a.pdf", make_text_pdf([REFUND_TEXT]), PDF).json()["document"]
    assert client.get(f"/api/v1/documents/{doc['document_id']}").json()["status"] == "indexed"
    assert client.get("/api/v1/documents/" + "0" * 32).status_code == 404


# ------------------------------------------------------------------ OCR languages


@requires_tesseract
def test_ocr_languages_endpoint_lists_installed_languages(client):
    body = client.get("/api/v1/documents/ocr-languages").json()
    assert "eng" in body["languages"]
    assert "osd" not in body["languages"]  # helper file, not a language
    assert body["default"] == "eng"


@requires_tesseract
def test_upload_rejects_unknown_ocr_language(client):
    res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("a.pdf", make_text_pdf([REFUND_TEXT]), PDF)},
        data={"ocr_language": "klingon"},
    )
    assert res.status_code == 400
    assert "not installed" in res.json()["message"]


def test_upload_rejects_malformed_ocr_language(client):
    for bad in ("eng;rm -rf", "a+b+c+d", "+"):
        res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("a.pdf", make_text_pdf([REFUND_TEXT]), PDF)},
            data={"ocr_language": bad},
        )
        assert res.status_code == 400, bad


@requires_tesseract
def test_chosen_ocr_language_is_used_and_stored(client, container):
    if "fra" not in container.extractor.available_languages():
        pytest.skip("French OCR data not installed")
    res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("note.png", render_text_image("Le remboursement est possible sous trente jours"), PNG)},
        data={"ocr_language": "fra+eng"},
    )
    doc = res.json()["document"]
    assert doc["status"] == "indexed"
    assert doc["ocr_language"] == "fra+eng"


# ------------------------------------------------------------------ database upgrade


def test_old_database_gets_new_column(tmp_path):
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as conn:  # the table as it was before ocr_language existed
        conn.execute(
            "CREATE TABLE documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, "
            "stored_filename TEXT NOT NULL, file_kind TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
            "status TEXT NOT NULL, created_at TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0, "
            "chunk_count INTEGER NOT NULL DEFAULT 0, ocr_used INTEGER NOT NULL DEFAULT 0, "
            "error_message TEXT, warnings TEXT NOT NULL DEFAULT '[]')"
        )
        conn.execute(
            "INSERT INTO documents VALUES ('" + "a" * 32 + "', 'x.pdf', 'x.pdf', 'pdf', 1, 'indexed', "
            "'2026-01-01', 1, 1, 0, NULL, '[]')"
        )
    records = DocumentRepository(db).list_all()
    assert len(records) == 1 and records[0].ocr_language is None
