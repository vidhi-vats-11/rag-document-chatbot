from tests.conftest import (
    DOCX,
    PDF,
    PNG,
    REFUND_TEXT,
    SHIPPING_TEXT,
    make_docx,
    make_scanned_pdf,
    make_text_pdf,
    render_text_image,
    requires_tesseract,
    upload,
)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True


def test_upload_pdf_indexes_document(client, container):
    res = upload(client, "policy.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), PDF)
    assert res.status_code == 201
    doc = res.json()["document"]
    assert doc["status"] == "indexed"
    assert doc["filename"] == "policy.pdf"
    assert doc["page_count"] == 2
    assert doc["chunk_count"] >= 2
    assert res.json()["already_indexed"] is False
    assert "stored_filename" not in doc  # server path is not exposed
    assert container.vector_store.count_for_document(doc["document_id"]) == doc["chunk_count"]


def test_upload_docx(client):
    res = upload(client, "notes.docx", make_docx([REFUND_TEXT]), DOCX)
    assert res.status_code == 201
    assert res.json()["document"]["status"] == "indexed"


@requires_tesseract
def test_upload_image_and_scanned_pdf(client):
    img = upload(client, "photo.png", render_text_image("Warranty lasts two years"), PNG).json()["document"]
    assert img["status"] == "indexed" and img["ocr_used"] is True

    scan = upload(client, "scan.pdf", make_scanned_pdf("Warranty lasts two years scanned"), PDF).json()["document"]
    assert scan["status"] == "indexed" and scan["ocr_used"] is True


def test_reupload_same_file_is_not_reembedded(client, container, monkeypatch):
    content = make_text_pdf([REFUND_TEXT])
    first = upload(client, "a.pdf", content, PDF).json()["document"]

    calls = []
    original = container.embedder.embed_documents
    monkeypatch.setattr(container.embedder, "embed_documents", lambda t: calls.append(t) or original(t))

    res = upload(client, "renamed.pdf", content, PDF)
    assert res.status_code == 200
    assert res.json()["already_indexed"] is True
    assert res.json()["document"]["document_id"] == first["document_id"]
    assert calls == []


def test_upload_rejects_bad_extension(client):
    res = upload(client, "script.sh", b"#!/bin/sh\necho hi", "text/x-sh")
    assert res.status_code == 415
    assert res.json()["error"] == "unsupported_file_type"


def test_upload_rejects_disguised_file(client):
    res = upload(client, "fake.pdf", b"not a pdf at all", PDF)
    assert res.status_code == 415


def test_upload_rejects_too_large(client):
    big = b"%PDF-1.4\n" + b"0" * (3 * 1024 * 1024)  # limit in tests is 2 MB
    res = upload(client, "big.pdf", big, PDF)
    assert res.status_code == 413
    assert res.json()["error"] == "file_too_large"


def test_upload_without_file_is_422(client):
    res = client.post("/api/v1/documents/upload")
    assert res.status_code == 422


def test_path_traversal_filename_is_neutralised(client, settings):
    res = upload(client, "../../../evil.pdf", make_text_pdf([REFUND_TEXT]), PDF)
    assert res.status_code == 201
    assert res.json()["document"]["filename"] == "evil.pdf"
    stored = list(settings.upload_dir.iterdir())
    assert len(stored) == 1 and stored[0].parent == settings.upload_dir.resolve()
    assert not (settings.upload_dir.parent.parent / "evil.pdf").exists()


def test_document_with_no_text_is_marked_failed(client, monkeypatch, container):
    # A scanned PDF with OCR disabled has no extractable text.
    monkeypatch.setattr(container.extractor, "ocr_available", lambda: False)
    res = upload(client, "blank-scan.pdf", make_scanned_pdf("hidden"), PDF)
    assert res.status_code == 201
    doc = res.json()["document"]
    assert doc["status"] == "failed"
    assert "No readable text" in doc["error_message"]
    assert doc["chunk_count"] == 0


def test_list_documents(client):
    upload(client, "a.pdf", make_text_pdf([REFUND_TEXT]), PDF)
    upload(client, "b.docx", make_docx([SHIPPING_TEXT]), DOCX)
    body = client.get("/api/v1/documents").json()
    assert body["total"] == 2
    assert {d["filename"] for d in body["documents"]} == {"a.pdf", "b.docx"}


def test_delete_removes_vectors_file_and_record(client, container, settings):
    doc = upload(client, "a.pdf", make_text_pdf([REFUND_TEXT]), PDF).json()["document"]
    doc_id = doc["document_id"]
    assert container.vector_store.count_for_document(doc_id) > 0
    assert any(settings.upload_dir.iterdir())

    res = client.delete(f"/api/v1/documents/{doc_id}")
    assert res.status_code == 200
    assert res.json() == {"document_id": doc_id, "deleted": True}

    assert container.vector_store.count_for_document(doc_id) == 0
    assert not any(settings.upload_dir.iterdir())
    assert client.get("/api/v1/documents").json()["total"] == 0


def test_delete_unknown_document_is_404(client):
    res = client.delete("/api/v1/documents/" + "0" * 32)
    assert res.status_code == 404
    assert res.json()["error"] == "document_not_found"


def test_delete_rejects_malformed_id(client):
    assert client.delete("/api/v1/documents/..%2F..%2Fetc").status_code in (404, 422)
    assert client.delete("/api/v1/documents/not-a-valid-id").status_code == 422


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "What would you like to know?" in res.text
