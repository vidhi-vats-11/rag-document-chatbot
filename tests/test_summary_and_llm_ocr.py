"""'Summarize this document' reads the whole document; OCR_PROVIDER=llm reads images with the LLM."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import SecretStr

from app.core.config import OCRProvider
from app.domain import FileKind
from app.services.chat_service import is_summary_request
from app.services.extraction_service import ExtractionService
from app.services.llm_ocr import LLMImageReader
from tests.conftest import PDF, REFUND_TEXT, SHIPPING_TEXT, make_text_pdf, render_text_image, upload


def ask(client, question, **extra):
    return client.post("/api/v1/chat/ask", json={"question": question, **extra})


# ------------------------------------------------------------------ summaries


@pytest.mark.parametrize(
    "question",
    ["Summarize this document", "summarise the file", "Give me a summary", "What is this document about?",
     "What's the PDF about", "List the key points", "overview please", "tl;dr"],
)
def test_summary_requests_are_recognised(question):
    assert is_summary_request(question)


@pytest.mark.parametrize("question", ["What is the refund window?", "Who signed the contract?"])
def test_normal_questions_are_not_summary_requests(question):
    assert not is_summary_request(question)


def _three_page_pdf() -> bytes:
    return make_text_pdf([f"Page {i} topic: {word} details." for i, word in enumerate(["alpha", "bravo", "charlie"], 1)])


def test_summary_of_selected_document_sends_every_page_in_order(client, fake_llm):
    doc_id = upload(client, "three.pdf", _three_page_pdf(), PDF).json()["document"]["document_id"]
    upload(client, "policy.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), PDF)
    res = ask(client, "Summarize this document", document_id=doc_id).json()

    assert res["grounded"] is True
    prompt = fake_llm.calls[-1][1]
    assert prompt.index("alpha") < prompt.index("bravo") < prompt.index("charlie")
    assert "Refund" not in prompt  # only the selected document


def test_summary_works_with_all_documents_when_only_one_exists(client, fake_llm):
    upload(client, "three.pdf", _three_page_pdf(), PDF)
    res = ask(client, "Summarize this document").json()
    assert res["grounded"] is True
    assert all(word in fake_llm.calls[-1][1] for word in ("alpha", "bravo", "charlie"))


def test_summary_of_all_documents_still_uses_search(client, container, fake_llm, monkeypatch):
    # With several documents and none selected, "summarize" can't know which one: normal search.
    upload(client, "three.pdf", _three_page_pdf(), PDF)
    upload(client, "policy.pdf", make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), PDF)
    retrieval = container.chat_service.retrieval
    monkeypatch.setattr(retrieval, "document_chunks", lambda *a, **k: pytest.fail("whole-document path used"))
    assert ask(client, "Summarize the refund policy").json()["grounded"] is True


def test_summary_is_limited_to_summary_max_chunks(settings, embedder, fake_llm):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_container

    settings.summary_max_chunks = 2
    with TestClient(create_app(container=make_container(settings, embedder, fake_llm))) as client:
        doc_id = upload(client, "three.pdf", _three_page_pdf(), PDF).json()["document"]["document_id"]
        ask(client, "Summarize this document", document_id=doc_id)
    prompt = fake_llm.calls[-1][1]
    assert "alpha" in prompt and "bravo" in prompt and "charlie" not in prompt


# ------------------------------------------------------------------ OCR by the LLM


def test_extraction_uses_the_image_reader_for_images():
    seen = []
    reader = lambda image: seen.append(image.size) or "Meeting room B is closed."  # noqa: E731
    service = ExtractionService(tesseract_cmd="definitely-not-installed", image_reader=reader)
    assert service.ocr_available()
    result = service.extract(render_text_image("anything"), FileKind.IMAGE)
    assert result.pages[0].text == "Meeting room B is closed." and result.ocr_used
    assert len(seen) == 1


def test_extraction_uses_the_image_reader_for_scanned_pdf_pages():
    from tests.conftest import make_scanned_pdf

    service = ExtractionService(tesseract_cmd="definitely-not-installed", image_reader=lambda image: "Invoice total 42")
    result = service.extract(make_scanned_pdf("Invoice total 42"), FileKind.PDF)
    assert "Invoice total 42" in result.pages[0].text and result.ocr_used


def test_llm_image_reader_sends_a_small_jpeg_and_returns_the_text(settings):
    settings.llm_api_key = SecretStr("test-key")
    reader = LLMImageReader(settings)
    sent = {}

    def create(model, messages):
        sent["model"], sent["content"] = model, messages[0]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  Hello world \n"))])

    reader._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert reader(Image.new("RGB", (5000, 3000), "white")) == "Hello world"

    image_part = sent["content"][1]["image_url"]["url"]
    assert image_part.startswith("data:image/jpeg;base64,")
    import base64

    decoded = Image.open(io.BytesIO(base64.b64decode(image_part.split(",", 1)[1])))
    assert max(decoded.size) <= 2000  # big scans are shrunk before sending


def test_ocr_provider_llm_is_wired_by_the_container(settings, embedder, fake_llm):
    from tests.conftest import make_container

    settings.ocr_provider = OCRProvider.LLM
    settings.llm_api_key = SecretStr("test-key")
    container = make_container(settings, embedder, fake_llm)
    assert container.extractor.ocr_available()
    assert container.extractor.available_languages() == []  # the LLM reads any language
