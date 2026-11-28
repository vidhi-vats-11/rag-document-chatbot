import pytest

from app.core.exceptions import ExtractionError
from app.domain import FileKind
from app.services.extraction_service import ExtractionService
from tests.conftest import (
    REFUND_TEXT,
    SHIPPING_TEXT,
    make_docx,
    make_scanned_pdf,
    make_text_pdf,
    render_text_image,
    requires_tesseract,
)


@pytest.fixture
def extractor() -> ExtractionService:
    return ExtractionService()


def test_text_pdf_preserves_page_numbers(extractor):
    result = extractor.extract(make_text_pdf([REFUND_TEXT, SHIPPING_TEXT]), FileKind.PDF)
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "30 days" in result.pages[0].text
    assert "Berlin" in result.pages[1].text
    assert not result.ocr_used


@requires_tesseract
def test_scanned_pdf_uses_ocr(extractor):
    result = extractor.extract(make_scanned_pdf("Invoice total due is 450 euros"), FileKind.PDF)
    assert result.ocr_used
    assert result.pages[0].page_number == 1
    assert "invoice" in result.pages[0].text.lower()


@requires_tesseract
def test_image_ocr(extractor):
    result = extractor.extract(render_text_image("Meeting room booked for Friday"), FileKind.IMAGE)
    assert result.ocr_used
    assert "friday" in result.pages[0].text.lower()


def test_scanned_pdf_without_tesseract_warns(monkeypatch, extractor):
    monkeypatch.setattr(extractor, "ocr_available", lambda: False)
    result = extractor.extract(make_scanned_pdf("anything"), FileKind.PDF)
    assert result.total_chars == 0
    assert any("OCR is not available" in w for w in result.warnings)


def test_image_without_tesseract_raises(monkeypatch, extractor):
    monkeypatch.setattr(extractor, "ocr_available", lambda: False)
    with pytest.raises(ExtractionError, match="OCR is not available"):
        extractor.extract(render_text_image("x"), FileKind.IMAGE)


def test_docx_paragraphs_and_tables(extractor):
    from docx import Document
    import io

    doc = Document()
    doc.add_paragraph("First paragraph about budgets.")
    doc.add_paragraph("")  # empty paragraphs are skipped
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Q1"
    table.rows[0].cells[1].text = "1200 EUR"
    buf = io.BytesIO()
    doc.save(buf)

    result = extractor.extract(buf.getvalue(), FileKind.DOCX)
    assert len(result.pages) == 1
    assert result.pages[0].page_number is None
    assert "First paragraph about budgets." in result.pages[0].text
    assert "Q1 | 1200 EUR" in result.pages[0].text


def test_docx_helper(extractor):
    result = extractor.extract(make_docx([REFUND_TEXT]), FileKind.DOCX)
    assert "30 days" in result.pages[0].text


def test_corrupted_pdf_raises(extractor):
    with pytest.raises(ExtractionError):
        extractor.extract(b"%PDF-1.4 this is not really a pdf", FileKind.PDF)
