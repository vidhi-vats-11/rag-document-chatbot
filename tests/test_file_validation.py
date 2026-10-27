import pytest

from app.core.exceptions import FileTooLargeError, FileValidationError, UnsupportedFileTypeError
from app.domain import FileKind
from app.utils.file_validation import sanitize_display_filename, validate_upload
from tests.conftest import DOCX, PDF, PNG, make_docx, make_text_pdf, render_text_image

MAX = 1024 * 1024


def test_accepts_valid_pdf():
    result = validate_upload("report.pdf", PDF, make_text_pdf(["hello"]), MAX)
    assert result.kind is FileKind.PDF
    assert result.extension == ".pdf"


def test_accepts_png_and_docx():
    assert validate_upload("scan.PNG", PNG, render_text_image("hi"), MAX).kind is FileKind.IMAGE
    assert validate_upload("notes.docx", DOCX, make_docx(["hi"]), MAX).kind is FileKind.DOCX


def test_accepts_generic_octet_stream_when_bytes_match():
    assert validate_upload("a.pdf", "application/octet-stream", make_text_pdf(["x"]), MAX).kind is FileKind.PDF


def test_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("malware.exe", "application/octet-stream", b"MZ\x90\x00", MAX)


def test_rejects_disguised_file():
    """A text file renamed to .pdf must be rejected: we check the bytes, not the name."""
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("fake.pdf", PDF, b"just some text pretending to be a pdf", MAX)


def test_rejects_extension_content_mismatch():
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("image.pdf", PDF, render_text_image("hi"), MAX)


def test_rejects_mismatched_declared_content_type():
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("report.pdf", "image/png", make_text_pdf(["x"]), MAX)


def test_rejects_plain_zip_named_docx():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "not a word document")
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("x.docx", DOCX, buf.getvalue(), MAX)


def test_rejects_empty_and_oversized():
    with pytest.raises(FileValidationError):
        validate_upload("a.pdf", PDF, b"", MAX)
    with pytest.raises(FileTooLargeError):
        validate_upload("a.pdf", PDF, b"%PDF-" + b"0" * 100, max_bytes=50)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
        ("normal name.pdf", "normal name.pdf"),
        ("bad\x00name\n.pdf", "badname.pdf"),
        ("", "unnamed"),
        (None, "unnamed"),
    ],
)
def test_sanitize_display_filename(raw, expected):
    assert sanitize_display_filename(raw) == expected
