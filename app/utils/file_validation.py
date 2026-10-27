"""Upload validation.

We never trust the client-provided filename or MIME type on their own.
The file's *actual bytes* ("magic numbers") decide what it is, and the
extension and declared content type must agree with that.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath

from app.core.exceptions import FileTooLargeError, FileValidationError, UnsupportedFileTypeError
from app.domain import FileKind

# extension -> file kind
ALLOWED_EXTENSIONS: dict[str, FileKind] = {
    ".pdf": FileKind.PDF,
    ".png": FileKind.IMAGE,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
    ".docx": FileKind.DOCX,
}

# Content types a browser may reasonably send for each extension.
# "application/octet-stream" is accepted because some clients send it for everything;
# the magic-byte check below is what actually decides.
_GENERIC = {"application/octet-stream", ""}
ALLOWED_CONTENT_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf", "application/x-pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg", "image/jpg", "image/pjpeg"},
    ".jpeg": {"image/jpeg", "image/jpg", "image/pjpeg"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
}

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"


@dataclass(frozen=True)
class ValidatedFile:
    original_filename: str  # sanitised, for display only — never used as a path
    extension: str
    kind: FileKind
    content: bytes


def sanitize_display_filename(filename: str | None) -> str:
    """Return a safe, human-readable name: no directories, no control chars, bounded length."""
    if not filename:
        return "unnamed"
    # Strip any directory components, whether Windows- or POSIX-style.
    name = PurePath(filename.replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip().lstrip(".")
    if len(name) > 200:
        stem, ext = Path(name).stem, Path(name).suffix
        name = stem[: 200 - len(ext)] + ext
    return name or "unnamed"


def detect_kind_from_bytes(content: bytes) -> tuple[FileKind, set[str]] | None:
    """Identify the file from its leading bytes. Returns (kind, compatible extensions)."""
    if content.startswith(_PDF_MAGIC):
        return FileKind.PDF, {".pdf"}
    if content.startswith(_PNG_MAGIC):
        return FileKind.IMAGE, {".png"}
    if content.startswith(_JPEG_MAGIC):
        return FileKind.IMAGE, {".jpg", ".jpeg"}
    if content.startswith(_ZIP_MAGIC) and _is_docx(content):
        return FileKind.DOCX, {".docx"}
    return None


def _is_docx(content: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return "word/document.xml" in zf.namelist()
    except zipfile.BadZipFile:
        return False


def validate_upload(
    filename: str | None,
    content_type: str | None,
    content: bytes,
    max_bytes: int,
) -> ValidatedFile:
    """Raise a FileValidationError subclass if the upload is not acceptable."""
    display_name = sanitize_display_filename(filename)
    extension = Path(display_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"Unsupported file extension '{extension or '(none)'}'. Allowed: {allowed}.")

    if len(content) == 0:
        raise FileValidationError("The uploaded file is empty.")
    if len(content) > max_bytes:
        raise FileTooLargeError(f"File exceeds the maximum upload size of {max_bytes / (1024 * 1024):.0f} MB.")

    declared = (content_type or "").split(";")[0].strip().lower()
    if declared not in ALLOWED_CONTENT_TYPES[extension] | _GENERIC:
        raise UnsupportedFileTypeError(
            f"Content type '{declared}' does not match the '{extension}' extension."
        )

    detected = detect_kind_from_bytes(content)
    if detected is None or extension not in detected[1]:
        raise UnsupportedFileTypeError(
            f"File content does not look like a valid '{extension}' file."
        )

    return ValidatedFile(
        original_filename=display_name,
        extension=extension,
        kind=detected[0],
        content=content,
    )
