"""Text extraction for PDFs (digital and scanned), images, and DOCX files."""

from __future__ import annotations

import io
import logging
import re
import shutil

import pymupdf
import pytesseract
from docx import Document as DocxDocument
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.exceptions import ExtractionError, FileValidationError
from app.domain import ExtractedPage, ExtractionResult, FileKind

logger = logging.getLogger(__name__)

# Guard against "decompression bomb" images (tiny file, gigantic pixel count).
Image.MAX_IMAGE_PIXELS = 100_000_000

# Tesseract data files that are helpers, not languages people would pick.
_NOT_LANGUAGES = {"osd", "snum", "equ"}
_LANGUAGE_CODE = re.compile(r"^[A-Za-z_]{2,30}$")
MAX_OCR_LANGUAGES = 3  # each extra language makes OCR slower


class ExtractionService:
    def __init__(
        self,
        ocr_language: str = "eng",
        ocr_dpi: int = 300,
        ocr_min_chars_per_page: int = 50,
        tesseract_cmd: str | None = None,
    ) -> None:
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.ocr_min_chars_per_page = ocr_min_chars_per_page
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_cmd = tesseract_cmd or "tesseract"

    # ------------------------------------------------------------------ public

    def ocr_available(self) -> bool:
        return shutil.which(self._tesseract_cmd) is not None

    def available_languages(self) -> list[str]:
        """Language codes Tesseract has installed (e.g. ["eng", "fra", "hin"])."""
        if not self.ocr_available():
            return []
        try:
            installed = pytesseract.get_languages(config="")
        except Exception:  # broken install: treat it as "no languages"
            logger.warning("Could not list Tesseract languages", exc_info=True)
            return []
        return sorted(code for code in installed if code not in _NOT_LANGUAGES)

    def normalize_language(self, language: str | None) -> str:
        """Check a user-chosen language string like "hin+eng" and return it cleaned up.

        Raises FileValidationError if a code is malformed or not installed, so a bad
        choice is reported straight away instead of failing later in the background.
        """
        if language is None or not language.strip():
            return self.ocr_language
        codes = [c.strip() for c in language.split("+") if c.strip()]
        if not codes or len(codes) > MAX_OCR_LANGUAGES:
            raise FileValidationError(f"Choose between 1 and {MAX_OCR_LANGUAGES} OCR languages.")
        if not all(_LANGUAGE_CODE.match(c) for c in codes):
            raise FileValidationError("Invalid OCR language code.")
        installed = set(self.available_languages())
        missing = [c for c in codes if c not in installed]
        if installed and missing:
            raise FileValidationError(f"OCR language not installed on the server: {', '.join(missing)}.")
        return "+".join(dict.fromkeys(codes))  # drop duplicates, keep order

    def extract(self, content: bytes, kind: FileKind, ocr_language: str | None = None) -> ExtractionResult:
        language = ocr_language or self.ocr_language
        if kind is FileKind.PDF:
            return self._extract_pdf(content, language)
        if kind is FileKind.IMAGE:
            return self._extract_image(content, language)
        if kind is FileKind.DOCX:
            return self._extract_docx(content)
        raise ExtractionError(f"No extractor for file kind '{kind}'.")

    # ------------------------------------------------------------------ PDF

    def _extract_pdf(self, content: bytes, language: str) -> ExtractionResult:
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:  # PyMuPDF raises several error types for bad files
            raise ExtractionError("The PDF could not be opened. It may be corrupted.") from exc

        if pdf.needs_pass:
            pdf.close()
            raise ExtractionError("Password-protected PDFs are not supported.")

        pages: list[ExtractedPage] = []
        warnings: list[str] = []
        ocr_ok = self.ocr_available()
        ocr_missing_pages: list[int] = []

        with pdf:
            for index, page in enumerate(pdf):
                page_number = index + 1
                text = page.get_text("text") or ""

                # Little or no embedded text: probably a scanned page, so OCR it.
                if len(text.strip()) < self.ocr_min_chars_per_page:
                    if ocr_ok:
                        ocr_text = self._ocr_pdf_page(page, language)
                        if len(ocr_text.strip()) > len(text.strip()):
                            pages.append(ExtractedPage(ocr_text, page_number, ocr_used=True))
                            continue
                    else:
                        ocr_missing_pages.append(page_number)

                pages.append(ExtractedPage(text, page_number))

        if ocr_missing_pages:
            warnings.append(
                f"Tesseract OCR is not installed, so {len(ocr_missing_pages)} text-poor page(s) "
                "could not be OCR'd."
            )
        return ExtractionResult(pages=pages, warnings=warnings)

    def _ocr_pdf_page(self, page: "pymupdf.Page", language: str) -> str:
        pixmap = page.get_pixmap(dpi=self.ocr_dpi)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return self._run_ocr(image, language)

    # ------------------------------------------------------------------ images

    def _extract_image(self, content: bytes, language: str) -> ExtractionResult:
        if not self.ocr_available():
            raise ExtractionError(
                "Tesseract OCR is not installed on the server, so text cannot be read from images."
            )
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ExtractionError("The image could not be read. It may be corrupted or too large.") from exc

        image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
        text = self._run_ocr(image, language)
        return ExtractionResult(pages=[ExtractedPage(text, page_number=1, ocr_used=True)])

    def _run_ocr(self, image: Image.Image, language: str) -> str:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        try:
            return pytesseract.image_to_string(image, lang=language)
        except pytesseract.TesseractError as exc:
            logger.warning("Tesseract failed: %s", exc)
            raise ExtractionError("OCR failed while reading the document.") from exc

    # ------------------------------------------------------------------ DOCX

    def _extract_docx(self, content: bytes) -> ExtractionResult:
        try:
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise ExtractionError("The DOCX file could not be opened. It may be corrupted.") from exc

        blocks: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
        # Tables are common in business documents; include their cell text row by row.
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))

        # DOCX has no fixed pages (layout depends on the renderer), so page_number is None.
        return ExtractionResult(
            pages=[ExtractedPage("\n\n".join(blocks), page_number=None)],
            warnings=["DOCX files have no fixed page numbers; sources will show the filename only."],
        )
