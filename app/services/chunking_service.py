"""Split extracted text into overlapping chunks.

Design choices:
- Chunks never span two pages, so every chunk has one accurate page number.
- We prefer to cut at a paragraph, then a sentence, then a word boundary,
  so chunks read naturally and words are not split in half.
- Chunk IDs are deterministic (`{document_id}:p{page}:c{index}`), so
  re-indexing the same document produces the same IDs (upsert, not duplicate).
"""

from __future__ import annotations

import re

from app.domain import ExtractedPage, TextChunk

# Preferred cut points, best first. Each is searched for in the last part of the window.
_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into pieces of at most `chunk_size` characters.

    Consecutive pieces share roughly `chunk_overlap` characters, so a sentence
    that falls on a boundary still appears whole in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_break(text, start, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Step back by the overlap, but always move forward, and start on a word boundary.
        next_start = max(end - chunk_overlap, start + 1)
        if next_start < end and not text[next_start - 1].isspace():
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return chunks


def _find_break(text: str, start: int, end: int, chunk_size: int) -> int:
    """Find the best cut point in the second half of the window [start, end)."""
    search_from = start + chunk_size // 2
    for boundary in _BOUNDARIES:
        pos = text.rfind(boundary, search_from, end)
        if pos != -1:
            return pos + len(boundary)
    return end  # no natural break (e.g. a very long token): hard cut


class ChunkingService:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self, document_id: str, filename: str, pages: list[ExtractedPage]
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for page in pages:
            pieces = split_text(normalize_text(page.text), self.chunk_size, self.chunk_overlap)
            page_label = page.page_number if page.page_number is not None else 0
            for local_index, piece in enumerate(pieces):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}:p{page_label}:c{local_index}",
                        document_id=document_id,
                        filename=filename,
                        text=piece,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                    )
                )
        return chunks
