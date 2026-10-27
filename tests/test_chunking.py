import pytest

from app.domain import ExtractedPage
from app.services.chunking_service import ChunkingService, normalize_text, split_text


def _long_text(n_sentences: int = 60) -> str:
    return " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(n_sentences))


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ", 100, 10) == []


def test_chunks_respect_size_limit():
    chunks = split_text(_long_text(), chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(_long_text(), chunk_size=200, chunk_overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        # The start of the next chunk should already appear at the end of the previous one.
        head = nxt[:20]
        assert head in prev, f"no overlap between:\n{prev!r}\n{nxt!r}"


def test_zero_overlap_covers_all_words_once():
    text = _long_text()
    chunks = split_text(text, chunk_size=200, chunk_overlap=0)
    assert " ".join(chunks).split() == text.split()


def test_prefers_sentence_boundaries():
    chunks = split_text(_long_text(), chunk_size=200, chunk_overlap=0)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_does_not_split_words():
    text = "alpha " * 200
    for chunk in split_text(text, 100, 20):
        assert set(chunk.split()) == {"alpha"}


def test_hard_cut_when_no_boundary():
    chunks = split_text("x" * 450, 100, 10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 5


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        ChunkingService(100, 150)


def test_normalize_text():
    assert normalize_text("a  \t b\r\n\n\n\nc ") == "a b\n\nc"


def test_chunk_pages_metadata_and_deterministic_ids():
    service = ChunkingService(chunk_size=200, chunk_overlap=20)
    pages = [ExtractedPage(_long_text(10), 1), ExtractedPage(_long_text(10), 2)]
    chunks = service.chunk_pages("doc123", "file.pdf", pages)

    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.document_id == "doc123" and c.filename == "file.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique
    assert chunks[0].chunk_id == "doc123:p1:c0"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    again = service.chunk_pages("doc123", "file.pdf", pages)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]


def test_chunk_never_spans_pages():
    service = ChunkingService(chunk_size=1000, chunk_overlap=100)
    chunks = service.chunk_pages("d", "f.pdf", [ExtractedPage("Page one text.", 1), ExtractedPage("Page two text.", 2)])
    assert [c.text for c in chunks] == ["Page one text.", "Page two text."]


def test_docx_chunks_have_no_page():
    service = ChunkingService(chunk_size=200, chunk_overlap=20)
    chunks = service.chunk_pages("d", "f.docx", [ExtractedPage("Some text.", None)])
    assert chunks[0].page_number is None
    assert chunks[0].chunk_id == "d:p0:c0"
