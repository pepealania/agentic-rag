import pytest

from agentic_rag.chunking import chunk_document_by_words
from agentic_rag.schemas import Document


def test_chunk_document_by_words_uses_configured_overlap():
    document = Document(
        document_id="doc-1",
        source="corpus.jsonl",
        text=" ".join(f"word-{index}" for index in range(501)),
    )

    chunks = chunk_document_by_words(
        document,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) == 2
    assert len(chunks[0].text.split()) == 500
    assert len(chunks[1].text.split()) == 51
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1_chunk_0000",
        "doc-1_chunk_0001",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunk_document_by_words_rejects_invalid_sizes(
    chunk_size,
    chunk_overlap,
):
    document = Document(
        document_id="doc-1",
        source="corpus.jsonl",
        text="some words",
    )

    with pytest.raises(ValueError):
        chunk_document_by_words(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
