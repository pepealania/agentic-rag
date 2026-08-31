from __future__ import annotations

from .schemas import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a document into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks: list[Chunk] = []

    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0

    while start < len(document.text):
        end = min(start + chunk_size, len(document.text))

        text = document.text[start:end].strip()

        if text:
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.document_id}_chunk_{chunk_index:04d}"
                    ),
                    document_id=document.document_id,
                    source=document.source,
                    text=text,
                    chunk_index=chunk_index,
                )
            )

        start += step
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split multiple documents into chunks."""

    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks
