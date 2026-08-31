from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document loaded from the raw data directory."""

    document_id: str
    source: str
    text: str


class Chunk(BaseModel):
    """A chunk of a source document used for retrieval."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int = Field(ge=0)


class RetrievedChunk(BaseModel):
    """A chunk returned by the vector store."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int = Field(ge=0)
    score: float


class Question(BaseModel):
    """An evaluation question."""

    question_id: str
    question: str
    answer: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A citation referring to a retrieved source chunk."""

    chunk_id: str
    document_id: str
    source: str


class Answer(BaseModel):
    """A generated RAG answer."""

    question_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
