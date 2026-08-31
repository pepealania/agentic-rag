from __future__ import annotations

import numpy as np
import faiss

from agentic_rag.embeddings import EmbeddingModel
from agentic_rag.schemas import Chunk, RetrievedChunk


class VectorStore:
    """FAISS-backed vector store for chunk retrieval."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build a FAISS index from document chunks."""

        self.chunks = list(chunks)

        if not chunks:
            self.index = None
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_model.embed_documents(texts)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index is None or not self.chunks:
            return []

        query_embedding = self.embedding_model.embed_query(query)

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.chunks)),
        )

        results: list[RetrievedChunk] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                )
            )

        return results
