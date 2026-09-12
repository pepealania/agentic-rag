"""
RAG Baseline
============

Simple deterministic Retrieval-Augmented Generation baseline.

Pipeline:

    Documents
        ↓
    Chunking
        ↓
    SentenceTransformer embeddings
        ↓
    FAISS cosine-similarity search
        ↓
    Top-K context
        ↓
    OpenAI-compatible LLM
        ↓
    Structured JSON answer

Designed to run both locally and in Google Colab.

The LLM API key is read from:

    OPENAI_API_KEY

No API key is hard-coded in this module.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class RAGBaseline:
    """
    Minimal RAG baseline for comparison with Agentic RAG.

    Expected configuration structure:

        config["model"]["name"]
        config["model"]["base_url"]
        config["model"]["temperature"]
        config["model"]["max_tokens"]

        config["embeddings"]["name"]

        config["retrieval"]["top_k"]
        config["retrieval"]["chunk_size"]
        config["retrieval"]["chunk_overlap"]
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the baseline RAG system."""

        if not isinstance(config, dict):
            raise TypeError(
                "config must be a dictionary."
            )

        self.config = config

        # ----------------------------------------------------
        # MODEL CONFIGURATION
        # ----------------------------------------------------

        model_config = config.get("model", {})

        self.model_name = model_config.get(
            "name",
            "gpt-4o-mini",
        )

        self.base_url = model_config.get(
            "base_url",
            "https://api.openai.com/v1",
        )

        self.temperature = float(
            model_config.get(
                "temperature",
                0.0,
            )
        )

        self.max_tokens = int(
            model_config.get(
                "max_tokens",
                1024,
            )
        )

        # ----------------------------------------------------
        # RETRIEVAL CONFIGURATION
        # ----------------------------------------------------

        retrieval_config = config.get(
            "retrieval",
            {},
        )

        self.top_k = int(
            retrieval_config.get(
                "top_k",
                5,
            )
        )

        self.chunk_size = int(
            retrieval_config.get(
                "chunk_size",
                500,
            )
        )

        self.chunk_overlap = int(
            retrieval_config.get(
                "chunk_overlap",
                50,
            )
        )

        if self.top_k <= 0:
            raise ValueError(
                "retrieval.top_k must be greater than zero."
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "retrieval.chunk_size must be greater than zero."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "retrieval.chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "retrieval.chunk_overlap must be smaller "
                "than retrieval.chunk_size."
            )

        # ----------------------------------------------------
        # EMBEDDING CONFIGURATION
        # ----------------------------------------------------

        embedding_config = config.get(
            "embeddings",
            {},
        )

        self.embedding_model_name = embedding_config.get(
            "name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        # ----------------------------------------------------
        # OPENAI CLIENT
        # ----------------------------------------------------

        api_key = os.environ.get(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "In Google Colab, run:\n\n"
                "from getpass import getpass\n"
                "import os\n\n"
                "os.environ['OPENAI_API_KEY'] = "
                "getpass('Enter your OPENAI_API_KEY: ')\n"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

        # ----------------------------------------------------
        # EMBEDDING MODEL
        # ----------------------------------------------------

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        # ----------------------------------------------------
        # RAG STATE
        # ----------------------------------------------------

        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.index: faiss.Index | None = None

    # ========================================================
    # DOCUMENT LOADING
    # ========================================================

    def load_documents(
        self,
        path: str | os.PathLike,
    ) -> list[dict[str, Any]]:
        """
        Load all JSONL documents recursively from a directory.

        Each JSONL line must contain at least:

            {
                "document_id": "...",
                "content": "..."
            }
        """

        root_path = Path(path)

        if not root_path.exists():
            raise FileNotFoundError(
                f"Document path does not exist: {root_path}"
            )

        if not root_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {root_path}"
            )

        documents: list[dict[str, Any]] = []

        jsonl_files = sorted(
            root_path.rglob("*.jsonl")
        )

        if not jsonl_files:
            raise FileNotFoundError(
                f"No .jsonl files found under: {root_path}"
            )

        for filepath in jsonl_files:

            with filepath.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line_number, line in enumerate(
                    f,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in "
                            f"{filepath}:{line_number}"
                        ) from exc

                    if not isinstance(
                        document,
                        dict,
                    ):
                        raise ValueError(
                            f"Expected JSON object in "
                            f"{filepath}:{line_number}"
                        )

                    document_id = document.get(
                        "document_id"
                    )

                    content = document.get(
                        "content"
                    )

                    if not document_id:
                        raise ValueError(
                            f"Missing document_id in "
                            f"{filepath}:{line_number}"
                        )

                    if content is None:
                        raise ValueError(
                            f"Missing content in "
                            f"{filepath}:{line_number}"
                        )

                    documents.append(
                        document
                    )

        self.documents = documents

        return self.documents

    # ========================================================
    # CHUNKING
    # ========================================================

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        Split text into overlapping word-based chunks.
        """

        if not isinstance(text, str):
            text = str(text)

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        words = text.split()

        if not words:
            return []

        chunks: list[str] = []

        start = 0

        step = chunk_size - overlap

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words),
            )

            chunk = " ".join(
                words[start:end]
            )

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

            start += step

        return chunks

    def build_chunks(
        self,
    ) -> list[dict[str, Any]]:
        """
        Build chunks from loaded documents.
        """

        if not self.documents:
            raise RuntimeError(
                "No documents loaded. "
                "Call load_documents() first."
            )

        self.chunks = []

        for document in self.documents:

            document_id = document.get(
                "document_id"
            )

            content = document.get(
                "content",
                "",
            )

            text_chunks = self.chunk_text(
                content,
                self.chunk_size,
                self.chunk_overlap,
            )

            for chunk_index, text in enumerate(
                text_chunks
            ):

                self.chunks.append(
                    {
                        "document_id": document_id,
                        "chunk_id": (
                            f"{document_id}_chunk_"
                            f"{chunk_index:04d}"
                        ),
                        "chunk_index": chunk_index,
                        "content": text,
                        "metadata": document,
                    }
                )

        if not self.chunks:
            raise RuntimeError(
                "Chunking produced zero chunks."
            )

        return self.chunks

    # ========================================================
    # VECTOR INDEX
    # ========================================================

    def build_index(
        self,
    ) -> faiss.Index:
        """
        Build a FAISS inner-product index.

        Embeddings are normalized, so inner product is
        equivalent to cosine similarity.
        """

        if not self.chunks:
            raise RuntimeError(
                "No chunks available. "
                "Call build_chunks() first."
            )

        texts = [
            chunk["content"]
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # Handle the one-chunk case safely.
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(
                1,
                -1,
            )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Embedding model returned an invalid "
                f"shape: {embeddings.shape}"
            )

        if embeddings.shape[0] != len(
            self.chunks
        ):
            raise RuntimeError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        dimension = embeddings.shape[1]

        if dimension <= 0:
            raise RuntimeError(
                "Embedding dimension is invalid."
            )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        return self.index

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-K most relevant chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built. "
                "Call build_index() first."
            )

        if not self.chunks:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        k = min(
            self.top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results: list[dict[str, Any]] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            chunk = self.chunks[
                int(index)
            ].copy()

            chunk["score"] = float(
                score
            )

            results.append(
                chunk
            )

        return results

    # ========================================================
    # PROMPT
    # ========================================================

    def build_prompt(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
    ) -> str:
        """
        Build the grounded generation prompt.
        """

        if not retrieved:
            evidence_text = (
                "NO EVIDENCE WAS RETRIEVED."
            )

        else:
            evidence_blocks = []

            for item in retrieved:

                evidence_blocks.append(
                    "\n".join(
                        [
                            (
                                f"DOCUMENT_ID: "
                                f"{item['document_id']}"
                            ),
                            (
                                f"CHUNK_ID: "
                                f"{item['chunk_id']}"
                            ),
                            (
                                f"SCORE: "
                                f"{item['score']:.6f}"
                            ),
                            "",
                            "CONTENT:",
                            item["content"],
                        ]
                    )
                )

            evidence_text = "\n\n".join(
                evidence_blocks
            )

        return f"""
Eres un sistema de análisis documental.

Responde la pregunta utilizando EXCLUSIVAMENTE
la evidencia proporcionada.

REGLAS:

1. No utilices conocimiento externo.
2. No inventes hechos.
3. Cada afirmación factual debe estar respaldada
   por una o más citas.
4. Utiliza únicamente los DOCUMENT_ID y CHUNK_ID
   presentes en la evidencia.
5. Si la evidencia no permite responder,
   debes indicarlo claramente.
6. Responde en español.
7. Devuelve exclusivamente JSON válido.
8. No incluyas markdown.
9. No incluyas texto antes o después del JSON.

Pregunta:
{question}

Evidencia:
{evidence_text}

Formato obligatorio:

{{
  "answer": "respuesta en español",
  "citations": [
    {{
      "document_id": "DOC-XXX",
      "chunk_id": "DOC-XXX_chunk_0000"
    }}
  ],
  "abstained": false
}}
""".strip()

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """
        Generate an answer using the configured
        OpenAI-compatible model.

        Returns:
            (raw_response, generation_latency)
        """

        prompt = self.build_prompt(
            question,
            retrieved,
        )

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded document "
                        "analysis system. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        latency = (
            time.perf_counter() - start
        )

        if not response.choices:
            raise RuntimeError(
                "The model returned no choices."
            )

        raw = response.choices[
            0
        ].message.content

        if raw is None:
            raise RuntimeError(
                "The model returned empty content."
            )

        return raw, latency

    # ========================================================
    # RESPONSE PARSING
    # ========================================================

    def parse_response(
        self,
        raw: str,
    ) -> dict[str, Any]:
        """
        Parse model JSON output.

        Handles ordinary JSON and common markdown
        code-fence wrapping.
        """

        if not isinstance(raw, str):
            raise TypeError(
                "Model response must be a string."
            )

        text = raw.strip()

        if not text:
            raise ValueError(
                "Model returned an empty response."
            )

        # Remove markdown code fences.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        # First attempt: direct JSON.
        try:
            parsed = json.loads(
                text
            )
        except json.JSONDecodeError:

            # Fallback: locate the outermost JSON object.
            start = text.find("{")
            end = text.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    "Model response does not contain "
                    "a valid JSON object."
                )

            parsed = json.loads(
                text[start : end + 1]
            )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Parsed model response must be "
                "a JSON object."
            )

        return parsed

    # ========================================================
    # CITATION VALIDATION
    # ========================================================

    def validate_citations(
        self,
        response: dict[str, Any],
        retrieved: list[dict[str, Any]],
    ) -> float:
        """
        Calculate the proportion of citations that
        refer to actually retrieved chunks.

        Returns:
            float in [0, 1]
        """

        valid_pairs = {
            (
                item["document_id"],
                item["chunk_id"],
            )
            for item in retrieved
        }

        citations = response.get(
            "citations",
            [],
        )

        if not isinstance(
            citations,
            list,
        ):
            return 0.0

        if not citations:
            return 0.0

        valid = 0

        for citation in citations:

            if not isinstance(
                citation,
                dict,
            ):
                continue

            pair = (
                citation.get(
                    "document_id"
                ),
                citation.get(
                    "chunk_id"
                ),
            )

            if pair in valid_pairs:
                valid += 1

        return valid / len(
            citations
        )

    # ========================================================
    # FULL PIPELINE
    # ========================================================

    def run(
        self,
        question_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Execute the complete baseline RAG pipeline.

        Returns a structured experiment result.
        """

        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()

        try:
            retrieved = self.retrieve(
                question
            )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

        except Exception as exc:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return {
                "question_id": question_id,
                "answer": "",
                "abstained": True,
                "json_valid": False,
                "citation_validity": 0.0,
                "latency_seconds": total_latency,
                "retrieval_latency_seconds": 0.0,
                "generation_latency_seconds": 0.0,
                "retrieved_documents": [],
                "error": (
                    f"Retrieval error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        try:

            raw_response, generation_latency = (
                self.generate(
                    question,
                    retrieved,
                )
            )

        except Exception as exc:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return {
                "question_id": question_id,
                "answer": "",
                "abstained": True,
                "json_valid": False,
                "citation_validity": 0.0,
                "latency_seconds": total_latency,
                "retrieval_latency_seconds": (
                    retrieval_latency
                ),
                "generation_latency_seconds": 0.0,
                "retrieved_documents": [
                    {
                        "document_id": x[
                            "document_id"
                        ],
                        "chunk_id": x[
                            "chunk_id"
                        ],
                        "score": x[
                            "score"
                        ],
                    }
                    for x in retrieved
                ],
                "error": (
                    f"Generation error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

        # ----------------------------------------------------
        # PARSING + VALIDATION
        # ----------------------------------------------------

        try:

            parsed = self.parse_response(
                raw_response
            )

            json_valid = True

            citation_validity = (
                self.validate_citations(
                    parsed,
                    retrieved,
                )
            )

            answer = parsed.get(
                "answer",
                "",
            )

            abstained = parsed.get(
                "abstained",
                False,
            )

            if not isinstance(
                answer,
                str,
            ):
                answer = str(
                    answer
                )

            error = None

        except Exception as exc:

            parsed = {}

            json_valid = False
            citation_validity = 0.0
            answer = raw_response
            abstained = None

            error = (
                f"Parse/validation error: "
                f"{type(exc).__name__}: {exc}"
            )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        total_latency = (
            time.perf_counter()
            - total_start
        )

        return {
            "question_id": question_id,
            "answer": answer,
            "abstained": abstained,
            "json_valid": json_valid,
            "citation_validity": citation_validity,
            "latency_seconds": total_latency,
            "retrieval_latency_seconds": (
                retrieval_latency
            ),
            "generation_latency_seconds": (
                generation_latency
            ),
            "retrieved_documents": [
                {
                    "document_id": x[
                        "document_id"
                    ],
                    "chunk_id": x[
                        "chunk_id"
                    ],
                    "score": x[
                        "score"
                    ],
                }
                for x in retrieved
            ],
            "error": error,
        }
