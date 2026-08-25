import json
import os
import time
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI


class RAGBaseline:

    def __init__(self, config):

        self.config = config

        self.model_name = config["model"]["name"]
        self.base_url = config["model"]["base_url"]
        self.temperature = config["model"]["temperature"]
        self.max_tokens = config["model"]["max_tokens"]

        self.top_k = config["retrieval"]["top_k"]

        self.embedding_model_name = config["embeddings"]["name"]

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama"
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        self.documents = []
        self.chunks = []
        self.index = None

    def load_documents(self, path):

        documents = []

        for root, _, files in os.walk(path):

            for filename in files:

                if not filename.endswith(".jsonl"):
                    continue

                filepath = os.path.join(root, filename)

                with open(filepath, "r", encoding="utf-8") as f:

                    for line in f:

                        if line.strip():
                            documents.append(
                                json.loads(line)
                            )

        self.documents = documents

        return documents

    def chunk_text(self, text, chunk_size=500, overlap=50):

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            )

            chunks.append(chunk)

            if end == len(words):
                break

            start = end - overlap

        return chunks
    
    def build_chunks(self):

        chunk_size = self.config["retrieval"]["chunk_size"]
        overlap = self.config["retrieval"]["chunk_overlap"]

        self.chunks = []

        for document in self.documents:

            content = document.get("content", "")

            text_chunks = self.chunk_text(
                content,
                chunk_size,
                overlap
            )

            for i, text in enumerate(text_chunks):

                self.chunks.append({
                    "document_id": document.get(
                        "document_id"
                    ),
                    "chunk_id": f"chunk_{i}",
                    "content": text,
                    "metadata": document
                })

        return self.chunks
    
    def build_index(self):

        texts = [
            chunk["content"]
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        return self.index    

    def retrieve(self, query):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            self.top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            chunk = self.chunks[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results
    
    def build_prompt(self, question, retrieved):

        context = []

        for item in retrieved:

            context.append(
                f"""
DOCUMENT_ID: {item['document_id']}
CHUNK_ID: {item['chunk_id']}
SCORE: {item['score']}

CONTENT:
{item['content']}
"""
            )

        context_text = "\n".join(context)

        return f"""
Eres un sistema de análisis documental.

Responde la pregunta utilizando EXCLUSIVAMENTE
la evidencia proporcionada.

No utilices conocimiento externo.

Si la evidencia no permite responder,
debes indicarlo explícitamente.

Cada afirmación factual debe estar respaldada
por una cita.

Pregunta:
{question}

Evidencia:
{context_text}

Devuelve exclusivamente un JSON válido:

{{
  "answer": "respuesta en español",
  "citations": [
    {{
      "document_id": "DOC-XXX",
      "chunk_id": "chunk_X"
    }}
  ],
  "abstained": false
}}
"""

    def generate(self, question, retrieved):

        prompt = self.build_prompt(
            question,
            retrieved
        )

        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency = time.time() - start

        raw = response.choices[0].message.content

        return raw, latency

    def parse_response(self, raw):

        raw = raw.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        return json.loads(raw)

    def validate_citations(
        self,
        response,
        retrieved
    ):

        valid_pairs = {
            (
                item["document_id"],
                item["chunk_id"]
            )
            for item in retrieved
        }

        citations = response.get(
            "citations",
            []
        )

        if not citations:
            return 0.0

        valid = 0

        for citation in citations:

            pair = (
                citation.get("document_id"),
                citation.get("chunk_id")
            )

            if pair in valid_pairs:
                valid += 1

        return valid / len(citations)

    def run(self, question_id, question):

        start = time.time()

        retrieved = self.retrieve(
            question
        )

        raw_response, generation_latency = (
            self.generate(
                question,
                retrieved
            )
        )

        try:

            parsed = self.parse_response(
                raw_response
            )

            json_valid = True

            citation_validity = (
                self.validate_citations(
                    parsed,
                    retrieved
                )
            )

            answer = parsed.get(
                "answer",
                ""
            )

            abstained = parsed.get(
                "abstained",
                False
            )

            error = None

        except Exception as e:

            parsed = {}

            json_valid = False
            citation_validity = 0.0
            answer = raw_response
            abstained = None
            error = str(e)

        total_latency = time.time() - start

        return {
            "question_id": question_id,
            "answer": answer,
            "abstained": abstained,
            "json_valid": json_valid,
            "citation_validity": citation_validity,
            "latency_seconds": total_latency,
            "generation_latency_seconds": generation_latency,
            "retrieved_documents": [
                {
                    "document_id": x["document_id"],
                    "chunk_id": x["chunk_id"],
                    "score": x["score"]
                }
                for x in retrieved
            ],
            "error": error
        }
