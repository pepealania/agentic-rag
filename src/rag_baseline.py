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
    