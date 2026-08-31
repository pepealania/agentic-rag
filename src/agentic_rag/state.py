from __future__ import annotations

from typing import TypedDict

from agentic_rag.schemas import Answer, Question, RetrievedChunk


class AgentState(TypedDict, total=False):
    """Shared state for the minimal Agentic RAG workflow."""

    question: Question

    query_used: str
    retrieved_chunks: list[RetrievedChunk]

    answer: Answer | None

    retry_count: int
    step_count: int
    max_steps: int

    route: str

    validation_errors: list[str]

    decision_log: list[dict]
    tool_results: list[dict]
