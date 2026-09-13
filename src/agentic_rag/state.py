from __future__ import annotations

from typing import TypedDict, Any

from agentic_rag.schemas import (
    Answer,
    Question,
    RetrievedChunk,
)


class AgentState(TypedDict, total=False):
    """
    Shared runtime state for all Agentic RAG workflows.

    This state is intentionally shared by:

        1. Baseline RAG
        2. Deterministic RAG
        3. Adaptive Agentic RAG

    The adaptive fields are optional so that the baseline and
    deterministic pipelines remain compatible with this schema.
    """

    # ============================================================
    # CORE RAG STATE
    # ============================================================

    question: Question

    query_used: str

    retrieved_chunks: list[RetrievedChunk]

    answer: Answer | None

    # ============================================================
    # COMMON CONTROL STATE
    # ============================================================

    retry_count: int

    step_count: int

    max_steps: int

    max_retries: int

    route: str

    # ============================================================
    # ADAPTIVE EXECUTION METRICS
    # ============================================================

    retrieval_attempts: int

    analysis_attempts: int

    validation_attempts: int

    # ============================================================
    # ADAPTIVE DECISION STATE
    # ============================================================

    evidence_sufficient: bool

    answer_sufficient: bool

    validation_passed: bool

    decision_reason: str

    # ============================================================
    # VALIDATION
    # ============================================================

    validation_errors: list[str]

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    decision_log: list[dict[str, Any]]

    tool_results: list[dict[str, Any]]

    retry_history: list[dict[str, Any]]