from __future__ import annotations

from typing import Any, TypedDict

from agentic_rag.state import AgentState


class AdaptiveState(TypedDict, total=False):
    """
    Runtime state for the adaptive Agentic RAG workflow.

    Unlike the deterministic workflow, the next action is selected
    dynamically from the current state.
    """

    # ------------------------------------------------------------
    # Shared question / retrieval state
    # ------------------------------------------------------------

    question: Any
    query_used: str
    retrieved_chunks: list[Any]

    # ------------------------------------------------------------
    # Answer state
    # ------------------------------------------------------------

    answer: Any

    # ------------------------------------------------------------
    # Adaptive control
    # ------------------------------------------------------------

    step_count: int
    retry_count: int

    max_steps: int
    max_retries: int

    route: str

    # ------------------------------------------------------------
    # Adaptive decision information
    # ------------------------------------------------------------

    retrieval_attempts: int
    analysis_attempts: int
    validation_attempts: int

    evidence_sufficient: bool
    answer_sufficient: bool
    validation_passed: bool

    decision_reason: str

    # ------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------

    decision_log: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    validation_errors: list[str]


def initialize_adaptive_state(
    state: AgentState,
    *,
    max_steps: int = 12,
    max_retries: int = 2,
) -> AgentState:
    """
    Initialize adaptive execution state.

    Existing values are preserved.
    """

    initialized = dict(state)

    initialized.setdefault("retry_count", 0)
    initialized.setdefault("step_count", 0)

    initialized.setdefault("max_steps", max_steps)
    initialized.setdefault("max_retries", max_retries)

    initialized.setdefault("retrieval_attempts", 0)
    initialized.setdefault("analysis_attempts", 0)
    initialized.setdefault("validation_attempts", 0)

    initialized.setdefault("evidence_sufficient", False)
    initialized.setdefault("answer_sufficient", False)
    initialized.setdefault("validation_passed", False)

    initialized.setdefault("decision_reason", "")

    initialized.setdefault("decision_log", [])
    initialized.setdefault("tool_results", [])
    initialized.setdefault("validation_errors", [])

    initialized.setdefault("route", "")

    return initialized


def validate_adaptive_state(state: AgentState) -> None:
    """
    Validate adaptive execution invariants.
    """

    step_count = state.get("step_count", 0)
    retry_count = state.get("retry_count", 0)

    max_steps = state.get("max_steps", 12)
    max_retries = state.get("max_retries", 2)

    if step_count < 0:
        raise ValueError("step_count cannot be negative.")

    if retry_count < 0:
        raise ValueError("retry_count cannot be negative.")

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    if step_count > max_steps:
        raise ValueError(
            "step_count cannot exceed max_steps."
        )

    if retry_count > max_retries:
        raise ValueError(
            "retry_count cannot exceed max_retries."
        )
