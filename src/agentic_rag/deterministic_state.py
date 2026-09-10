from __future__ import annotations

from typing import Any, TypedDict

from agentic_rag.state import AgentState


class DeterministicState(TypedDict, total=False):
    """
    State fields used by the deterministic Agentic RAG workflow.

    All fields are optional at the type level because LangGraph builds
    and updates state incrementally as the workflow executes.
    """

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    query: str

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    retrieved_documents: list[dict[str, Any]]
    selected_evidence: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    answer: str

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    validation_passed: bool
    validation_errors: list[str]

    # ------------------------------------------------------------------
    # Deterministic execution control
    # ------------------------------------------------------------------

    step_count: int
    retry_count: int
    max_steps: int
    max_retries: int

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    route: str

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    decision_log: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


def initialize_deterministic_state(
    state: AgentState,
    *,
    max_steps: int = 8,
    max_retries: int = 1,
) -> AgentState:
    """
    Initialize state for the deterministic Agentic RAG workflow.

    Existing values are preserved. Missing deterministic execution
    fields receive explicit defaults.
    """

    initialized = dict(state)

    initialized.setdefault("retry_count", 0)
    initialized.setdefault("step_count", 0)
    initialized.setdefault("max_steps", max_steps)
    initialized.setdefault("max_retries", max_retries)

    initialized.setdefault("validation_errors", [])
    initialized.setdefault("decision_log", [])
    initialized.setdefault("tool_results", [])

    initialized.setdefault("route", "")

    return initialized


def validate_deterministic_state(state: AgentState) -> None:
    """
    Validate the execution-control portion of the state.

    Raises:
        ValueError: if deterministic state invariants are violated.
    """

    step_count = state.get("step_count", 0)
    retry_count = state.get("retry_count", 0)
    max_steps = state.get("max_steps", 8)
    max_retries = state.get("max_retries", 1)

    if step_count < 0:
        raise ValueError("step_count cannot be negative.")

    if retry_count < 0:
        raise ValueError("retry_count cannot be negative.")

    if max_steps < 4:
        raise ValueError(
            "max_steps must be at least 4 for the deterministic workflow."
        )

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


def validation_passed(state: AgentState) -> bool:
    """
    Return the deterministic validation status.

    `validation_passed` is preferred when present.

    If it is absent, validation_errors is used as the fallback:
    an empty list means validation passed.
    """

    if "validation_passed" in state:
        return bool(state["validation_passed"])

    return len(state.get("validation_errors", [])) == 0
