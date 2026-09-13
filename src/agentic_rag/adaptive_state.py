from __future__ import annotations

from typing import Any

from agentic_rag.state import AgentState


def initialize_adaptive_state(
    state: AgentState,
    *,
    max_steps: int = 12,
    max_retries: int = 2,
) -> AgentState:
    """
    Initialize the runtime state required by the adaptive
    Agentic RAG workflow.

    Existing values are preserved.

    The adaptive workflow uses the shared AgentState schema so
    that baseline RAG, deterministic RAG, and adaptive RAG can
    coexist in the same project.
    """

    initialized = dict(state)

    # ============================================================
    # COMMON CONTROL STATE
    # ============================================================

    initialized.setdefault(
        "retry_count",
        0,
    )

    initialized.setdefault(
        "step_count",
        0,
    )

    initialized.setdefault(
        "max_steps",
        max_steps,
    )

    initialized.setdefault(
        "max_retries",
        max_retries,
    )

    initialized.setdefault(
        "route",
        "",
    )

    # ============================================================
    # EXECUTION ATTEMPTS
    # ============================================================

    initialized.setdefault(
        "retrieval_attempts",
        0,
    )

    initialized.setdefault(
        "analysis_attempts",
        0,
    )

    initialized.setdefault(
        "validation_attempts",
        0,
    )

    # ============================================================
    # ADAPTIVE DECISION STATE
    # ============================================================

    initialized.setdefault(
        "evidence_sufficient",
        False,
    )

    initialized.setdefault(
        "answer_sufficient",
        False,
    )

    initialized.setdefault(
        "validation_passed",
        False,
    )

    initialized.setdefault(
        "decision_reason",
        "",
    )

    # ============================================================
    # VALIDATION
    # ============================================================

    initialized.setdefault(
        "validation_errors",
        [],
    )

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    initialized.setdefault(
        "decision_log",
        [],
    )

    initialized.setdefault(
        "tool_results",
        [],
    )

    initialized.setdefault(
        "retry_history",
        [],
    )

    return initialized


def validate_adaptive_state(
    state: AgentState,
) -> None:
    """
    Validate adaptive execution invariants.
    """

    step_count = state.get(
        "step_count",
        0,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_steps = state.get(
        "max_steps",
        12,
    )

    max_retries = state.get(
        "max_retries",
        2,
    )

    # ============================================================
    # BASIC VALIDATION
    # ============================================================

    if step_count < 0:
        raise ValueError(
            "step_count cannot be negative."
        )

    if retry_count < 0:
        raise ValueError(
            "retry_count cannot be negative."
        )

    if max_steps < 1:
        raise ValueError(
            "max_steps must be at least 1."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries cannot be negative."
        )

    # ============================================================
    # EXECUTION LIMITS
    # ============================================================

    if step_count > max_steps:
        raise ValueError(
            "step_count cannot exceed max_steps."
        )

    if retry_count > max_retries:
        raise ValueError(
            "retry_count cannot exceed max_retries."
        )

    # ============================================================
    # ATTEMPT COUNTERS
    # ============================================================

    retrieval_attempts = state.get(
        "retrieval_attempts",
        0,
    )

    analysis_attempts = state.get(
        "analysis_attempts",
        0,
    )

    validation_attempts = state.get(
        "validation_attempts",
        0,
    )

    if retrieval_attempts < 0:
        raise ValueError(
            "retrieval_attempts cannot be negative."
        )

    if analysis_attempts < 0:
        raise ValueError(
            "analysis_attempts cannot be negative."
        )

    if validation_attempts < 0:
        raise ValueError(
            "validation_attempts cannot be negative."
        )