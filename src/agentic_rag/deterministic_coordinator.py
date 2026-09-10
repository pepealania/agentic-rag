from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.state import AgentState


Route = Literal["retry", "finalize"]


@dataclass(frozen=True)
class DeterministicCoordinator:
    """
    Rule-based coordinator for deterministic Agentic RAG.

    Routing policy:

    1. Valid answer -> finalize.
    2. Invalid answer + retries available -> retry.
    3. Invalid answer + no retries available -> finalize.

    The coordinator does not call an LLM and does not mutate state.
    """

    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

    def validation_passed(self, state: AgentState) -> bool:
        """
        Determine whether validation succeeded.

        Preferred state field:
            validation_passed: bool

        Fallback:
            validation_errors: list

        An empty validation_errors list means validation passed.
        """

        if "validation_passed" in state:
            return bool(state["validation_passed"])

        validation_errors = state.get("validation_errors", [])

        return len(validation_errors) == 0

    def retry_available(self, state: AgentState) -> bool:
        """Return True when the deterministic retry budget remains."""

        retry_count = state.get("retry_count", 0)

        return retry_count < self.max_retries

    def route_after_validation(self, state: AgentState) -> Route:
        """
        Deterministically select the next route after validation.

        This function is intentionally a pure decision function:
        the same state always produces the same route.
        """

        if self.validation_passed(state):
            return "finalize"

        if self.retry_available(state):
            return "retry"

        return "finalize"

    def should_retry(self, state: AgentState) -> bool:
        """Convenience predicate for deterministic retry decisions."""

        return (
            not self.validation_passed(state)
            and self.retry_available(state)
        )

    def decision_reason(self, state: AgentState) -> str:
        """
        Return a deterministic explanation for the routing decision.

        Useful for logging, evaluation, and experiment analysis.
        """

        if self.validation_passed(state):
            return "validation_passed"

        if self.retry_available(state):
            return "validation_failed_and_retry_available"

        return "validation_failed_and_retry_exhausted"
