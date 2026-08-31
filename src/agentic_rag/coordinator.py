from __future__ import annotations

from agentic_rag.state import AgentState


class Coordinator:
    """Coordinate the bounded Agentic RAG workflow."""

    def __init__(self, max_steps: int = 8) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        self.max_steps = max_steps

    def _log_decision(
        self,
        state: AgentState,
        decision: str,
        reason: str,
    ) -> None:
        decision_log = state.setdefault("decision_log", [])

        decision_log.append(
            {
                "agent": "coordinator",
                "decision": decision,
                "reason": reason,
                "step": state.get("step_count", 0),
            }
        )

    def route_after_validation(self, state: AgentState) -> str:
        """Choose retry or finalize after validation."""

        errors = state.get("validation_errors", [])
        retry_count = state.get("retry_count", 0)
        step_count = state.get("step_count", 0)

        if step_count >= self.max_steps:
            self._log_decision(
                state,
                "finalize",
                "global_step_limit_reached",
            )
            return "finalize"

        if not errors:
            self._log_decision(
                state,
                "finalize",
                "validation_passed",
            )
            return "finalize"

        if retry_count < 1:
            state["retry_count"] = retry_count + 1

            self._log_decision(
                state,
                "retry",
                "validation_failed_and_retry_available",
            )
            return "retry"

        self._log_decision(
            state,
            "finalize",
            "validation_failed_and_retry_exhausted",
        )
        return "finalize"

    def route_initial(self, state: AgentState) -> str:
        """Choose the first workflow operation."""

        if state.get("step_count", 0) >= self.max_steps:
            self._log_decision(
                state,
                "finalize",
                "global_step_limit_reached",
            )
            return "finalize"

        self._log_decision(
            state,
            "retrieve",
            "initial_evidence_required",
        )
        return "retrieve"
