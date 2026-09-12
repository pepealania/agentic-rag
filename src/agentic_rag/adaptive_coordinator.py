from __future__ import annotations

from typing import Any


class AdaptiveCoordinator:
    """
    Adaptive decision-maker for Agentic RAG.

    The coordinator chooses the next action from the current state.

    Possible actions:

        retrieve
        analyze
        validate
        retry_retrieval
        retry_analysis
        finalize

    The policy is bounded, but the path through the workflow is
    determined at runtime.
    """

    def __init__(
        self,
        *,
        max_steps: int = 12,
        max_retries: int = 2,
        min_evidence: int = 2,
        sufficient_evidence: int = 5,
    ) -> None:

        self.max_steps = max_steps
        self.max_retries = max_retries

        self.min_evidence = min_evidence
        self.sufficient_evidence = sufficient_evidence

    # ============================================================
    # PUBLIC DECISION FUNCTION
    # ============================================================

    def decide(self, state: dict[str, Any]) -> str:
        """
        Dynamically select the next action.
        """

        step_count = state.get("step_count", 0)

        if step_count >= self.max_steps:
            return self._record(
                state,
                "finalize",
                "maximum_step_budget_reached",
            )

        validation_passed = bool(
            state.get("validation_passed", False)
        )

        answer_sufficient = bool(
            state.get("answer_sufficient", False)
        )

        evidence_sufficient = self._evidence_is_sufficient(state)

        retrieved = state.get("retrieved_chunks", [])
        has_answer = state.get("answer") is not None

        retrieval_attempts = state.get(
            "retrieval_attempts",
            0,
        )

        analysis_attempts = state.get(
            "analysis_attempts",
            0,
        )

        retry_count = state.get(
            "retry_count",
            0,
        )

        # --------------------------------------------------------
        # 1. Successful answer -> finalize
        # --------------------------------------------------------

        if validation_passed and answer_sufficient:
            return self._record(
                state,
                "finalize",
                "validated_answer_is_sufficient",
            )

        # --------------------------------------------------------
        # 2. Existing answer failed validation
        # --------------------------------------------------------

        if has_answer and not validation_passed:

            if retry_count >= self.max_retries:
                return self._record(
                    state,
                    "finalize",
                    "validation_failed_retry_budget_exhausted",
                )

            errors = state.get(
                "validation_errors",
                [],
            )

            # If validation indicates an evidence problem,
            # retrieve again.
            if self._validation_requires_more_evidence(errors):
                return self._record(
                    state,
                    "retry_retrieval",
                    "validation_indicates_insufficient_evidence",
                )

            # Otherwise allow another analysis pass.
            return self._record(
                state,
                "retry_analysis",
                "validation_failed_but_existing_evidence_may_support_reanalysis",
            )

        # --------------------------------------------------------
        # 3. No evidence -> retrieve
        # --------------------------------------------------------

        if not retrieved:
            return self._record(
                state,
                "retrieve",
                "no_evidence_available",
            )

        # --------------------------------------------------------
        # 4. Evidence is weak -> retrieve again
        # --------------------------------------------------------

        if not evidence_sufficient:

            if retrieval_attempts < self.max_retries + 1:
                return self._record(
                    state,
                    "retry_retrieval",
                    "evidence_is_below_sufficiency_threshold",
                )

        # --------------------------------------------------------
        # 5. Evidence exists but no analysis yet
        # --------------------------------------------------------

        if analysis_attempts == 0:
            return self._record(
                state,
                "analyze",
                "evidence_available_and_analysis_required",
            )

        # --------------------------------------------------------
        # 6. Answer exists but hasn't been validated
        # --------------------------------------------------------

        if has_answer and not validation_passed:
            return self._record(
                state,
                "validate",
                "answer_exists_and_requires_validation",
            )

        # --------------------------------------------------------
        # 7. Fallback
        # --------------------------------------------------------

        if has_answer:
            return self._record(
                state,
                "finalize",
                "no_additional_action_required",
            )

        return self._record(
            state,
            "analyze",
            "fallback_analysis_required",
        )

    # ============================================================
    # EVIDENCE ASSESSMENT
    # ============================================================

    def _evidence_is_sufficient(
        self,
        state: dict[str, Any],
    ) -> bool:

        explicit = state.get("evidence_sufficient")

        if explicit is not None:
            return bool(explicit)

        retrieved = state.get(
            "retrieved_chunks",
            [],
        )

        count = len(retrieved)

        return count >= self.sufficient_evidence

    # ============================================================
    # VALIDATION INTERPRETATION
    # ============================================================

    def _validation_requires_more_evidence(
        self,
        errors: list[str],
    ) -> bool:

        if not errors:
            return False

        evidence_terms = (
            "citation",
            "evidence",
            "source",
            "context",
            "support",
            "ground",
            "document",
        )

        text = " ".join(
            str(error).lower()
            for error in errors
        )

        return any(
            term in text
            for term in evidence_terms
        )

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    def _record(
        self,
        state: dict[str, Any],
        decision: str,
        reason: str,
    ) -> str:

        log = state.setdefault(
            "decision_log",
            [],
        )

        log.append(
            {
                "agent": "adaptive_coordinator",
                "decision": decision,
                "reason": reason,
                "step": state.get(
                    "step_count",
                    0,
                ),
                "retrieval_attempts": state.get(
                    "retrieval_attempts",
                    0,
                ),
                "analysis_attempts": state.get(
                    "analysis_attempts",
                    0,
                ),
                "validation_attempts": state.get(
                    "validation_attempts",
                    0,
                ),
            }
        )

        state["route"] = decision
        state["decision_reason"] = reason

        return decision
