from __future__ import annotations

from typing import Any


class AdaptiveCoordinator:
    """
    Adaptive decision-maker for Agentic RAG.

    The coordinator chooses the next action based on the
    current state.

    Possible actions:

        retrieve
        analyze
        validate
        retry_retrieval
        retry_analysis
        finalize

    Important distinction:

        validation_attempts == 0
            -> answer has NOT been validated yet

        validation_attempts > 0 and validation_passed == False
            -> validation actually FAILED
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

    def decide(
        self,
        state: dict[str, Any],
    ) -> str:

        step_count = state.get(
            "step_count",
            0,
        )

        if step_count >= self.max_steps:
            return self._record(
                state,
                "finalize",
                "maximum_step_budget_reached",
            )

        validation_passed = bool(
            state.get(
                "validation_passed",
                False,
            )
        )

        answer_sufficient = bool(
            state.get(
                "answer_sufficient",
                False,
            )
        )

        evidence_sufficient = (
            self._evidence_is_sufficient(state)
        )

        retrieved = state.get(
            "retrieved_chunks",
            [],
        )

        has_answer = (
            state.get("answer") is not None
        )

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

        retry_count = state.get(
            "retry_count",
            0,
        )

        # ========================================================
        # 1. SUCCESS
        # ========================================================

        if (
            has_answer
            and validation_attempts > 0
            and validation_passed
            and answer_sufficient
        ):
            return self._record(
                state,
                "finalize",
                "validated_answer_is_sufficient",
            )

        # ========================================================
        # 2. NO EVIDENCE
        # ========================================================

        if not retrieved:
            return self._record(
                state,
                "retrieve",
                "no_evidence_available",
            )

        # ========================================================
        # 3. INSUFFICIENT EVIDENCE
        # ========================================================

        if not evidence_sufficient:

            if retrieval_attempts < (
                self.max_retries + 1
            ):
                return self._record(
                    state,
                    "retry_retrieval",
                    "evidence_is_below_sufficiency_threshold",
                )

        # ========================================================
        # 4. EVIDENCE EXISTS BUT NO ANALYSIS
        # ========================================================

        if (
            analysis_attempts == 0
            and not has_answer
        ):
            return self._record(
                state,
                "analyze",
                "evidence_available_and_analysis_required",
            )

        # ========================================================
        # 5. ANSWER EXISTS BUT HAS NEVER BEEN VALIDATED
        # ========================================================

        if (
            has_answer
            and validation_attempts == 0
        ):
            return self._record(
                state,
                "validate",
                "answer_exists_and_requires_validation",
            )

        # ========================================================
        # 6. VALIDATION ACTUALLY FAILED
        # ========================================================

        if (
            has_answer
            and validation_attempts > 0
            and not validation_passed
        ):

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

            # ----------------------------------------------------
            # Validation says evidence is inadequate
            # ----------------------------------------------------

            if self._validation_requires_more_evidence(
                errors
            ):
                return self._record(
                    state,
                    "retry_retrieval",
                    "validation_indicates_insufficient_evidence",
                )

            # ----------------------------------------------------
            # Evidence is available -> re-analysis
            # ----------------------------------------------------

            return self._record(
                state,
                "retry_analysis",
                "validation_failed_existing_evidence_supports_reanalysis",
            )

        # ========================================================
        # 7. ANSWER EXISTS BUT SOMETHING IS INCOMPLETE
        # ========================================================

        if has_answer:
            return self._record(
                state,
                "finalize",
                "no_additional_action_required",
            )

        # ========================================================
        # 8. FALLBACK
        # ========================================================

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

        explicit = state.get(
            "evidence_sufficient"
        )

        if explicit is not None:
            return bool(explicit)

        retrieved = state.get(
            "retrieved_chunks",
            [],
        )

        return (
            len(retrieved)
            >= self.sufficient_evidence
        )

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

        log = list(
            state.get(
                "decision_log",
                [],
            )
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
                "retry_count": state.get(
                    "retry_count",
                    0,
                ),
            }
        )

        state["decision_log"] = log
        state["route"] = decision
        state["decision_reason"] = reason

        return decision
