from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.analyst import AnalystAgent
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.state import AgentState
from agentic_rag.validation import AnswerValidator

from agentic_rag.adaptive_coordinator import (
    AdaptiveCoordinator,
)


class AdaptiveAgenticRAGWorkflow:
    """
    Adaptive Agentic RAG workflow.

    Unlike the deterministic workflow, this workflow does not use
    a fixed retrieve -> analyze -> validate sequence.

    The AdaptiveCoordinator selects the next action based on the
    current state.
    """

    def __init__(
        self,
        evidence_agent: EvidenceAgent,
        analyst_agent: AnalystAgent | None = None,
        validator: AnswerValidator | None = None,
        coordinator: AdaptiveCoordinator | None = None,
        max_steps: int = 12,
        max_retries: int = 2,
    ) -> None:

        if max_steps < 1:
            raise ValueError(
                "max_steps must be at least 1."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        self.evidence_agent = evidence_agent

        self.analyst_agent = (
            analyst_agent
            or AnalystAgent()
        )

        self.validator = (
            validator
            or AnswerValidator()
        )

        self.coordinator = (
            coordinator
            or AdaptiveCoordinator(
                max_steps=max_steps,
                max_retries=max_retries,
            )
        )

        self.max_steps = max_steps
        self.max_retries = max_retries

        # --------------------------------------------------------
        # Build graph
        # --------------------------------------------------------

        graph = StateGraph(AgentState)

        graph.add_node(
            "retrieve",
            self.retrieve,
        )

        graph.add_node(
            "analyze",
            self.analyze,
        )

        graph.add_node(
            "validate",
            self.validate,
        )

        graph.add_node(
            "retry_retrieval",
            self.retry_retrieval,
        )

        graph.add_node(
            "retry_analysis",
            self.retry_analysis,
        )

        graph.add_node(
            "finalize",
            self.finalize,
        )

        graph.add_edge(
            START,
            "adaptive_router",
        ) if False else None

        # LangGraph requires an actual node for the initial route.
        graph.add_node(
            "adaptive_router",
            self.adaptive_router,
        )

        graph.add_edge(
            START,
            "adaptive_router",
        )

        graph.add_conditional_edges(
            "adaptive_router",
            self.route,
            {
                "retrieve": "retrieve",
                "analyze": "analyze",
                "validate": "validate",
                "retry_retrieval": "retry_retrieval",
                "retry_analysis": "retry_analysis",
                "finalize": "finalize",
            },
        )

        # Every action returns control to the adaptive coordinator.
        graph.add_edge(
            "retrieve",
            "adaptive_router",
        )

        graph.add_edge(
            "analyze",
            "adaptive_router",
        )

        graph.add_edge(
            "validate",
            "adaptive_router",
        )

        graph.add_edge(
            "retry_retrieval",
            "adaptive_router",
        )

        graph.add_edge(
            "retry_analysis",
            "adaptive_router",
        )

        graph.add_edge(
            "finalize",
            END,
        )

        self.graph = graph.compile()

    # ============================================================
    # ROUTER
    # ============================================================

    def adaptive_router(
        self,
        state: AgentState,
    ) -> AgentState:

        return {
            "route": self.coordinator.decide(
                state
            )
        }

    def route(
        self,
        state: AgentState,
    ) -> str:

        route = state.get(
            "route",
            "finalize",
        )

        valid_routes = {
            "retrieve",
            "analyze",
            "validate",
            "retry_retrieval",
            "retry_analysis",
            "finalize",
        }

        if route not in valid_routes:
            return "finalize"

        return route

    # ============================================================
    # STEP MANAGEMENT
    # ============================================================

    def _increment_step(
        self,
        state: AgentState,
    ) -> int:

        step = (
            state.get("step_count", 0)
            + 1
        )

        if step > self.max_steps:
            raise RuntimeError(
                "Adaptive Agentic RAG step budget exceeded."
            )

        return step

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        state: AgentState,
    ) -> AgentState:

        step = self._increment_step(
            state
        )

        result = self.evidence_agent.run(
            state
        )

        attempts = (
            state.get(
                "retrieval_attempts",
                0,
            )
            + 1
        )

        retrieved = result.get(
            "retrieved_chunks",
            [],
        )

        # Explicit evidence sufficiency signal.
        evidence_sufficient = (
            len(retrieved)
            >= self.coordinator.sufficient_evidence
        )

        return {
            **result,
            "step_count": step,
            "retrieval_attempts": attempts,
            "evidence_sufficient": evidence_sufficient,
        }

    # ============================================================
    # RETRIEVAL RETRY
    # ============================================================

    def retry_retrieval(
        self,
        state: AgentState,
    ) -> AgentState:

        retry_count = (
            state.get(
                "retry_count",
                0,
            )
            + 1
        )

        if retry_count > self.max_retries:
            return {
                "retry_count": retry_count,
                "validation_errors": [
                    "Retrieval retry budget exhausted."
                ],
            }

        return {
            "retry_count": retry_count,
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
        state: AgentState,
    ) -> AgentState:

        step = self._increment_step(
            state
        )

        result = self.analyst_agent.run(
            state
        )

        attempts = (
            state.get(
                "analysis_attempts",
                0,
            )
            + 1
        )

        return {
            **result,
            "step_count": step,
            "analysis_attempts": attempts,
        }

    # ============================================================
    # ANALYSIS RETRY
    # ============================================================

    def retry_analysis(
        self,
        state: AgentState,
    ) -> AgentState:

        retry_count = (
            state.get(
                "retry_count",
                0,
            )
            + 1
        )

        if retry_count > self.max_retries:
            return {
                "retry_count": retry_count,
            }

        return {
            "retry_count": retry_count,
            "validation_passed": False,
            "answer_sufficient": False,
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(
        self,
        state: AgentState,
    ) -> AgentState:

        step = self._increment_step(
            state
        )

        result = self.validator.run(
            state
        )

        attempts = (
            state.get(
                "validation_attempts",
                0,
            )
            + 1
        )

        validation_errors = result.get(
            "validation_errors",
            [],
        )

        passed = result.get(
            "validation_passed",
            len(validation_errors) == 0,
        )

        # An answer that passes validation is considered sufficient.
        answer_sufficient = bool(
            passed
        )

        return {
            **result,
            "step_count": step,
            "validation_attempts": attempts,
            "validation_passed": passed,
            "answer_sufficient": answer_sufficient,
        }

    # ============================================================
    # FINALIZATION
    # ============================================================

    def finalize(
        self,
        state: AgentState,
    ) -> AgentState:

        step = self._increment_step(
            state
        )

        decision_log = list(
            state.get(
                "decision_log",
                [],
            )
        )

        decision_log.append(
            {
                "agent": "adaptive_coordinator",
                "decision": "finalize",
                "reason": state.get(
                    "decision_reason",
                    "workflow_complete",
                ),
                "step": step,
            }
        )

        return {
            "route": "finalize",
            "step_count": step,
            "decision_log": decision_log,
        }

    # ============================================================
    # RUN
    # ============================================================

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        state = dict(state)

        state.setdefault(
            "retry_count",
            0,
        )

        state.setdefault(
            "step_count",
            0,
        )

        state.setdefault(
            "max_steps",
            self.max_steps,
        )

        state.setdefault(
            "max_retries",
            self.max_retries,
        )

        state.setdefault(
            "retrieval_attempts",
            0,
        )

        state.setdefault(
            "analysis_attempts",
            0,
        )

        state.setdefault(
            "validation_attempts",
            0,
        )

        state.setdefault(
            "evidence_sufficient",
            False,
        )

        state.setdefault(
            "answer_sufficient",
            False,
        )

        state.setdefault(
            "validation_passed",
            False,
        )

        state.setdefault(
            "decision_log",
            [],
        )

        state.setdefault(
            "tool_results",
            [],
        )

        state.setdefault(
            "validation_errors",
            [],
        )

        return self.graph.invoke(
            state
        )
