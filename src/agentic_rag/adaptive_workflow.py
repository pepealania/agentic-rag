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

    The AdaptiveCoordinator dynamically selects the next action
    according to the current state.

    Possible actions:

        retrieve
        analyze
        validate
        retry_retrieval
        retry_analysis
        finalize

    Important:

    retry_retrieval and retry_analysis are REAL retry transitions.

        retry_retrieval -> retrieve
        retry_analysis  -> analyze

    Therefore a retry actually executes the corresponding agent
    again rather than merely incrementing a retry counter.
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

        # ========================================================
        # BUILD LANGGRAPH
        # ========================================================

        graph = StateGraph(AgentState)

        # --------------------------------------------------------
        # Nodes
        # --------------------------------------------------------

        graph.add_node(
            "adaptive_router",
            self.adaptive_router,
        )

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

        # --------------------------------------------------------
        # START -> adaptive router
        # --------------------------------------------------------

        graph.add_edge(
            START,
            "adaptive_router",
        )

        # --------------------------------------------------------
        # Adaptive routing
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Normal agent execution returns to coordinator
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # REAL RETRY TRANSITIONS
        #
        # A retry node records the retry and then executes the
        # corresponding agent again.
        # --------------------------------------------------------

        graph.add_edge(
            "retry_retrieval",
            "retrieve",
        )

        graph.add_edge(
            "retry_analysis",
            "analyze",
        )

        # --------------------------------------------------------
        # Finalization
        # --------------------------------------------------------

        graph.add_edge(
            "finalize",
            END,
        )

        self.graph = graph.compile()

    # ============================================================
    # ADAPTIVE ROUTER
    # ============================================================

    def adaptive_router(
        self,
        state: AgentState,
    ) -> AgentState:

        decision = self.coordinator.decide(
            state
        )

        return {
            "route": decision,
        }

    # ============================================================
    # ROUTING
    # ============================================================

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
            state.get(
                "step_count",
                0,
            )
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

        # --------------------------------------------------------
        # Execute the actual retrieval agent
        # --------------------------------------------------------

        result = self.evidence_agent.run(
            state
        )

        # --------------------------------------------------------
        # Count this retrieval execution
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Determine evidence sufficiency
        # --------------------------------------------------------

        evidence_sufficient = (
            len(retrieved)
            >= self.coordinator.sufficient_evidence
        )

        # --------------------------------------------------------
        # Preserve existing tool results
        # --------------------------------------------------------

        existing_tool_results = list(
            state.get(
                "tool_results",
                [],
            )
        )

        new_tool_results = result.get(
            "tool_results",
            [],
        )

        if new_tool_results:
            existing_tool_results.extend(
                new_tool_results
            )

        return {
            **result,

            "step_count": step,

            "retrieval_attempts": attempts,

            "evidence_sufficient": (
                evidence_sufficient
            ),

            "tool_results": (
                existing_tool_results
            ),
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

        retry_history = list(
            state.get(
                "retry_history",
                [],
            )
        )

        retry_history.append(
            {
                "retry_number": retry_count,
                "retry_type": "retrieval",
                "reason": state.get(
                    "decision_reason",
                    "retrieval_retry_requested",
                ),
                "step": state.get(
                    "step_count",
                    0,
                ),
            }
        )

        return {
            "retry_count": retry_count,
            "retry_history": retry_history,
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

        # --------------------------------------------------------
        # Execute the actual analyst agent
        # --------------------------------------------------------

        result = self.analyst_agent.run(
            state
        )

        # --------------------------------------------------------
        # Count this analysis execution
        # --------------------------------------------------------

        attempts = (
            state.get(
                "analysis_attempts",
                0,
            )
            + 1
        )

        # --------------------------------------------------------
        # Preserve existing tool results
        # --------------------------------------------------------

        existing_tool_results = list(
            state.get(
                "tool_results",
                [],
            )
        )

        new_tool_results = result.get(
            "tool_results",
            [],
        )

        if new_tool_results:
            existing_tool_results.extend(
                new_tool_results
            )

        return {
            **result,

            "step_count": step,

            "analysis_attempts": attempts,

            "tool_results": (
                existing_tool_results
            ),
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

        retry_history = list(
            state.get(
                "retry_history",
                [],
            )
        )

        retry_history.append(
            {
                "retry_number": retry_count,
                "retry_type": "analysis",
                "reason": state.get(
                    "decision_reason",
                    "analysis_retry_requested",
                ),
                "step": state.get(
                    "step_count",
                    0,
                ),
            }
        )

        return {
            "retry_count": retry_count,

            "retry_history": retry_history,

            # The previous validation result no longer represents
            # the new answer that will be generated.
            "validation_passed": False,

            "answer_sufficient": False,

            # Clear previous validation errors because the answer
            # will be regenerated.
            "validation_errors": [],
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

        # --------------------------------------------------------
        # Execute validator
        # --------------------------------------------------------

        result = self.validator.run(
            state
        )

        # --------------------------------------------------------
        # Count validation execution
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # A validated answer is sufficient only if validation
        # actually passed.
        # --------------------------------------------------------

        answer_sufficient = bool(
            passed
        )

        return {
            **result,

            "step_count": step,

            "validation_attempts": attempts,

            "validation_passed": passed,

            "answer_sufficient": (
                answer_sufficient
            ),

            "validation_errors": (
                validation_errors
            ),
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

        # --------------------------------------------------------
        # Make a copy so the caller's dictionary is not modified
        # directly.
        # --------------------------------------------------------

        state = dict(state)

        # --------------------------------------------------------
        # Initialize adaptive state fields
        # --------------------------------------------------------

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
            "validation_errors",
            [],
        )

        state.setdefault(
            "decision_log",
            [],
        )

        state.setdefault(
            "retry_history",
            [],
        )

        state.setdefault(
            "tool_results",
            [],
        )

        state.setdefault(
            "route",
            "",
        )

        state.setdefault(
            "decision_reason",
            "",
        )

        # --------------------------------------------------------
        # Execute graph
        # --------------------------------------------------------

        return self.graph.invoke(
            state
        )
