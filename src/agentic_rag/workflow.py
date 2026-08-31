from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.analyst import AnalystAgent
from agentic_rag.coordinator import Coordinator
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.state import AgentState
from agentic_rag.validation import AnswerValidator


class AgenticRAGWorkflow:
    """Minimal bounded Agentic RAG workflow."""

    def __init__(
        self,
        evidence_agent: EvidenceAgent,
        analyst_agent: AnalystAgent | None = None,
        validator: AnswerValidator | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 4:
            raise ValueError(
                "max_steps must be at least 4 for retrieve, analyze, "
                "validate, and finalize."
            )

        self.evidence_agent = evidence_agent
        self.analyst_agent = analyst_agent or AnalystAgent()
        self.validator = validator or AnswerValidator()
        self.coordinator = Coordinator(max_steps=max_steps)
        self.max_steps = max_steps

        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self.retrieve)
        graph.add_node("analyze", self.analyze)
        graph.add_node("validate", self.validate)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "analyze")
        graph.add_edge("analyze", "validate")

        graph.add_conditional_edges(
            "validate",
            self.route_after_validation,
            {
                "retry": "prepare_retry",
                "finalize": "finalize",
            },
        )

        graph.add_node("prepare_retry", self.prepare_retry)
        graph.add_edge("prepare_retry", "retrieve")

        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _increment_step(self, state: AgentState) -> int:
        step_count = state.get("step_count", 0) + 1

        if step_count > self.max_steps:
            raise RuntimeError(
                "Global Agentic RAG step limit exceeded."
            )

        return step_count

    def retrieve(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)

        result = self.evidence_agent.run(state)

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "coordinator",
                "decision": "retrieve",
                "step": step_count,
            }
        )

        return {
            **result,
            "step_count": step_count,
            "decision_log": decision_log,
        }

    def analyze(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)

        result = self.analyst_agent.run(state)

        existing_log = list(state.get("decision_log", []))
        result_log = list(result.get("decision_log", []))

        # Agents may return their own log entries. Preserve the workflow
        # history so routing and retry decisions remain auditable.
        if result_log[:len(existing_log)] == existing_log:
            decision_log = result_log
        else:
            decision_log = existing_log + result_log

        return {
            **result,
            "step_count": step_count,
            "decision_log": decision_log,
        }

    def validate(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)

        result = self.validator.run(state)

        return {
            **result,
            "step_count": step_count,
        }

    def route_after_validation(self, state: AgentState) -> str:
        route = self.coordinator.route_after_validation(
            dict(state)
        )

        # LangGraph conditional routing functions should not be relied on
        # to persist mutations. Return the route only; retry_count is updated
        # explicitly in the retry node.
        return route

    def prepare_retry(self, state: AgentState) -> AgentState:
        """Persist the bounded retry decision before retrieval."""

        retry_count = state.get("retry_count", 0)

        if retry_count >= 1:
            raise RuntimeError(
                "Retry limit exceeded."
            )

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "coordinator",
                "decision": "retry",
                "reason": "validation_failed_and_retry_available",
                "step": state.get("step_count", 0),
            }
        )

        return {
            "retry_count": retry_count + 1,
            "decision_log": decision_log,
        }

    def finalize(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "coordinator",
                "decision": "finalize",
                "step": step_count,
            }
        )

        return {
            "route": "finalize",
            "step_count": step_count,
            "decision_log": decision_log,
        }

    def run(self, state: AgentState) -> AgentState:
        """Execute the bounded workflow."""

        state = dict(state)

        state.setdefault("retry_count", 0)
        state.setdefault("step_count", 0)
        state.setdefault("max_steps", self.max_steps)
        state.setdefault("validation_errors", [])
        state.setdefault("decision_log", [])
        state.setdefault("tool_results", [])

        return self.graph.invoke(state)


