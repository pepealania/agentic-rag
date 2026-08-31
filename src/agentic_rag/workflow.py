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
                "retry": "retrieve",
                "finalize": "finalize",
            },
        )

        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def _increment_step(self, state: AgentState) -> None:
        step_count = state.get("step_count", 0) + 1
        state["step_count"] = step_count

        if step_count > self.max_steps:
            raise RuntimeError(
                "Global Agentic RAG step limit exceeded."
            )

    def retrieve(self, state: AgentState) -> AgentState:
        self._increment_step(state)

        result = self.evidence_agent.run(state)

        tool_results = list(result.get("tool_results", []))
        decision_log = list(state.get("decision_log", []))

        decision_log.append(
            {
                "agent": "coordinator",
                "decision": "retrieve",
                "step": state["step_count"],
            }
        )

        # return {
        #     **result,
        #     "decision_log": decision_log,
        # }
        
        return {
            **result,
            "step_count": state["step_count"],
            "decision_log": decision_log,
        }        
        
        

    def analyze(self, state: AgentState) -> AgentState:
        self._increment_step(state)

        result = self.analyst_agent.run(state)

        # return {
        #     **result,
        # }
        return {
            **result,
            "step_count": state["step_count"],
        }
        

    def validate(self, state: AgentState) -> AgentState:
        self._increment_step(state)

        result = self.validator.run(state)

        # return {
        #     **result,
        # }
        return {
            **result,
            "step_count": state["step_count"],
        }

    def route_after_validation(self, state: AgentState) -> str:
        return self.coordinator.route_after_validation(state)

    def finalize(self, state: AgentState) -> AgentState:
        self._increment_step(state)

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "coordinator",
                "decision": "finalize",
                "step": state["step_count"],
            }
        )

        # return {
        #     "route": "finalize",
        #     "decision_log": decision_log,
        # }
        return {
            "route": "finalize",
            "step_count": state["step_count"],
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
