from __future__ import annotations

from typing import Any, Protocol, cast

from langgraph.graph import END, START, StateGraph

from agentic_rag.analyst import AnalystAgent
from agentic_rag.coordinator import Coordinator
from agentic_rag.deterministic_workflow import (
    DeterministicAgenticRAGWorkflow,
)
from agentic_rag.gestor_prolog import PrologKnowledgeManager
from agentic_rag.schemas import RetrievedChunk
from agentic_rag.prolog_state import PrologAgentState
from agentic_rag.validation import AnswerValidator


class PrologManager(Protocol):
    def evaluate(
        self,
        raw_data: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class PrologEvidenceAgent:
    """Adapt Prolog deductions to the deterministic workflow evidence API."""

    def __init__(self, knowledge_manager: PrologManager) -> None:
        self.knowledge_manager = knowledge_manager

    def run(self, state: PrologAgentState) -> PrologAgentState:
        question = state["question"]
        evaluation = self.knowledge_manager.evaluate(
            state.get("raw_data", [])
        )
        results = evaluation["prolog_results"]

        retrieved_chunks: list[RetrievedChunk] = []
        if evaluation["facts_loaded"]:
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id="prolog_inference_1",
                    document_id="prolog_knowledge_base",
                    source="prolog://employee-task-rules",
                    text=self._render_results(results),
                    chunk_index=0,
                    score=1.0,
                )
            )

        tool_results = [
            {
                "tool": "prolog_query",
                "query": "sobrecargado(X)",
                "result_count": len(
                    results["empleados_sobrecargados"]
                ),
            },
            {
                "tool": "prolog_query",
                "query": "requiere_asistencia(X)",
                "result_count": len(results["requieren_asistencia"]),
            },
        ]

        return {
            "query_used": question.question,
            "facts_loaded": evaluation["facts_loaded"],
            "prolog_results": results,
            "retrieved_chunks": retrieved_chunks,
            "tool_results": [
                *state.get("tool_results", []),
                *tool_results,
            ],
        }

    @staticmethod
    def _render_results(results: dict[str, list[str]]) -> str:
        overloaded = results["empleados_sobrecargados"]
        assistance = results["requieren_asistencia"]

        return "\n".join(
            (
                "Resultados deducidos por las reglas Prolog.",
                "Empleados sobrecargados: "
                + (", ".join(overloaded) if overloaded else "ninguno"),
                "Empleados que requieren asistencia: "
                + (", ".join(assistance) if assistance else "ninguno"),
            )
        )


class PrologDeterministicAgenticRAGWorkflow(
    DeterministicAgenticRAGWorkflow
):
    """Deterministic Agentic RAG workflow backed by Prolog deductions."""

    def __init__(
        self,
        knowledge_manager: PrologManager | None = None,
        analyst_agent: AnalystAgent | None = None,
        validator: AnswerValidator | None = None,
        max_steps: int = 8,
        max_retries: int = 1,
    ) -> None:
        self.prolog_evidence_agent = PrologEvidenceAgent(
            knowledge_manager
            if knowledge_manager is not None
            else PrologKnowledgeManager()
        )
        if max_steps < 4:
            raise ValueError(
                "max_steps must be at least 4 for retrieve, analyze, "
                "validate, and finalize."
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.evidence_agent = self.prolog_evidence_agent
        self.analyst_agent = analyst_agent or AnalystAgent()
        self.validator = validator or AnswerValidator()
        self.coordinator = Coordinator(max_steps=max_steps)
        self.max_steps = max_steps
        self.max_retries = max_retries

        graph = StateGraph(PrologAgentState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("analyze", self.analyze)
        graph.add_node("validate", self.validate)
        graph.add_node("prepare_retry", self.prepare_retry)
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
        graph.add_edge("prepare_retry", "retrieve")
        graph.add_edge("finalize", END)

        self.graph = graph.compile()

    def retrieve(self, state: PrologAgentState) -> PrologAgentState:
        return cast(PrologAgentState, super().retrieve(state))

    def run(self, state: PrologAgentState) -> PrologAgentState:
        state = dict(state)
        state.setdefault("retry_count", 0)
        state.setdefault("step_count", 0)
        state.setdefault("max_steps", self.max_steps)
        state.setdefault("validation_errors", [])
        state.setdefault("decision_log", [])
        state.setdefault("tool_results", [])

        return self.graph.invoke(state)