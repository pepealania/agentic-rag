from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, START, StateGraph

from agentic_rag.analyst import AnalystAgent
from agentic_rag.embeddings import EmbeddingModel
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.retrieval import VectorStore
from agentic_rag.schemas import Chunk, Question
from agentic_rag.state import AgentState
from agentic_rag.validation import AnswerValidator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default_desempeno.yaml"


def load_domain_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the performance-audit domain configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )

    return config


class DeterministicDomainAgenticRAGWorkflow:
    """Deterministic LangGraph workflow for the performance-audit domain."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        evidence_agent: EvidenceAgent | None = None,
        analyst_agent: AnalystAgent | None = None,
        validator: AnswerValidator | None = None,
        max_steps: int = 8,
        max_retries: int = 1,
    ) -> None:
        self.config = config or load_domain_config()

        if max_steps < 4:
            raise ValueError(
                "max_steps must be at least 4 for retrieve, analyze, validate, "
                "and finalize."
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.max_steps = max_steps
        self.max_retries = max_retries

        vector_store = self._build_vector_store()
        self.evidence_agent = evidence_agent or EvidenceAgent(
            vector_store=vector_store,
            top_k=int(self.config.get("retrieval", {}).get("top_k", 5)),
        )

        model_cfg = self.config.get("model", {})
        self.analyst_agent = analyst_agent or AnalystAgent(
            model_name=model_cfg.get("name", "gpt-4o-mini"),
            base_url=model_cfg.get("base_url"),
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )
        self.validator = validator or AnswerValidator()
        self.graph = self._build_graph()

    def _build_vector_store(self) -> VectorStore:
        embedding_cfg = self.config.get("embeddings", {})
        embedding_model = EmbeddingModel(
            embedding_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        vector_store = VectorStore(embedding_model)

        corpus_dir = ROOT / self.config["paths"]["raw_data"]
        chunks = self._load_corpus_chunks(corpus_dir)
        vector_store.build(chunks)
        return vector_store

    def _load_corpus_chunks(self, corpus_dir: Path) -> list[Chunk]:
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Domain corpus directory does not exist: {corpus_dir}")

        chunks: list[Chunk] = []
        for jsonl_path in sorted(corpus_dir.rglob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    content = line.strip()
                    if not content:
                        continue

                    try:
                        record = json.loads(content)
                    except json.JSONDecodeError as exc:  # pragma: no cover
                        raise ValueError(
                            f"Invalid JSON in {jsonl_path}:{line_number}"
                        ) from exc

                    if not isinstance(record, dict):
                        continue

                    text = record.get("content") or record.get("text") or ""
                    if not text:
                        continue

                    document_id = str(record.get("document_id", jsonl_path.stem))
                    source = str(record.get("source", str(jsonl_path)))
                    chunk_id = f"{document_id}_chunk_{len(chunks):04d}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            source=source,
                            text=text,
                            chunk_index=len(chunks),
                        )
                    )

        if not chunks:
            raise ValueError(f"No corpus chunks were loaded from: {corpus_dir}")

        return chunks

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

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

        return graph.compile()

    def _increment_step(self, state: AgentState) -> int:
        step_count = state.get("step_count", 0) + 1
        if step_count > self.max_steps:
            raise RuntimeError("Global deterministic Agentic RAG step limit exceeded.")
        return step_count

    def retrieve(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)
        result = self.evidence_agent.run(state)

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "deterministic_evidence_agent",
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

        if result_log[: len(existing_log)] == existing_log:
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
        validation_passed = state.get("validation_passed")
        if validation_passed is None:
            validation_passed = len(state.get("validation_errors", [])) == 0

        retry_count = state.get("retry_count", 0)

        if bool(validation_passed):
            return "finalize"

        if retry_count < self.max_retries:
            return "retry"

        return "finalize"

    def prepare_retry(self, state: AgentState) -> AgentState:
        retry_count = state.get("retry_count", 0)
        if retry_count >= self.max_retries:
            raise RuntimeError("Deterministic retry limit exceeded.")

        next_retry_count = retry_count + 1
        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "deterministic_coordinator",
                "decision": "retry",
                "reason": "validation_failed_and_retry_available",
                "retry_count": next_retry_count,
                "step": state.get("step_count", 0),
            }
        )

        return {
            "retry_count": next_retry_count,
            "decision_log": decision_log,
        }

    def finalize(self, state: AgentState) -> AgentState:
        step_count = self._increment_step(state)
        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "deterministic_coordinator",
                "decision": "finalize",
                "step": step_count,
            }
        )

        return {
            "route": "finalize",
            "step_count": step_count,
            "decision_log": decision_log,
        }

    def run(
        self,
        question: Question | str,
        *,
        question_id: str | None = None,
    ) -> AgentState:
        if isinstance(question, str):
            question_id = question_id or "q_det_001"
            question = Question(
                question_id=question_id,
                question=question,
            )

        state: AgentState = {
            "question": question,
            "retry_count": 0,
            "step_count": 0,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "validation_errors": [],
            "decision_log": [],
            "tool_results": [],
        }

        return self.graph.invoke(state)


__all__ = [
    "DeterministicDomainAgenticRAGWorkflow",
    "load_domain_config",
]
