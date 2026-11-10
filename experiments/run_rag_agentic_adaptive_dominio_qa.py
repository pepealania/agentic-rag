import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from agentic_rag.adaptive_coordinator import AdaptiveCoordinator
from agentic_rag.adaptive_workflow import AdaptiveAgenticRAGWorkflow
from agentic_rag.analyst import AnalystAgent
from agentic_rag.embeddings import EmbeddingModel
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.retrieval import VectorStore
from agentic_rag.schemas import Chunk, Question
from agentic_rag.validation import AnswerValidator
from experiments import domain_qa_helpers as qa


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}:{line_number}"
                )

            records.append(record)

    return records


def load_domain_config(path: Path | None = None) -> dict:
    config_path = path or ROOT / "configs" / "default_desempeno.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return config


def load_chunks(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Domain corpus directory does not exist: {corpus_dir}")

    for jsonl_path in sorted(corpus_dir.rglob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as corpus_file:
            document_chunk_index = 0
            for line_number, line in enumerate(corpus_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object in {jsonl_path}:{line_number}")

                text = record.get("content") or record.get("text") or ""
                if not text:
                    continue

                document_id = str(record.get("document_id", jsonl_path.stem))
                metadata = record.get("source_metadata", {})
                source = str(
                    record.get("source")
                    or (metadata.get("source") if isinstance(metadata, dict) else "")
                    or jsonl_path
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chunk_{document_chunk_index:04d}",
                        document_id=document_id,
                        source=source,
                        text=text,
                        chunk_index=document_chunk_index,
                    )
                )
                document_chunk_index += 1

    if not chunks:
        raise ValueError(f"No corpus chunks were loaded from: {corpus_dir}")
    return chunks


def calculate_citation_validity(citations: list[dict], retrieved_chunks: list) -> float:
    if not citations:
        return 0.0

    retrieved_pairs = {
        (chunk.document_id, chunk.chunk_id)
        for chunk in retrieved_chunks
    }
    valid_count = sum(
        1
        for citation in citations
        if (citation.get("document_id"), citation.get("chunk_id"))
        in retrieved_pairs
    )
    return valid_count / len(citations)


def main() -> None:
    config = load_domain_config()
    corpus_dir = ROOT / config["paths"]["raw_data"]
    questions_path = ROOT / config["evaluation"]["questions_path"]
    output_dir = ROOT / config["paths"]["agentic_adaptive_qa_outputs"]

    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Domain corpus directory does not exist: {corpus_dir}"
        )

    if not questions_path.is_file():
        raise FileNotFoundError(
            f"Domain questions file does not exist: {questions_path}"
        )

    questions = qa.select_qa_questions(qa.load_jsonl(questions_path))
    if not questions:
        raise ValueError(f"No evaluation questions found in {questions_path}")

    for question in questions:
        question["paraphrased_question"] = qa.paraphrase_question(question)

    embedding_config = config.get("embeddings", {})
    embedding_model = EmbeddingModel(
        embedding_config.get("name", "sentence-transformers/all-MiniLM-L6-v2")
    )
    vector_store = VectorStore(embedding_model)
    vector_store.build(load_chunks(corpus_dir))
    evidence_agent = EvidenceAgent(
        vector_store=vector_store,
        top_k=int(config.get("retrieval", {}).get("top_k", 5)),
    )

    model_config = config.get("model", {})
    analyst_agent = AnalystAgent(
        model_name=model_config.get("name", "gpt-4o-mini"),
        base_url=model_config.get("base_url"),
        temperature=float(model_config.get("temperature", 0.0)),
        max_tokens=int(model_config.get("max_tokens", 1024)),
    )
    adaptive_config = config.get("adaptive", {})
    coordinator = AdaptiveCoordinator(
        max_steps=int(adaptive_config.get("max_steps", 12)),
        max_retries=int(adaptive_config.get("max_retries", 2)),
        min_evidence=int(adaptive_config.get("min_evidence", 2)),
        sufficient_evidence=int(
            adaptive_config.get("sufficient_evidence", 5)
        ),
    )
    workflow = AdaptiveAgenticRAGWorkflow(
        evidence_agent=evidence_agent,
        analyst_agent=analyst_agent,
        validator=AnswerValidator(),
        coordinator=coordinator,
        max_steps=coordinator.max_steps,
        max_retries=coordinator.max_retries,
    )
    print(f"Corpus directory: {corpus_dir}")
    print("Adaptive agentic workflow ready (Prolog disabled).")

    print("Selected adaptive QA cases:")
    for question in questions:
        print(
            question["qa_case"],
            "|",
            question["question_id"],
            "|",
            question.get("difficulty", ""),
            "|",
            question.get("question_type", ""),
        )
        print("  Original:", question["question"])
        print("  Paraphrase:", question["paraphrased_question"])

    results = []
    for question in questions:
        for variant, query in (
            ("original", question["question"]),
            ("paraphrase", question["paraphrased_question"]),
        ):
            started_at = time.perf_counter()
            initial_state = {
                "question": Question(
                    question_id=question["question_id"],
                    question=query,
                ),
                "retry_count": 0,
                "step_count": 0,
                "decision_log": [],
                "tool_results": [],
                "validation_errors": [],
            }
            state = workflow.run(initial_state)
            answer = state.get("answer")
            retrieved_chunks = state.get("retrieved_chunks", [])
            citations = [
                citation.model_dump(mode="json")
                for citation in answer.citations
            ] if answer else []
            result = {
                "question_id": question["question_id"],
                "answer": answer.answer if answer else "",
                "facts": answer.facts if answer else [],
                "indicators": answer.indicators if answer else [],
                "uncertainty": answer.uncertainty if answer else [],
                "citations": citations,
                "citation_validity": qa.calculate_citation_validity(
                    citations,
                    retrieved_chunks,
                ),
                "human_review": answer.human_review if answer else True,
                "abstained": answer.abstained if answer else True,
                "validation_passed": state.get("validation_passed", False),
                "validation_errors": state.get("validation_errors", []),
                "query_used": state.get("query_used", ""),
                "retrieved_chunks": [
                    chunk.model_dump(mode="json") for chunk in retrieved_chunks
                ],
                "retry_count": state.get("retry_count", 0),
                "step_count": state.get("step_count", 0),
                "retrieval_attempts": state.get("retrieval_attempts", 0),
                "analysis_attempts": state.get("analysis_attempts", 0),
                "validation_attempts": state.get("validation_attempts", 0),
                "decision_reason": state.get("decision_reason", ""),
                "retry_history": state.get("retry_history", []),
                "decision_log": state.get("decision_log", []),
                "tool_results": state.get("tool_results", []),
                "latency_seconds": round(time.perf_counter() - started_at, 4),
            }
            result = qa.annotate_result(
                result,
                question,
                variant,
                query,
                embedding_model,
            )
            results.append(result)
            print(
                question["question_id"],
                "|",
                variant,
                "|",
                result["citation_validity"],
                "|",
                result["expected_evidence_recall"],
                "|",
                result["reference_answer_similarity"],
                "|",
                result["retry_count"],
            )
            print("  Answer:", result.get("answer", ""))

    qa.save_qa_results(results, output_dir)


if __name__ == "__main__":
    main()