import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_agentic_deterministic_dominio import (
    DeterministicDomainAgenticRAGWorkflow,
    load_domain_config,
)
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


def calculate_citation_validity(citations: list[dict], retrieved_chunks: list) -> float:
    if not citations:
        return 0.0

    # Extract IDs safely supporting both dictionary objects and class instances
    retrieved_pairs = set()
    for chunk in retrieved_chunks:
        doc_id = getattr(chunk, "document_id", None) or (chunk.get("document_id") if isinstance(chunk, dict) else None)
        chk_id = getattr(chunk, "chunk_id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
        if doc_id:
            retrieved_pairs.add((str(doc_id).strip(), str(chk_id).strip() if chk_id else ""))

    valid_count = 0
    for citation in citations:
        cit_doc = str(citation.get("document_id", "")).strip()
        cit_chk = str(citation.get("chunk_id", "")).strip() if citation.get("chunk_id") else ""
        
        # Fallback check: matching just document_id if chunk tracking is nested
        if (cit_doc, cit_chk) in retrieved_pairs or any(cit_doc == p[0] for p in retrieved_pairs):
            valid_count += 1

    return valid_count / len(citations)


def main() -> None:
    config = load_domain_config()
    corpus_dir = ROOT / config["paths"]["raw_data"]
    questions_path = ROOT / config["evaluation"]["questions_path"]
    output_dir = ROOT / config["paths"]["agentic_deterministic_qa_outputs"]

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

    print("Selected QA cases:")
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

    pipeline_config = config.get("pipeline", {})
    workflow = DeterministicDomainAgenticRAGWorkflow(
        config=config,
        max_steps=int(pipeline_config.get("max_iterations", 8)),
        max_retries=int(pipeline_config.get("max_retries", 1)),
    )
    print(f"Corpus directory: {corpus_dir}")
    print("Deterministic Prolog agentic QA workflow ready.")

    results = []
    for question in questions:
        for variant, query in (
            ("original", question["question"]),
            ("paraphrase", question["paraphrased_question"]),
        ):
            started_at = time.perf_counter()
            state = workflow.run(
                query,
                question_id=question["question_id"],
                employee_id=question.get("employee_id"),
                period_id=question.get("evaluation_period"),
            )
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
                "prolog_findings": state.get("prolog_findings", []),
                "query_used": state.get("query_used", ""),
                "retrieved_chunks": [
                    chunk.model_dump(mode="json") for chunk in retrieved_chunks
                ],
                "retry_count": state.get("retry_count", 0),
                "step_count": state.get("step_count", 0),
                "decision_log": state.get("decision_log", []),
                "tool_results": state.get("tool_results", []),
                "latency_seconds": round(time.perf_counter() - started_at, 4),
            }
            result = qa.annotate_result(
                result,
                question,
                variant,
                query,
                workflow.evidence_agent.vector_store.embedding_model,
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
            )

    qa.save_qa_results(results, output_dir)


if __name__ == "__main__":
    main()