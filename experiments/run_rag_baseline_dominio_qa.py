import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_baseline_dominio import RAGBaseline, load_domain_config
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


def select_qa_questions(questions: list[dict]) -> list[dict]:
    case_specs = (
        (
            "easy",
            lambda question: question.get("difficulty") == "baja"
            and question.get("question_type") == "busqueda_directa",
        ),
        (
            "medium",
            lambda question: question.get("difficulty") == "media"
            and question.get("question_type") == "resolucion_versiones",
        ),
        (
            "multihop",
            lambda question: question.get("question_type") == "integracion_multifuente"
            and len(question.get("expected_evidence_ids", [])) >= 3,
        ),
    )

    selected = []
    selected_ids = set()
    for case_name, matches in case_specs:
        question = next(
            (
                candidate
                for candidate in questions
                if candidate.get("question_id") not in selected_ids
                and matches(candidate)
            ),
            None,
        )
        if question is None:
            raise ValueError(
                f"Evaluation set has no question matching QA case: {case_name}"
            )

        selected_ids.add(question["question_id"])
        selected.append({**question, "qa_case": case_name})

    return selected


def paraphrase_question(question: dict) -> str:
    employee_id = question["employee_id"]
    case_name = question["qa_case"]

    if case_name == "easy":
        objective_count = question.get("objective_count", 10)
        return (
            f"De los {objective_count} objetivos atribuidos a {employee_id}, "
            "¿cuántos cuentan con evidencia documental localizada?"
        )
    if case_name == "medium":
        return (
            f"¿Cuál es la versión y calificación vigente de la evaluación de "
            f"{employee_id}, y qué evidencia permite confirmarlo?"
        )
    if case_name == "multihop":
        return (
            f"¿Cómo puede reconstruirse el vínculo entre la calificación de "
            f"{employee_id}, su evaluación y la evidencia correspondiente al periodo?"
        )
    raise ValueError(f"Unsupported QA case for paraphrasing: {case_name}")


def calculate_reference_similarity(
    embedding_model,
    answer: str,
    reference_answer: str,
) -> float | None:
    if not answer.strip() or not reference_answer.strip():
        return None

    vectors = embedding_model.encode(
        [answer, reference_answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(vectors[0], vectors[1]))


def annotate_result(
    result: dict,
    question: dict,
    variant: str,
    query: str,
    embedding_model,
) -> dict:
    expected_evidence_ids = question.get("expected_evidence_ids", [])
    retrieved_document_ids = {
        document.get("document_id")
        for document in result.get("retrieved_documents", [])
    }
    retrieved_expected_ids = [
        document_id
        for document_id in expected_evidence_ids
        if document_id in retrieved_document_ids
    ]
    missing_expected_ids = [
        document_id
        for document_id in expected_evidence_ids
        if document_id not in retrieved_document_ids
    ]

    result.update(
        {
            "question": query,
            "original_question": question["question"],
            "paraphrased_question": question["paraphrased_question"],
            "variant": variant,
            "reference_answer": question.get("reference_answer", ""),
            "reference_answer_similarity": calculate_reference_similarity(
                embedding_model,
                result.get("answer", ""),
                question.get("reference_answer", ""),
            ),
            "expected_evidence_ids": expected_evidence_ids,
            "retrieved_expected_evidence_ids": retrieved_expected_ids,
            "missing_expected_evidence_ids": missing_expected_ids,
            "expected_evidence_recall": (
                len(retrieved_expected_ids) / len(expected_evidence_ids)
                if expected_evidence_ids
                else None
            ),
            "multihop_evidence_complete": (
                not missing_expected_ids
                if question["qa_case"] == "multihop"
                else None
            ),
            "expected_evidence_types": question.get(
                "expected_evidence_types", []
            ),
            "scenario": question.get("scenario", ""),
            "question_type": question.get("question_type", ""),
            "difficulty": question.get("difficulty", ""),
            "qa_case": question["qa_case"],
            "pair_id": question["question_id"],
        }
    )
    return result


def compare_variants(results: list[dict]) -> list[dict]:
    comparisons = []
    metric_names = (
        "citation_validity",
        "expected_evidence_recall",
        "reference_answer_similarity",
    )

    for question_id in dict.fromkeys(row["question_id"] for row in results):
        pair = {
            row["variant"]: row
            for row in results
            if row["question_id"] == question_id
        }
        original = pair["original"]
        paraphrase = pair["paraphrase"]
        deltas = {
            metric: (
                paraphrase[metric] - original[metric]
                if paraphrase[metric] is not None
                and original[metric] is not None
                else None
            )
            for metric in metric_names
        }
        changed = [
            delta
            for delta in deltas.values()
            if delta is not None and abs(delta) >= 0.02
        ]
        if any(delta < 0 for delta in changed) and any(
            delta > 0 for delta in changed
        ):
            assessment = "mixed_change"
        elif any(delta < 0 for delta in changed):
            assessment = "measured_quality_decreased"
        elif any(delta > 0 for delta in changed):
            assessment = "measured_quality_increased"
        else:
            assessment = "no_material_change_in_measured_metrics"

        comparisons.append(
            {
                "question_id": question_id,
                "qa_case": original["qa_case"],
                "original_question": original["original_question"],
                "paraphrased_question": paraphrase["paraphrased_question"],
                "original_answer": original.get("answer", ""),
                "paraphrased_answer": paraphrase.get("answer", ""),
                "deltas_paraphrase_minus_original": deltas,
                "assessment": assessment,
                "assessment_note": (
                    "Citation validity and evidence recall are retrieval/support "
                    "proxies; embedding similarity is not a factuality judgment."
                ),
            }
        )

    return comparisons


def main() -> None:
    config = load_domain_config()
    corpus_dir = ROOT / config["paths"]["raw_data"]
    questions_path = ROOT / config["evaluation"]["questions_path"]
    output_dir = ROOT / config["paths"]["baseline_qa_outputs"]

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

    print("Selected QA cases:")
    for question in questions:
        question["paraphrased_question"] = qa.paraphrase_question(question)
        print(
            question["qa_case"],
            "|",
            question["question_id"],
            "|",
            question.get("difficulty", ""),
            "|",
            question.get("question_type", ""),
        )
        print("  Paraphrase:", question["paraphrased_question"])

    rag = RAGBaseline(config)

    documents = rag.load_documents(corpus_dir)
    print(f"Documents: {len(documents)}")

    chunks = rag.build_chunks()
    print(f"Chunks: {len(chunks)}")

    rag.build_index()
    print("Index ready.")

    results = []
    for question in questions:
        for variant, query in (
            ("original", question["question"]),
            ("paraphrase", question["paraphrased_question"]),
        ):
            raw_result = rag.run(question["question_id"], query)
            result = qa.annotate_result(
                raw_result,
                question,
                variant,
                query,
                rag.embedding_model,
            )
            results.append(result)

            print(
                question["question_id"],
                "|",
                variant,
                "|",
                result["json_valid"],
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