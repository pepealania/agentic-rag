import csv
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            records.append(record)
    return records


def select_qa_questions(questions: list[dict]) -> list[dict]:
    case_question_ids = (
        ("direct_answer", ("QAUD-02", "QAUD-04", "QAUD-11")),
        ("analysis", ("QAUD-03", "QAUD-06", "QAUD-12")),
        ("multihop", ("QAUD-01", "QAUD-05", "QAUD-07")),
    )
    questions_by_id = {}
    for question in questions:
        question_id = question.get("question_id")
        if question_id in questions_by_id:
            raise ValueError(f"Duplicate evaluation question ID: {question_id}")
        questions_by_id[question_id] = question

    selected = []
    for case_name, question_ids in case_question_ids:
        missing_ids = [
            question_id
            for question_id in question_ids
            if question_id not in questions_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"Evaluation set is missing questions for QA case {case_name}: "
                f"{', '.join(missing_ids)}."
            )

        selected.extend(
            {
                **questions_by_id[question_id],
                "qa_case": case_name,
            }
            for question_id in question_ids
        )
    return selected


def paraphrase_question(question: dict) -> str:
    paraphrases = {
        "QAUD-02": (
            "Al revisar el expediente de EMP-AUD-002, ¿para cuántos objetivos "
            "se encontró respaldo documental?"
        ),
        "QAUD-04": (
            "¿Qué meses del segundo trimestre de 2026 quedan sin comprobar "
            "en el expediente de EMP-AUD-004?"
        ),
        "QAUD-11": (
            "El expediente de EMP-AUD-011 contiene varias versiones: "
            "¿cuál quedó aprobada y qué nota recibió?"
        ),
        "QAUD-03": (
            "¿Coinciden la nota de la evaluación y la del sistema para "
            "EMP-AUD-003? Indica los valores y si fueron conciliados."
        ),
        "QAUD-06": (
            "Al recalcular los indicadores de EMP-AUD-006, ¿qué promedio se "
            "obtiene y por qué difiere del 90% informado?"
        ),
        "QAUD-12": (
            "¿Es válido comparar la evaluación de EMP-AUD-012 con las de otras "
            "áreas, o faltan criterios comunes para hacerlo?"
        ),
        "QAUD-01": (
            "Para comprobar el desempeño de EMP-AUD-001 en 2026-S1, ¿cómo "
            "se corroboran entre sí la evaluación, las entregas y la auditoría?"
        ),
        "QAUD-05": (
            "¿Qué información habría que recuperar para reconstruir quién "
            "evaluó a EMP-AUD-005 y con qué criterios se respaldó la nota?"
        ),
        "QAUD-07": (
            "¿Qué problemas de privacidad y de permisos de acceso encontró "
            "la auditoría en el expediente de EMP-AUD-007?"
        ),
    }
    question_id = question.get("question_id")
    try:
        return paraphrases[question_id]
    except KeyError as error:
        raise ValueError(
            f"No paraphrase is defined for question {question_id!r}."
        ) from error


def _field(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def calculate_citation_validity(citations: list[dict], retrieved_items: list) -> float:
    if not citations:
        return 0.0

    retrieved_pairs = {
        (_field(item, "document_id"), _field(item, "chunk_id"))
        for item in retrieved_items
    }
    valid_count = sum(
        1
        for citation in citations
        if (citation.get("document_id"), citation.get("chunk_id"))
        in retrieved_pairs
    )
    return valid_count / len(citations)


def calculate_reference_similarity(
    embedding_model,
    answer: str,
    reference_answer: str,
) -> float | None:
    if not answer.strip() or not reference_answer.strip():
        return None

    texts = [answer, reference_answer]
    if hasattr(embedding_model, "encode"):
        vectors = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    else:
        vectors = embedding_model.embed_documents(texts)
    return float(np.dot(vectors[0], vectors[1]))


def annotate_result(
    result: dict,
    question: dict,
    variant: str,
    query: str,
    embedding_model,
) -> dict:
    expected_evidence_ids = question.get("expected_evidence_ids", [])
    retrieved_items = result.get(
        "retrieved_documents",
        result.get("retrieved_chunks", []),
    )
    retrieved_document_ids = {
        _field(document, "document_id") for document in retrieved_items
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
                "latency_delta_seconds": (
                    paraphrase.get("latency_seconds", 0)
                    - original.get("latency_seconds", 0)
                ),
                "assessment": assessment,
                "assessment_note": (
                    "Citation validity and evidence recall are retrieval/support "
                    "proxies; embedding similarity is not a factuality judgment."
                ),
            }
        )
    return comparisons


def save_qa_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    csv_path = output_dir / "results.csv"
    fieldnames = list(dict.fromkeys(key for result in results for key in result))
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in result.items()
                }
            )

    comparison_path = output_dir / "paraphrase_impact.json"
    with comparison_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            compare_variants(results),
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Results JSON: {json_path}")
    print(f"Results CSV: {csv_path}")
    print(f"Paraphrase comparison: {comparison_path}")