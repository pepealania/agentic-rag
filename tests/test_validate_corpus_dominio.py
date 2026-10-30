import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "experiments" / "validate_corpus_dominio.py"
SPEC = importlib.util.spec_from_file_location("validate_corpus_dominio", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def build_calculation_records(answer="El promedio simple es 70 %."):
    documents = [
        {
            "document_id": "DOC-EVAL",
            "employee_id": "EMP-AUD-006",
            "document_type": "evaluacion",
            "document_date": "2026-06-30",
            "content": "La evaluación informa un promedio de 90 %.",
            "source_metadata": {"audit_scenario": "error_de_calculo"},
        },
        {
            "document_id": "DOC-SOURCE",
            "employee_id": "EMP-AUD-006",
            "document_type": "evidencia_desempeno",
            "document_date": "2026-06-28",
            "content": "Los indicadores originales son 8 de 10 y 6 de 10.",
            "source_metadata": {"audit_scenario": "error_de_calculo"},
        },
        {
            "document_id": "DOC-AUDIT",
            "employee_id": "EMP-AUD-006",
            "document_type": "registro_auditoria",
            "document_date": "2026-07-02",
            "content": "El promedio recalculado es 70 %.",
            "source_metadata": {"audit_scenario": "error_de_calculo"},
        },
    ]
    question = {
        "question_id": "QAUD-006",
        "employee_id": "EMP-AUD-006",
        "scenario": "error_de_calculo",
        "question": "¿Qué promedio resulta de los indicadores?",
        "reference_answer": answer,
        "expected_evidence_ids": ["DOC-EVAL", "DOC-SOURCE", "DOC-AUDIT"],
        "expected_evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
        "question_type": "verificacion_calculo",
        "difficulty": "media",
    }
    return question, documents


def test_factual_validation_recalculates_from_original_indicators():
    question, documents = build_calculation_records()
    result = validator.validate_question(question, {document["document_id"]: document for document in documents})
    calculation_check = next(
        check for check in result["factual_checks"]
        if check["check"] == "promedio_calculado_desde_indicadores_fuente"
    )

    assert calculation_check["passed"]
    assert calculation_check["detail"]["source_percentages"] == [80.0, 60.0]
    assert calculation_check["detail"]["expected_average"] == 70.0
    assert result["decision"] == "NEEDS_REVIEW"


def test_factual_validation_rejects_answer_with_wrong_calculated_average():
    question, documents = build_calculation_records(answer="El promedio simple es 75 %.")
    result = validator.validate_question(question, {document["document_id"]: document for document in documents})
    calculation_check = next(
        check for check in result["factual_checks"]
        if check["check"] == "promedio_calculado_desde_indicadores_fuente"
    )

    assert not calculation_check["passed"]
    assert result["decision"] == "REJECT"


def test_evidence_from_another_employee_is_rejected():
    question, documents = build_calculation_records()
    documents[1]["employee_id"] = "EMP-AUD-999"
    result = validator.validate_question(question, {document["document_id"]: document for document in documents})

    assert result["decision"] == "REJECT"
    assert not result["evidence_association"]["passed"]
    assert any("otro empleado" in error for error in result["evidence_association"]["errors"])


def test_semantic_judge_acceptance_requires_supported_claims_with_citations():
    question, documents = build_calculation_records()
    by_id = {document["document_id"]: document for document in documents}
    judge_result = {
        "question_is_answerable": True,
        "answer_is_supported": True,
        "evidence_is_relevant": True,
        "contains_unsupported_claims": False,
        "contains_contradictions": False,
        "semantic_score": 1.0,
        "decision": "ACCEPT",
        "reason": "El cálculo coincide con los indicadores originales.",
        "claims": [
            {
                "claim": "El promedio es 70 %.",
                "supported": True,
                "document_ids": ["DOC-SOURCE"],
                "reason": "8/10 y 6/10 promedian 70 %.",
            }
        ],
    }
    result = validator.validate_question(question, by_id, judge=lambda *_args: judge_result)

    assert result["decision"] == "ACCEPT"
    assert result["verified"]
    assert result["unsupported_claims"] == 0


def test_validator_writes_individual_results_and_aggregate_report(tmp_path):
    question, documents = build_calculation_records()
    documents_path = tmp_path / "documents.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    results_path = tmp_path / "semantic_validation.jsonl"
    report_path = tmp_path / "validation_report.json"
    documents_path.write_text("".join(json.dumps(item) + "\n" for item in documents), encoding="utf-8")
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")

    results, report = validator.validate_corpus(documents_path, questions_path, results_path, report_path)

    assert len(results) == 1
    assert report["total_questions"] == 1
    assert report["pending_questions"] == 1
    assert report["verified_questions"] == 0
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_empty_answer_and_missing_evidence_are_rejected_and_counted(tmp_path):
    question, documents = build_calculation_records(answer="")
    question["expected_evidence_ids"] = []
    question["expected_evidence_types"] = []
    results = [validator.validate_question(question, {document["document_id"]: document for document in documents})]
    report = validator.build_report([question], results, judge_enabled=False)

    assert results[0]["decision"] == "REJECT"
    assert report["questions_without_evidence"] == 1
    assert report["inconsistent_answers"] == 1
    assert report["incorrectly_associated_evidence"] == 1


def test_missing_or_irrelevant_evidence_ids_are_rejected():
    question, documents = build_calculation_records()
    question["expected_evidence_ids"] = ["DOC-EVAL", "DOC-SOURCE", "DOC-MISSING"]
    by_id = {document["document_id"]: document for document in documents}
    result = validator.validate_question(question, by_id)

    assert result["decision"] == "REJECT"
    assert not result["evidence_association"]["passed"]
    assert any("No existe el documento DOC-MISSING" in error for error in result["evidence_association"]["errors"])


def test_semantic_judge_rejects_unsupported_or_contradictory_claims():
    question, documents = build_calculation_records()
    by_id = {document["document_id"]: document for document in documents}
    judge_result = {
        "question_is_answerable": False,
        "answer_is_supported": False,
        "evidence_is_relevant": True,
        "contains_unsupported_claims": True,
        "contains_contradictions": True,
        "semantic_score": 0.2,
        "decision": "ACCEPT",
        "reason": "La respuesta contradice los indicadores originales.",
        "claims": [
            {
                "claim": "El promedio es 90 %.",
                "supported": False,
                "document_ids": [],
                "reason": "Los indicadores promedian 70 %.",
            }
        ],
    }
    result = validator.validate_question(question, by_id, judge=lambda *_args: judge_result)

    assert result["decision"] == "REJECT"
    assert result["unsupported_claims"] == 1
    assert result["semantic_judge"]["status"] == "REJECT"


def test_accepted_report_classification_is_reproducible(tmp_path):
    question, documents = build_calculation_records()
    documents_path = tmp_path / "documents.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    documents_path.write_text("".join(json.dumps(item) + "\n" for item in documents), encoding="utf-8")
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
    judge_result = {
        "question_is_answerable": True,
        "answer_is_supported": True,
        "evidence_is_relevant": True,
        "contains_unsupported_claims": False,
        "contains_contradictions": False,
        "semantic_score": 0.95,
        "decision": "ACCEPT",
        "reason": "La respuesta está respaldada.",
        "claims": [
            {
                "claim": "El promedio es 70 %.",
                "supported": True,
                "document_ids": ["DOC-SOURCE"],
                "reason": "Los indicadores originales dan 70 %.",
            }
        ],
    }
    judge = lambda *_args: judge_result
    first_results, first_report = validator.validate_corpus(
        documents_path, questions_path, tmp_path / "first.jsonl", tmp_path / "first.json", judge
    )
    second_results, second_report = validator.validate_corpus(
        documents_path, questions_path, tmp_path / "second.jsonl", tmp_path / "second.json", judge
    )

    assert first_report == second_report
    assert first_results == second_results
    assert first_report["verified_questions"] == 1
    assert first_report["question_type_distribution"] == {"verificacion_calculo": 1}
    assert first_report["difficulty_distribution"] == {"media": 1}