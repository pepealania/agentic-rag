import runpy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generate_corpus_dominio_creates_100_questions():
    script_path = ROOT / "experiments" / "generate_corpus_dominio.py"
    namespace = runpy.run_path(str(script_path))

    assert len(namespace["QUESTIONS"]) == 100
    assert [question["question_id"] for question in namespace["QUESTIONS"]] == [
        f"QAUD-{number:02d}" for number in range(1, 101)
    ]


def test_generate_corpus_dominio_classifies_all_three_difficulty_levels():
    script_path = ROOT / "experiments" / "generate_corpus_dominio.py"
    namespace = runpy.run_path(str(script_path))
    questions = namespace["QUESTIONS"]

    assert {question["difficulty"] for question in questions} == {
        "baja",
        "media",
        "alta",
    }
    assert all(
        question["difficulty"] == "alta"
        for question in questions
        if question["question_type"] == "integracion_multifuente"
    )
    assert namespace["VALIDATION_REPORT"]["question_type_by_difficulty"][
        "integracion_multifuente"
    ] == {
        "alta": namespace["VALIDATION_REPORT"]["question_type_distribution"][
            "integracion_multifuente"
        ]
    }


def test_generate_corpus_dominio_tracks_objective_counts_for_incomplete_evidence():
    script_path = ROOT / "experiments" / "generate_corpus_dominio.py"
    namespace = runpy.run_path(str(script_path))

    incomplete_questions = [
        question
        for question in namespace["QUESTIONS"]
        if question["scenario"] == "evidencia_incompleta"
    ]

    assert incomplete_questions
    question = incomplete_questions[0]

    assert question["objective_count"] == question["supported_count"] + question["unsupported_count"]
    assert question["supported_count"] < question["objective_count"]
    assert question["unsupported_count"] > 0


def test_generate_corpus_dominio_balances_all_scenarios_in_bulk_records():
    script_path = ROOT / "experiments" / "generate_corpus_dominio.py"
    namespace = runpy.run_path(str(script_path))

    bulk_documents = [
        document
        for document in namespace["DOCUMENTS"]
        if int(document["employee_id"].rsplit("-", 1)[1]) >= 13
    ]
    bulk_scenarios = [
        document["source_metadata"]["audit_scenario"]
        for document in bulk_documents
        if document["document_type"] == "evaluacion"
    ]
    scenario_counts = {
        scenario: sum(
            generated_scenario == scenario
            for generated_scenario in bulk_scenarios
        )
        for scenario in namespace["AUDIT_SCENARIOS"]
    }

    assert len(namespace["AUDIT_SCENARIOS"]) == 12
    assert all(count > 0 for count in scenario_counts.values())
    assert max(scenario_counts.values()) - min(scenario_counts.values()) <= 1
    assert bulk_scenarios[:12] != bulk_scenarios[12:24]

    documents_by_id = {
        document["document_id"]: document
        for document in namespace["DOCUMENTS"]
    }
    for question in namespace["QUESTIONS"][12:]:
        question_documents = [
            documents_by_id[document_id]
            for document_id in question["expected_evidence_ids"]
        ]
        assert all(
            document["source_metadata"]["audit_scenario"] == question["scenario"]
            for document in question_documents
        )


def test_generate_corpus_dominio_reports_validation_and_question_distribution():
    script_path = ROOT / "experiments" / "generate_corpus_dominio.py"
    namespace = runpy.run_path(str(script_path))
    report = namespace["VALIDATION_REPORT"]

    assert report["total_questions"] == 100
    assert report["verified_questions"] == 100
    assert report["rejected_questions"] == 0
    assert report["questions_without_evidence"] == 0
    assert report["inconsistent_answers"] == 0
    assert report["incorrectly_associated_evidence"] == 0
    assert sum(report["question_type_distribution"].values()) == 100
    assert sum(report["difficulty_distribution"].values()) == 100

    saved_report = json.loads(namespace["validation_report_path"].read_text(encoding="utf-8"))
    assert saved_report == report
