import pytest

from experiments.domain_qa_helpers import select_qa_questions


def _question(question_id, difficulty, question_type, evidence_count=2):
    return {
        "question_id": question_id,
        "difficulty": difficulty,
        "question_type": question_type,
        "expected_evidence_ids": [
            f"DOC-{question_id}-{index}"
            for index in range(evidence_count)
        ],
    }


def test_selects_three_questions_for_each_qa_case():
    questions = [
        *[
            _question(f"easy-{index}", "baja", "busqueda_directa")
            for index in range(4)
        ],
        *[
            _question(f"medium-{index}", "media", "resolucion_versiones")
            for index in range(4)
        ],
        *[
            _question(
                f"multihop-{index}",
                "alta",
                "integracion_multifuente",
                evidence_count=3,
            )
            for index in range(4)
        ],
    ]

    selected = select_qa_questions(questions)

    assert len(selected) == 9
    assert len({question["question_id"] for question in selected}) == 9
    assert {
        case_name: sum(question["qa_case"] == case_name for question in selected)
        for case_name in ("easy", "medium", "multihop")
    } == {"easy": 3, "medium": 3, "multihop": 3}


def test_requires_three_questions_for_each_qa_case():
    questions = [
        _question("easy-1", "baja", "busqueda_directa"),
        _question("easy-2", "baja", "busqueda_directa"),
    ]

    with pytest.raises(ValueError, match="expected 3"):
        select_qa_questions(questions)
