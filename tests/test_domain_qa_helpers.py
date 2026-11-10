import pytest

from experiments.domain_qa_helpers import (
    paraphrase_question,
    select_qa_questions,
)


CASE_QUESTION_IDS = {
    "direct_answer": ("QAUD-02", "QAUD-04", "QAUD-11"),
    "analysis": ("QAUD-03", "QAUD-06", "QAUD-12"),
    "multihop": ("QAUD-01", "QAUD-05", "QAUD-07"),
}


def _questions():
    return [
        {
            "question_id": question_id,
            "question": f"Original question {question_id}?",
            "expected_evidence_ids": ["DOC-1", "DOC-2", "DOC-3"],
        }
        for question_ids in CASE_QUESTION_IDS.values()
        for question_id in question_ids
    ]


def test_selects_nine_curated_questions_in_diverse_cases():
    selected = select_qa_questions(_questions())

    assert len(selected) == 9
    assert [question["question_id"] for question in selected] == [
        question_id
        for question_ids in CASE_QUESTION_IDS.values()
        for question_id in question_ids
    ]
    assert [question["qa_case"] for question in selected] == [
        case_name
        for case_name, question_ids in CASE_QUESTION_IDS.items()
        for _ in question_ids
    ]


def test_requires_all_curated_questions():
    questions = [
        question
        for question in _questions()
        if question["question_id"] != "QAUD-07"
    ]

    with pytest.raises(ValueError, match="QAUD-07"):
        select_qa_questions(questions)


def test_paraphrases_are_question_specific_and_distinct():
    selected = select_qa_questions(_questions())
    paraphrases = [
        paraphrase_question(question)
        for question in selected
    ]

    assert len(set(paraphrases)) == len(paraphrases)
    assert all(
        paraphrase != question["question"]
        for question, paraphrase in zip(selected, paraphrases)
    )
    assert "EMP-AUD-002" in paraphrases[0]
    assert "EMP-AUD-004" in paraphrases[1]
    assert "EMP-AUD-011" in paraphrases[2]


def test_paraphrase_rejects_unknown_question():
    with pytest.raises(ValueError, match="No paraphrase is defined"):
        paraphrase_question({"question_id": "QAUD-999"})
