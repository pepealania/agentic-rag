from agentic_rag.schemas import Answer, Citation, RetrievedChunk
from agentic_rag.validation import validate_answer


def make_chunk():
    return RetrievedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        source="source.txt",
        text="This is relevant evidence.",
        chunk_index=0,
        score=0.9,
    )


def test_validator_accepts_matching_citation():
    chunk = make_chunk()

    answer = Answer(
        question_id="q1",
        answer="This is relevant evidence.",
        citations=[
            Citation(
                chunk_id="chunk_1",
                document_id="doc_1",
                source="source.txt",
            )
        ],
    )

    assert validate_answer(answer, [chunk]) == []


def test_validator_rejects_unknown_citation():
    chunk = make_chunk()

    answer = Answer(
        question_id="q1",
        answer="This is relevant evidence.",
        citations=[
            Citation(
                chunk_id="fake",
                document_id="doc_1",
                source="source.txt",
            )
        ],
    )

    errors = validate_answer(answer, [chunk])

    assert any(
        "citation_not_in_evidence" in error
        for error in errors
    )
