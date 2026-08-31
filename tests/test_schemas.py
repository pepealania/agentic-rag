from agentic_rag.schemas import Answer, Citation


def test_answer_has_required_agentic_fields():
    answer = Answer(
        question_id="q1",
        answer="Respuesta.",
        facts=["Hecho 1"],
        indicators=["Indicador 1"],
        uncertainty=["Incertidumbre 1"],
        citations=[
            Citation(
                chunk_id="chunk_1",
                document_id="doc_1",
                source="source.txt",
            )
        ],
        human_review=False,
    )

    assert answer.facts == ["Hecho 1"]
    assert answer.indicators == ["Indicador 1"]
    assert answer.uncertainty == ["Incertidumbre 1"]
    assert len(answer.citations) == 1
    assert answer.human_review is False
