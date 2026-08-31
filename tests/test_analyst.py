from agentic_rag.analyst import AnalystAgent
from agentic_rag.schemas import Question, RetrievedChunk


class FakeMessage:
    content = """
{
  "answer": "La evidencia indica que el indicador fue 42.",
  "facts": ["El indicador fue 42."],
  "indicators": ["42"],
  "uncertainty": [],
  "citations": [
    {
      "document_id": "doc_1",
      "chunk_id": "chunk_1",
      "source": "source.txt"
    }
  ],
  "human_review": false,
  "abstained": false
}
"""


class FakeChoice:
    message = FakeMessage()


class FakeCompletions:
    def create(self, **kwargs):
        return type("Response", (), {"choices": [FakeChoice()]})()


class FakeClient:
    chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_analyst_returns_required_structured_output():
    agent = AnalystAgent(client=FakeClient())

    state = {
        "question": Question(
            question_id="q1",
            question="¿Cuál fue el indicador?",
        ),
        "retrieved_chunks": [
            RetrievedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                source="source.txt",
                text="El indicador fue 42.",
                chunk_index=0,
                score=0.95,
            )
        ],
        "decision_log": [],
    }

    result = agent.run(state)
    answer = result["answer"]

    assert answer.question_id == "q1"
    assert answer.answer
    assert answer.facts
    assert answer.indicators
    assert answer.uncertainty == []
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk_1"
    assert answer.human_review is False
    assert answer.abstained is False


def test_analyst_abstains_without_evidence():
    agent = AnalystAgent(client=FakeClient())

    state = {
        "question": Question(
            question_id="q2",
            question="¿Cuál fue el indicador?",
        ),
        "retrieved_chunks": [],
        "decision_log": [],
    }

    result = agent.run(state)
    answer = result["answer"]

    assert answer.abstained is True
    assert answer.human_review is True
    assert answer.citations == []
    assert answer.uncertainty
