from agentic_rag.schemas import Question, RetrievedChunk
from agentic_rag.workflow import AgenticRAGWorkflow


class FakeEvidenceAgent:
    def __init__(self):
        self.calls = 0

    def run(self, state):
        self.calls += 1

        return {
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
            "query_used": state["question"].question,
            "tool_results": [
                {
                    "tool": "vector_search",
                    "query": state["question"].question,
                    "result_count": 1,
                }
            ],
        }


class FakeAnalyst:
    def __init__(self):
        self.calls = 0

    def run(self, state):
        self.calls += 1

        # Deliberately produce an invalid citation on the first attempt.
        citation_chunk = (
            "wrong_chunk"
            if self.calls == 1
            else "chunk_1"
        )

        from agentic_rag.schemas import Answer, Citation

        answer = Answer(
            question_id=state["question"].question_id,
            answer="El indicador fue 42.",
            facts=["El indicador fue 42."],
            indicators=["42"],
            uncertainty=[],
            citations=[
                Citation(
                    chunk_id=citation_chunk,
                    document_id="doc_1",
                    source="source.txt",
                )
            ],
            human_review=False,
            abstained=False,
        )

        return {
            "answer": answer,
            "decision_log": [
                {
                    "agent": "analyst",
                    "decision": "generated_structured_answer",
                    "attempt": self.calls,
                }
            ],
        }


def test_real_workflow_retries_after_validation_failure():
    evidence = FakeEvidenceAgent()
    analyst = FakeAnalyst()

    workflow = AgenticRAGWorkflow(
        evidence_agent=evidence,
        analyst_agent=analyst,
        max_steps=8,
    )

    state = {
        "question": Question(
            question_id="q1",
            question="¿Cuál fue el indicador?",
        ),
    }

    result = workflow.run(state)

    assert evidence.calls == 2
    assert analyst.calls == 2

    assert result["route"] == "finalize"
    assert result["retry_count"] == 1
    assert result["validation_errors"] == []

    decisions = [
        entry["decision"]
        for entry in result["decision_log"]
    ]

    assert "retry" in decisions
    assert decisions[-1] == "finalize"
