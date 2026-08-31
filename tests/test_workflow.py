import pytest

from agentic_rag.workflow import AgenticRAGWorkflow


class FakeEvidenceAgent:
    def run(self, state):
        return {
            "retrieved_chunks": [],
            "query_used": state["question"].question,
            "tool_results": [],
        }


def test_workflow_requires_safe_step_limit():
    with pytest.raises(ValueError):
        AgenticRAGWorkflow(
            evidence_agent=FakeEvidenceAgent(),
            max_steps=3,
        )
