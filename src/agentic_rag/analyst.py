from __future__ import annotations

from agentic_rag.schemas import Answer, Citation
from agentic_rag.state import AgentState


class AnalystAgent:
    """Analyze retrieved evidence and produce a structured RAG answer."""

    def run(self, state: AgentState) -> AgentState:
        question = state["question"]
        retrieved_chunks = state.get("retrieved_chunks", [])

        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
            )
            for chunk in retrieved_chunks
        ]

        if not retrieved_chunks:
            answer_text = (
                "Insufficient evidence was retrieved to answer the question."
            )
        else:
            evidence_text = "\n".join(
                chunk.text for chunk in retrieved_chunks
            )

            answer_text = (
                f"Question: {question.question}\n\n"
                f"Evidence:\n{evidence_text}"
            )

        answer = Answer(
            question_id=question.question_id,
            answer=answer_text,
            citations=citations,
        )

        decision_log = list(state.get("decision_log", []))
        decision_log.append(
            {
                "agent": "analyst",
                "decision": "generated_structured_answer",
                "citation_count": len(citations),
                "evidence_count": len(retrieved_chunks),
            }
        )

        return {
            "answer": answer,
            "decision_log": decision_log,
        }
