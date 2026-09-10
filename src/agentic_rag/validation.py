from __future__ import annotations

from agentic_rag.schemas import Answer
from agentic_rag.state import AgentState


def validate_answer(
    answer: Answer | None,
    retrieved_chunks,
) -> list[str]:
    """Validate an answer against the retrieved evidence."""

    errors: list[str] = []

    if answer is None:
        errors.append("answer_missing")
        return errors

    # Validate the Pydantic schema explicitly.
    try:
        Answer.model_validate(answer.model_dump())
    except Exception as exc:
        errors.append(f"invalid_answer_schema: {exc}")
        return errors

    if not answer.citations:
        errors.append("citations_missing")

    # chunk_id is only unique within a document in this corpus.
    # Use (document_id, chunk_id) as the deterministic evidence key.
    retrieved_by_id = {
        (chunk.document_id, chunk.chunk_id): chunk
        for chunk in retrieved_chunks
    }

    for citation in answer.citations:
        citation_key = (
            citation.document_id,
            citation.chunk_id,
        )

        if citation_key not in retrieved_by_id:
            errors.append(
                "citation_not_in_evidence: "
                f"{citation.document_id}/{citation.chunk_id}"
            )
            continue

        chunk = retrieved_by_id[citation_key]

        if citation.source != chunk.source:
            errors.append(
                f"citation_source_mismatch: "
                f"{citation.document_id}/{citation.chunk_id}"
            )

    if retrieved_chunks and answer.answer:
        answer_lower = answer.answer.lower()

        evidence_terms = {
            word
            for chunk in retrieved_chunks
            for word in chunk.text.lower().split()
            if len(word.strip(".,!?;:()[]")) >= 5
        }

        if evidence_terms:
            matching_terms = sum(
                1
                for term in evidence_terms
                if term.strip(".,!?;:()[]") in answer_lower
            )

            if matching_terms == 0:
                errors.append(
                    "answer_has_no_basic_evidence_overlap"
                )

    return errors


class AnswerValidator:
    """Deterministic validator for Agentic RAG answers."""

    def run(self, state: AgentState) -> AgentState:
        answer = state.get("answer")
        retrieved_chunks = state.get("retrieved_chunks", [])

        errors = validate_answer(
            answer,
            retrieved_chunks,
        )

        decision_log = list(state.get("decision_log", []))

        if errors:
            decision = "validation_failed"
        else:
            decision = "validation_passed"

        decision_log.append(
            {
                "agent": "validator",
                "decision": decision,
                "error_count": len(errors),
            }
        )

        return {
            "validation_errors": errors,
            "decision_log": decision_log,
        }
