from __future__ import annotations

from agentic_rag.schemas import Answer
from agentic_rag.state import AgentState


def validate_answer(
    answer: Answer | None,
    retrieved_chunks,
) -> list[str]:
    """Valida de forma estricta una respuesta contrastándola con la evidencia recuperada."""

    # Creamos una lista vacía para ir acumulando los errores que encontremos.
    errors: list[str] = []

    # REGLA 1: Verificar si el Analista entregó o no una respuesta.
    if answer is None:
        errors.append("answer_missing")
        return errors

    # REGLA 2: Validar explícitamente la estructura del esquema Pydantic.
    # Nos aseguramos de que el Analista no haya alterado las llaves o modificado los tipos de datos.
    try:
        Answer.model_validate(answer.model_dump())
    except Exception as exc:
        errors.append(f"invalid_answer_schema: {exc}")
        return errors

    # REGLA 3: Verificar que existan citas.
    # Un RAG serio no puede dar una respuesta sin decir de dónde sacó la información.
    if not answer.citations:
        errors.append("citations_missing")

    # REGLA 4: Verificación cruzada de citas (Crucial para evitar alucinaciones).
    # Como el 'chunk_id' solo es único dentro de su propio documento, creamos una llave
    # determinista combinando (document_id, chunk_id) para mapear toda la evidencia real.
    retrieved_by_id = {
        (chunk.document_id, chunk.chunk_id): chunk
        for chunk in retrieved_chunks
    }

    # Auditamos cada una de las citas que el Analista puso en su respuesta.
    for citation in answer.citations:
        citation_key = (
            citation.document_id,
            citation.chunk_id,
        )

        # Sub-regla A: ¿La cita que inventó el Analista existe en los fragmentos reales buscados?
        if citation_key not in retrieved_by_id:
            errors.append(
                "citation_not_in_evidence: "
                f"{citation.document_id}/{citation.chunk_id}"
            )
            continue

        chunk = retrieved_by_id[citation_key]

        # Sub-regla B: ¿El texto o la URL de la fuente coincide exactamente con el original?
        if citation.source != chunk.source:
            errors.append(
                f"citation_source_mismatch: "
                f"{citation.document_id}/{citation.chunk_id}"
            )

    # REGLA 5: Coincidencia mínima de vocabulario relevante (Overlap).
    # Si tenemos evidencia y una respuesta escrita, verificamos que compartan palabras clave.
    if retrieved_chunks and answer.answer:
        answer_lower = answer.answer.lower()

        # Extraemos palabras de los fragmentos que tengan 5 o más caracteres (ignorando puntuación)
        # para quedarnos únicamente con conceptos e ideas de valor, descartando conectores cortos.
        evidence_terms = {
            word
            for chunk in retrieved_chunks
            for word in chunk.text.lower().split()
            if len(word.strip(".,!?;:()[]")) >= 5
        }

        # Contamos cuántos de estos términos de evidencia aparecen en la respuesta del analista.
        if evidence_terms:
            matching_terms = sum(
                1
                for term in evidence_terms
                if term.strip(".,!?;:()[]") in answer_lower
            )

            # Si no comparten ni una sola palabra de valor, la respuesta se descarta por falta de alineación.
            if matching_terms == 0:
                errors.append(
                    "answer_has_no_basic_evidence_overlap"
                )

    return errors


class AnswerValidator:
    """Validador determinista (por código tradicional) para las respuestas del RAG Agéntico."""

    def run(self, state: AgentState) -> AgentState:
        # Extraemos los datos necesarios de la memoria compartida (State).
        answer = state.get("answer")
        retrieved_chunks = state.get("retrieved_chunks", [])

        # Ejecutamos nuestra batería de 5 reglas lógicas de validación.
        errors = validate_answer(
            answer,
            retrieved_chunks,
        )

        # Si la lista de errores terminó vacía, significa que la validación fue exitosa.
        passed = len(errors) == 0

        # Definimos la decisión técnica de control que se guardará en los logs.
        decision = (
            "validation_passed"
            if passed
            else "validation_failed"
        )

        decision_log = list(
            state.get("decision_log", [])
        )

        # Registramos la auditoría técnica detallando qué componente evaluó y cuántos errores halló.
        decision_log.append(
            {
                "agent": "validator",
                "decision": decision,
                "error_count": len(errors),
            }
        )

        # Devolvemos un nuevo diccionario actualizando las banderas de control del grafo de LangGraph.
        return {
            "validation_errors": errors,
            "validation_passed": passed,
            "answer_sufficient": passed,
            "decision_log": decision_log,
        }
