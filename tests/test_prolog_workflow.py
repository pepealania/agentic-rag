import shutil

import pytest

from agentic_rag.prolog_workflow import (
    PrologDeterministicAgenticRAGWorkflow,
)
from agentic_rag.schemas import Answer, Citation, Question


class FakeKnowledgeManager:
    def __init__(self):
        self.received_data = None

    def evaluate(self, raw_data):
        self.received_data = raw_data
        return {
            "facts_loaded": bool(raw_data),
            "fact_count": len(raw_data),
            "prolog_results": {
                "empleados_sobrecargados": ["Ana"],
                "requieren_asistencia": ["Ana"],
            },
        }


class FakeAnalyst:
    def run(self, state):
        chunk = state["retrieved_chunks"][0]
        return {
            "answer": Answer(
                question_id=state["question"].question_id,
                answer=chunk.text,
                facts=[chunk.text],
                citations=[
                    Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                    )
                ],
            ),
            "decision_log": state.get("decision_log", []),
        }


def test_prolog_workflow_uses_deductions_as_validated_evidence():
    manager = FakeKnowledgeManager()
    workflow = PrologDeterministicAgenticRAGWorkflow(
        knowledge_manager=manager,
        analyst_agent=FakeAnalyst(),
    )
    raw_data = [
        {"tipo": "empleado", "nombre": "Ana", "rol": "analista"}
    ]

    result = workflow.run(
        {
            "question": Question(
                question_id="q1",
                question="¿Quién está sobrecargada?",
            ),
            "raw_data": raw_data,
        }
    )

    assert manager.received_data == raw_data
    assert result["prolog_results"]["empleados_sobrecargados"] == ["Ana"]
    assert result["validation_passed"] is True
    assert result["validation_errors"] == []
    assert result["retry_count"] == 0
    assert result["route"] == "finalize"


@pytest.mark.skipif(
    shutil.which("swipl") is None,
    reason="SWI-Prolog is not installed in this environment.",
)
def test_prolog_manager_applies_overload_and_assistance_rules():
    pytest.importorskip("pyswip")
    from agentic_rag.gestor_prolog import PrologKnowledgeManager

    manager = PrologKnowledgeManager()
    records = [
        {"tipo": "empleado", "nombre": "Ana", "rol": "analista"},
        *[
            {
                "tipo": "tarea",
                "id": f"task_{index}",
                "empleado": "Ana",
                "estado": "Pendiente",
            }
            for index in range(4)
        ],
        {
            "tipo": "tarea",
            "id": "task_late",
            "empleado": "Ana",
            "estado": "retrasada",
        },
    ]

    result = manager.evaluate(records)

    assert result["facts_loaded"] is True
    assert result["prolog_results"] == {
        "empleados_sobrecargados": ["Ana"],
        "requieren_asistencia": ["Ana"],
    }