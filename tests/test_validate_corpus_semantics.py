import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "experiments" / "validate_corpus_semantics.py"
SPEC = importlib.util.spec_from_file_location("validate_corpus_semantics", VALIDATOR_PATH)
semantics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(semantics)


def build_records(answer="El promedio simple es 70 %."):
	documents = [
		{
			"document_id": "DOC-EVAL",
			"employee_id": "EMP-AUD-006",
			"document_type": "evaluacion",
			"document_date": "2026-06-30",
			"content": "La evaluación informa un promedio de 90 %.",
			"source_metadata": {"audit_scenario": "error_de_calculo"},
		},
		{
			"document_id": "DOC-SOURCE",
			"employee_id": "EMP-AUD-006",
			"document_type": "evidencia_desempeno",
			"document_date": "2026-06-28",
			"content": "Los indicadores originales son 8 de 10 y 6 de 10.",
			"source_metadata": {"audit_scenario": "error_de_calculo"},
		},
		{
			"document_id": "DOC-AUDIT",
			"employee_id": "EMP-AUD-006",
			"document_type": "registro_auditoria",
			"document_date": "2026-07-02",
			"content": "La auditoría calculó un promedio de 70 %.",
			"source_metadata": {"audit_scenario": "error_de_calculo"},
		},
	]
	question = {
		"question_id": "QAUD-006",
		"employee_id": "EMP-AUD-006",
		"scenario": "error_de_calculo",
		"question": "¿Qué promedio resulta de los indicadores?",
		"reference_answer": answer,
		"expected_evidence_ids": ["DOC-EVAL", "DOC-SOURCE", "DOC-AUDIT"],
		"expected_evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "verificacion_calculo",
		"difficulty": "media",
	}
	return question, documents


def supported_judgment(answer_span="El promedio simple es 70 %."):
	return {
		"question_is_answerable": True,
		"answer_contains_required_facts": True,
		"answer_is_supported": True,
		"evidence_is_relevant": True,
		"contains_unsupported_claims": False,
		"contains_contradictions": False,
		"semantic_score": 1.0,
		"decision": "ACCEPT",
		"reason": "El cálculo se deriva de los indicadores fuente.",
		"required_facts": [
			{
				"fact": "El promedio calculado es 70 %.",
				"present_in_answer": True,
				"supported": True,
				"document_ids": ["DOC-SOURCE"],
				"reason": "8/10 y 6/10 promedian 70 %.",
			}
		],
		"claims": [
			{
				"claim": "El promedio simple es 70 %.",
				"answer_span": answer_span,
				"supported": True,
				"contradicted": False,
				"document_ids": ["DOC-SOURCE"],
				"reason": "Los valores fuente dan 70 %.",
			}
		],
	}


def validate_with_judgment(question, documents, judgment):
	by_id = {document["document_id"]: document for document in documents}
	return semantics.validate_question(question, by_id, judge=lambda *_args: judgment)


def test_complete_supported_claims_and_required_facts_are_accepted():
	question, documents = build_records()
	result = validate_with_judgment(question, documents, supported_judgment())

	assert result["decision"] == "ACCEPT"
	assert result["verified"]
	assert result["claim_coverage"]["complete"]
	assert result["required_facts"][0]["supported"]


def test_missing_required_fact_is_rejected_even_if_judge_says_accept():
	question, documents = build_records()
	judgment = supported_judgment()
	judgment["answer_contains_required_facts"] = False
	judgment["required_facts"][0]["present_in_answer"] = False
	result = validate_with_judgment(question, documents, judgment)

	assert result["decision"] == "REJECT"
	assert not result["claim_coverage"]["required_facts_complete"]


def test_unsupported_conclusion_is_rejected():
	question, documents = build_records()
	question["reference_answer"] = "El promedio simple es 70 %. El empleado merece un ascenso."
	judgment = supported_judgment("El promedio simple es 70 %.")
	judgment["claims"].append({
		"claim": "El empleado merece un ascenso.",
		"answer_span": "El empleado merece un ascenso.",
		"supported": False,
		"contradicted": False,
		"document_ids": [],
		"reason": "Los documentos no evalúan ascensos.",
	})
	judgment["contains_unsupported_claims"] = True
	result = validate_with_judgment(question, documents, judgment)

	assert result["decision"] == "REJECT"
	assert result["unsupported_claims"] == 1


def test_unreviewed_answer_tail_is_rejected_as_incomplete_claim_coverage():
	question, documents = build_records("El promedio simple es 70 %. El reporte es definitivo.")
	result = validate_with_judgment(question, documents, supported_judgment())

	assert result["decision"] == "REJECT"
	assert not result["claim_coverage"]["complete"]


def test_claim_cannot_cite_document_outside_expected_evidence():
	question, documents = build_records()
	judgment = supported_judgment()
	judgment["claims"][0]["document_ids"] = ["DOC-OTHER-EMPLOYEE"]
	result = validate_with_judgment(question, documents, judgment)

	assert result["decision"] == "REJECT"
	assert result["semantic_judge"]["status"] == "INVALID"


def test_offline_mode_never_claims_semantic_verification():
	question, documents = build_records()
	by_id = {document["document_id"]: document for document in documents}
	result = semantics.validate_question(question, by_id)

	assert result["decision"] == "PENDING"
	assert not result["verified"]
	assert result["semantic_judge"]["status"] == "NOT_RUN"