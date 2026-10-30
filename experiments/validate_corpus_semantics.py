import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "data" / "raw" / "auditoria_desempeno" / "corpus_auditoria_desempeno.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "evaluation" / "questions_auditoria_desempeno.jsonl"
DEFAULT_RESULTS = ROOT / "data" / "evaluation" / "semantic_validation.jsonl"
DEFAULT_REPORT = ROOT / "data" / "evaluation" / "validation_report.json"

REQUIRED_EVIDENCE_TYPES = {
	"trazabilidad_completa": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"respaldo_completo": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"evidencia_incompleta": {"evidencia_desempeno", "registro_auditoria"},
	"calificaciones_discrepantes": {"evaluacion", "registro_sistema", "registro_auditoria"},
	"evidencia_desactualizada": {"evidencia_desempeno", "registro_auditoria"},
	"metadatos_insuficientes": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"error_de_calculo": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"minimizacion_datos_personales": {"evaluacion", "registro_acceso", "registro_auditoria"},
	"cambio_sin_trazabilidad": {"evaluacion", "historial_versiones", "registro_auditoria"},
	"comparacion_no_sustentada": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"objetivos_parcialmente_respaldados": {"evaluacion", "evidencia_desempeno", "registro_auditoria"},
	"version_vigente_identificada": {"evaluacion_corregida", "registro_auditoria"},
	"comparabilidad_limitada": {"marco_criterios", "registro_auditoria"},
}


def read_jsonl(path):
	with Path(path).open(encoding="utf-8") as source:
		return [json.loads(line) for line in source if line.strip()]


def _find_source_documents(documents, employee_id, document_type):
	return [
		document for document in documents
		if document.get("employee_id") == employee_id
		and document.get("document_type") == document_type
	]


def validate_evidence_association(question, documents_by_id):
	evidence_ids = question.get("expected_evidence_ids") or []
	employee_id = question.get("employee_id")
	scenario = question.get("scenario")
	selected_documents = []
	errors = []
	if not evidence_ids:
		errors.append("La pregunta no incluye evidencias.")
	if len(evidence_ids) != len(set(evidence_ids)):
		errors.append("La lista de evidencias contiene IDs duplicados.")
	for document_id in evidence_ids:
		document = documents_by_id.get(document_id)
		if document is None:
			errors.append(f"No existe el documento {document_id}.")
			continue
		selected_documents.append(document)
		if document.get("employee_id") != employee_id:
			errors.append(f"{document_id} pertenece a otro empleado.")
		if document.get("source_metadata", {}).get("audit_scenario") != scenario:
			errors.append(f"{document_id} pertenece a otro escenario.")

	actual_types = {document.get("document_type") for document in selected_documents}
	expected_types = set(question.get("expected_evidence_types") or [])
	required_types = REQUIRED_EVIDENCE_TYPES.get(scenario, set())
	if actual_types != expected_types:
		errors.append("Los tipos de evidencia no coinciden con los declarados.")
	if not required_types or not required_types <= actual_types:
		errors.append("Falta un tipo de documento necesario para el escenario.")
	return {
		"passed": not errors,
		"errors": errors,
		"expected_evidence_ids": list(evidence_ids),
		"actual_document_types": sorted(str(value) for value in actual_types),
		"required_document_types": sorted(required_types),
		"selected_documents": selected_documents,
	}


def validate_calculation_from_sources(question, employee_documents):
	if question.get("scenario") != "error_de_calculo":
		return {"applicable": False, "passed": True, "details": {}}
	source_documents = _find_source_documents(
		employee_documents,
		question.get("employee_id"),
		"evidencia_desempeno",
	)
	source_ratios = []
	for document in source_documents:
		matches = re.findall(
			r"(\d+(?:\.\d+)?)\s+de\s+(\d+(?:\.\d+)?)",
			document.get("content", ""),
			flags=re.IGNORECASE,
		)
		for numerator, denominator in matches:
			denominator_value = float(denominator)
			if denominator_value:
				source_ratios.append(float(numerator) / denominator_value * 100)
	expected_average = sum(source_ratios) / len(source_ratios) if source_ratios else None
	answer_percentages = [
		float(value)
		for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", question.get("reference_answer", ""))
	]
	passed = expected_average is not None and any(
		abs(value - expected_average) < 0.001 for value in answer_percentages
	)
	return {
		"applicable": True,
		"passed": passed,
		"details": {
			"source_percentages": source_ratios,
			"calculated_average": expected_average,
			"answer_percentages": answer_percentages,
			"source_document_ids": [document["document_id"] for document in source_documents],
		},
	}


def _validate_judgment_shape(judgment, answer_text, allowed_evidence_ids):
	if not isinstance(judgment, dict):
		return False, "El juez no devolvió un objeto JSON."
	boolean_fields = (
		"question_is_answerable",
		"answer_contains_required_facts",
		"answer_is_supported",
		"evidence_is_relevant",
		"contains_unsupported_claims",
		"contains_contradictions",
	)
	if any(not isinstance(judgment.get(field), bool) for field in boolean_fields):
		return False, "Falta un booleano requerido en el juicio semántico."
	if judgment.get("decision") not in {"ACCEPT", "REJECT"}:
		return False, "La decisión debe ser ACCEPT o REJECT."
	if not isinstance(judgment.get("reason"), str) or not judgment["reason"].strip():
		return False, "El juez debe justificar su decisión."
	try:
		score = float(judgment.get("semantic_score"))
	except (TypeError, ValueError):
		return False, "semantic_score debe ser numérico."
	if not 0 <= score <= 1:
		return False, "semantic_score debe estar entre 0 y 1."

	required_facts = judgment.get("required_facts")
	if not isinstance(required_facts, list) or not required_facts:
		return False, "El juicio debe enumerar los hechos mínimos requeridos para responder."
	for fact in required_facts:
		if (
			not isinstance(fact, dict)
			or not isinstance(fact.get("fact"), str)
			or not fact.get("fact", "").strip()
			or not isinstance(fact.get("present_in_answer"), bool)
			or not isinstance(fact.get("supported"), bool)
			or not isinstance(fact.get("reason"), str)
			or not fact.get("reason", "").strip()
		):
			return False, "Un hecho requerido no tiene campos verificables completos."
		if not _valid_citations(fact.get("document_ids"), allowed_evidence_ids):
			return False, "Un hecho requerido cita IDs ausentes de las evidencias esperadas."
		if fact["supported"] and not fact["document_ids"]:
			return False, "Un hecho marcado como respaldado no cita documentos."

	claims = judgment.get("claims")
	if not isinstance(claims, list) or not claims:
		return False, "El juez debe descomponer la respuesta en afirmaciones atómicas."
	cursor = 0
	for claim in claims:
		if (
			not isinstance(claim, dict)
			or not isinstance(claim.get("claim"), str)
			or not claim.get("claim", "").strip()
			or not isinstance(claim.get("answer_span"), str)
			or not claim.get("answer_span", "").strip()
			or not isinstance(claim.get("supported"), bool)
			or not isinstance(claim.get("contradicted"), bool)
			or not isinstance(claim.get("reason"), str)
			or not claim.get("reason", "").strip()
		):
			return False, "Una afirmación no tiene texto, estado o justificación verificable."
		span = claim["answer_span"]
		span_start = answer_text.find(span, cursor)
		if span_start < 0:
			return False, "Las afirmaciones no corresponden a fragmentos literales ordenados de la respuesta."
		unreviewed_text = answer_text[cursor:span_start]
		if any(character.isalnum() for character in unreviewed_text):
			return False, "Quedó texto de la respuesta sin analizar como afirmación."
		cursor = span_start + len(span)
		if not _valid_citations(claim.get("document_ids"), allowed_evidence_ids):
			return False, "Una afirmación cita IDs que no pertenecen a la evidencia esperada."
		if claim["supported"] and not claim["document_ids"]:
			return False, "Una afirmación marcada como respaldada no cita documentos."
	if any(character.isalnum() for character in answer_text[cursor:]):
		return False, "Quedó texto al final de la respuesta sin analizar como afirmación."
	return True, ""


def _valid_citations(document_ids, allowed_evidence_ids):
	return (
		isinstance(document_ids, list)
		and all(isinstance(document_id, str) and document_id in allowed_evidence_ids for document_id in document_ids)
	)


class OpenAISemanticJudge:
	def __init__(self, model=None):
		from openai import OpenAI

		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise RuntimeError("--llm-judge requiere OPENAI_API_KEY.")
		self.model = model or os.environ.get("SEMANTIC_VALIDATION_MODEL", "gpt-4o-mini")
		self.client = OpenAI(
			api_key=api_key,
			base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		)

	def __call__(self, question, employee_documents, expected_documents):
		expected_ids = {document["document_id"] for document in expected_documents}
		prompt = {
			"task": (
				"Evalúa de forma independiente la respuesta, sin confiar en conclusiones de auditoría "
				"cuando contradigan los datos fuente. Enumera todos los hechos mínimos que hacen falta "
				"para responder la pregunta. Descompón exhaustivamente el texto de la respuesta en "
				"afirmaciones atómicas y asocia cada hecho y afirmación con IDs de documentos concretos. "
				"No uses conocimiento externo."
			),
			"question": question.get("question"),
			"reference_answer": question.get("reference_answer"),
			"scenario": question.get("scenario"),
			"expected_evidence_ids": sorted(expected_ids),
			"expected_evidence_types": question.get("expected_evidence_types", []),
			"documents": [
				{
					"document_id": document["document_id"],
					"document_type": document["document_type"],
					"content": document["content"],
				}
				for document in employee_documents
			],
			"required_output": {
				"question_is_answerable": "boolean",
				"answer_contains_required_facts": "boolean",
				"answer_is_supported": "boolean",
				"evidence_is_relevant": "boolean",
				"contains_unsupported_claims": "boolean",
				"contains_contradictions": "boolean",
				"semantic_score": "number 0..1",
				"decision": "ACCEPT or REJECT",
				"reason": "string",
				"required_facts": [{
					"fact": "necessary fact",
					"present_in_answer": "boolean",
					"supported": "boolean",
					"document_ids": ["exact expected source IDs"],
					"reason": "why the fact is necessary and supported or not",
				}],
				"claims": [{
					"claim": "atomic claim",
					"answer_span": "exact contiguous substring from reference_answer",
					"supported": "boolean",
					"contradicted": "boolean",
					"document_ids": ["exact expected source IDs"],
					"reason": "specific support or contradiction in cited documents",
				}],
			},
		}
		response = self.client.chat.completions.create(
			model=self.model,
			temperature=0,
			response_format={"type": "json_object"},
			messages=[
				{"role": "system", "content": "Eres un evaluador independiente de QA. Devuelve únicamente JSON válido y cita IDs exactos."},
				{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
			],
		)
		return json.loads(response.choices[0].message.content)


def validate_question(question, documents_by_id, judge=None):
	employee_id = question.get("employee_id")
	all_employee_documents = [
		document for document in documents_by_id.values()
		if document.get("employee_id") == employee_id
	]
	evidence_result = validate_evidence_association(question, documents_by_id)
	factual_result = validate_calculation_from_sources(question, all_employee_documents)
	precheck_errors = list(evidence_result["errors"])
	if not question.get("reference_answer", "").strip():
		precheck_errors.append("La respuesta de referencia está vacía.")
	if factual_result["applicable"] and not factual_result["passed"]:
		precheck_errors.append("La respuesta no coincide con el cálculo derivado de los indicadores fuente.")

	semantic_judge = {"status": "NOT_RUN", "result": None, "error": None}
	claim_coverage = {"complete": False, "required_facts_complete": False, "unreviewed_answer_text": None}
	claims = []
	required_facts = []
	unsupported_claims = None
	decision = "PENDING"
	reasons = list(precheck_errors)
	judgment = None

	if precheck_errors:
		decision = "REJECT"
	elif judge is None:
		reasons.append("Juez semántico independiente no ejecutado; requiere revisión.")
	else:
		try:
			judgment = judge(
				question,
				all_employee_documents,
				evidence_result["selected_documents"],
			)
			valid_shape, shape_error = _validate_judgment_shape(
				judgment,
				question.get("reference_answer", ""),
				set(question.get("expected_evidence_ids") or []),
			)
			if not valid_shape:
				semantic_judge = {"status": "INVALID", "result": judgment, "error": shape_error}
				decision = "REJECT"
				reasons.append(shape_error)
			else:
				claims = judgment["claims"]
				required_facts = judgment["required_facts"]
				unreviewed_claims = [claim for claim in claims if not claim["supported"] or claim["contradicted"]]
				missing_or_unsupported_facts = [
					fact for fact in required_facts
					if not fact["present_in_answer"] or not fact["supported"]
				]
				claim_coverage = {
					"complete": True,
					"required_facts_complete": not missing_or_unsupported_facts,
					"unreviewed_answer_text": None,
				}
				unsupported_claims = len(unreviewed_claims)
				judge_accepts = (
					judgment["decision"] == "ACCEPT"
					and judgment["question_is_answerable"]
					and judgment["answer_contains_required_facts"]
					and judgment["answer_is_supported"]
					and judgment["evidence_is_relevant"]
					and not judgment["contains_unsupported_claims"]
					and not judgment["contains_contradictions"]
					and not unreviewed_claims
					and not missing_or_unsupported_facts
					and float(judgment["semantic_score"]) >= 0.8
				)
				decision = "ACCEPT" if judge_accepts else "REJECT"
				semantic_judge = {"status": "PASS" if judge_accepts else "REJECT", "result": judgment, "error": None}
				if not judge_accepts:
					reasons.append(judgment["reason"])
		except Exception as error:
			semantic_judge = {"status": "ERROR", "result": None, "error": str(error)}
			decision = "REJECT"
			reasons.append("Falló el evaluador semántico; la pregunta no se considera verificada.")

	return {
		"question_id": question.get("question_id"),
		"employee_id": employee_id,
		"scenario": question.get("scenario"),
		"question": question.get("question"),
		"reference_answer": question.get("reference_answer"),
		"decision": decision,
		"verified": decision == "ACCEPT",
		"reasons": reasons,
		"factual_validation": factual_result,
		"evidence_validation": {key: value for key, value in evidence_result.items() if key != "selected_documents"},
		"claim_coverage": claim_coverage,
		"required_facts": required_facts,
		"claims": claims,
		"unsupported_claims": unsupported_claims,
		"semantic_judge": semantic_judge,
	}


def validate_corpus(documents_path, questions_path, results_path, report_path, judge=None):
	documents = read_jsonl(documents_path)
	questions = read_jsonl(questions_path)
	documents_by_id = {document["document_id"]: document for document in documents}
	if len(documents_by_id) != len(documents):
		raise ValueError("El corpus contiene document_id duplicados.")
	results = [validate_question(question, documents_by_id, judge=judge) for question in questions]
	accepted = sum(result["decision"] == "ACCEPT" for result in results)
	rejected = sum(result["decision"] == "REJECT" for result in results)
	unsupported_claim_count = None
	if judge is not None and all(result["unsupported_claims"] is not None for result in results):
		unsupported_claim_count = sum(result["unsupported_claims"] for result in results)
	issues = [
		{"question_id": result["question_id"], "decision": result["decision"], "reasons": result["reasons"]}
		for result in results
		if result["decision"] != "ACCEPT"
	]
	report = {
		"total_questions": len(questions),
		"verified_questions": accepted,
		"rejected_questions": rejected,
		"pending_questions": len(questions) - accepted - rejected,
		"questions_without_evidence": sum(not question.get("expected_evidence_ids") for question in questions),
		"inconsistent_answers": sum(
			result["decision"] == "REJECT"
			and (not result["factual_validation"]["passed"] or result["semantic_judge"]["status"] == "REJECT")
			for result in results
		),
		"incorrectly_associated_evidence": sum(not result["evidence_validation"]["passed"] for result in results),
		"unsupported_claims": unsupported_claim_count,
		"question_type_distribution": dict(sorted(Counter(question.get("question_type", "unclassified") for question in questions).items())),
		"difficulty_distribution": dict(sorted(Counter(question.get("difficulty", "unclassified") for question in questions).items())),
		"semantic_judge_enabled": judge is not None,
		"manual_review_status": "pending",
		"validation_issues": issues,
	}

	results_path = Path(results_path)
	report_path = Path(report_path)
	results_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with results_path.open("w", encoding="utf-8") as output:
		for result in results:
			output.write(json.dumps(result, ensure_ascii=False) + "\n")
	with report_path.open("w", encoding="utf-8") as output:
		json.dump(report, output, ensure_ascii=False, indent=2)
	return results, report


def main():
	parser = argparse.ArgumentParser(description="Valida hechos y trazabilidad semántica del corpus de auditoría.")
	parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
	parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
	parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
	parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
	parser.add_argument("--llm-judge", action="store_true", help="Ejecuta el juez semántico claim-level OpenAI-compatible.")
	parser.add_argument("--model", default=None, help="Sobrescribe SEMANTIC_VALIDATION_MODEL.")
	args = parser.parse_args()
	judge = OpenAISemanticJudge(model=args.model) if args.llm_judge else None
	results, report = validate_corpus(args.documents, args.questions, args.results, args.report, judge=judge)
	print(json.dumps(report, ensure_ascii=False, indent=2))
	print(f"\nValidación por pregunta: {args.results}")
	print(f"Informe agregado: {args.report}")
	print(f"Preguntas procesadas: {len(results)}")
	if report["rejected_questions"]:
		raise SystemExit(1)


if __name__ == "__main__":
	main()