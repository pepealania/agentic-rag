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

SPANISH_NUMBERS = {
	"cero": 0,
	"un": 1,
	"uno": 1,
	"una": 1,
	"dos": 2,
	"tres": 3,
	"cuatro": 4,
	"cinco": 5,
	"seis": 6,
	"siete": 7,
	"ocho": 8,
	"nueve": 9,
	"diez": 10,
	"once": 11,
	"doce": 12,
	"trece": 13,
	"catorce": 14,
	"quince": 15,
	"dieciséis": 16,
	"diecisiete": 17,
	"dieciocho": 18,
	"diecinueve": 19,
	"veinte": 20,
}

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


def parse_spanish_number(value):
	cleaned = value.strip().lower().strip(".,;:")
	if cleaned.isdigit():
		return int(cleaned)
	return SPANISH_NUMBERS.get(cleaned)


def _contains_number(text, number):
	without_employee_ids = re.sub(r"EMP-AUD-\d+", "", text, flags=re.IGNORECASE)
	if re.search(rf"(?<!\d){number}(?!\d)", without_employee_ids):
		return True
	return any(
		word in without_employee_ids.lower()
		for word, value in SPANISH_NUMBERS.items()
		if value == number
	)


def _documents_of_type(documents, document_type):
	return [document for document in documents if document.get("document_type") == document_type]


def _find_objective_total(documents):
	for document in _documents_of_type(documents, "evaluacion") + _documents_of_type(documents, "evidencia_desempeno"):
		match = re.search(r"([\wáéíóú]+)\s+objetivos", document.get("content", ""), flags=re.IGNORECASE)
		if match:
			count = parse_spanish_number(match.group(1))
			if count is not None:
				return count
	return None


def _find_objective_counts(documents):
	for document in _documents_of_type(documents, "evidencia_desempeno") + _documents_of_type(documents, "registro_auditoria"):
		content = document.get("content", "")
		match = re.search(r"([\wáéíóú]+)\s+de\s+([\wáéíóú]+)\s+objetivos", content, flags=re.IGNORECASE)
		if match:
			supported = parse_spanish_number(match.group(1))
			total = parse_spanish_number(match.group(2))
			if supported is not None and total is not None:
				return supported, total

	for document in _documents_of_type(documents, "evidencia_desempeno"):
		match = re.search(r"contiene\s+([\wáéíóú]+)\s+registros\s+verificables", document.get("content", ""), flags=re.IGNORECASE)
		if match:
			supported = parse_spanish_number(match.group(1))
			total = _find_objective_total(documents)
			if supported is not None and total is not None:
				return supported, total
	return None, _find_objective_total(documents)


def _scores_from_documents(documents, document_types):
	values = []
	for document in documents:
		if document.get("document_type") not in document_types:
			continue
		values.extend(
			float(match.group(1))
			for match in re.finditer(r"(\d+(?:\.\d+)?)\s*/\s*5", document.get("content", ""))
		)
	return values


def _factual_checks(question, all_employee_documents):
	scenario = question.get("scenario", "")
	answer = question.get("reference_answer", "")
	answer_lower = answer.lower()
	checks = []

	def record(name, passed, detail, source_documents):
		checks.append({
			"check": name,
			"passed": bool(passed),
			"detail": detail,
			"source_document_ids": [document.get("document_id") for document in source_documents],
		})

	record("respuesta_no_vacia", bool(answer.strip()), "La respuesta de referencia debe contener texto.", [])

	if scenario == "error_de_calculo":
		source_documents = _documents_of_type(all_employee_documents, "evidencia_desempeno")
		source_values = []
		for document in source_documents:
			for numerator, denominator in re.findall(
				r"(\d+(?:\.\d+)?)\s+de\s+(\d+(?:\.\d+)?)", document.get("content", ""), flags=re.IGNORECASE
			):
				if float(denominator) != 0:
					source_values.append(float(numerator) / float(denominator) * 100)
		expected_average = sum(source_values) / len(source_values) if source_values else None
		answer_percentages = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)]
		passed = expected_average is not None and any(abs(value - expected_average) < 0.001 for value in answer_percentages)
		record(
			"promedio_calculado_desde_indicadores_fuente",
			passed,
			{"source_percentages": source_values, "expected_average": expected_average, "answer_percentages": answer_percentages},
			source_documents,
		)
		reported_documents = _documents_of_type(all_employee_documents, "evaluacion")
		reported_percentages = [
			float(value)
			for document in reported_documents
			for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", document.get("content", ""))
		]
		record(
			"discrepancia_con_la_evaluacion_reportada",
			bool(expected_average is not None and reported_percentages and any(abs(value - expected_average) >= 0.001 for value in reported_percentages)),
			{"evaluation_percentages": reported_percentages, "computed_average": expected_average},
			reported_documents + source_documents,
		)
	elif scenario in {"evidencia_incompleta", "objetivos_parcialmente_respaldados"}:
		source_documents = _documents_of_type(all_employee_documents, "evaluacion") + _documents_of_type(all_employee_documents, "evidencia_desempeno") + _documents_of_type(all_employee_documents, "registro_auditoria")
		supported, total = _find_objective_counts(all_employee_documents)
		if total is None:
			total = _find_objective_total(all_employee_documents)
		unsupported = total - supported if total is not None and supported is not None else None
		if scenario == "objetivos_parcialmente_respaldados":
			performance_text = " ".join(document.get("content", "") for document in _documents_of_type(all_employee_documents, "evidencia_desempeno"))
			audit_text = " ".join(document.get("content", "") for document in _documents_of_type(all_employee_documents, "registro_auditoria"))
			supported_ids = set(re.findall(r"OBJ-\d+", performance_text))
			all_objective_ids = set(re.findall(r"OBJ-\d+", performance_text + " " + audit_text))
			unsupported_ids = all_objective_ids - supported_ids
			count_answered = (
				unsupported is not None
				and _contains_number(answer, supported)
				and _contains_number(answer, unsupported)
				and any(token in answer_lower for token in ("no puede determinar", "no pueden determinar", "sin determinar", "no se localizó"))
			)
			ids_answered = bool(supported_ids) and supported_ids <= set(re.findall(r"OBJ-\d+", answer)) and (
				not unsupported_ids
				or any(objective_id in answer and any(token in answer_lower for token in ("sin determinar", "no se localizó", "no permite confirmar")) for objective_id in unsupported_ids)
			)
			passed = count_answered or ids_answered
		else:
			passed = (
				total is not None
				and supported is not None
				and unsupported is not None
				and total == supported + unsupported
				and _contains_number(answer, supported)
				and _contains_number(answer, unsupported)
			)
		record(
			"objetivos_totales_respaldados_y_pendientes",
			passed,
			{"total_from_evaluation": total, "supported_from_evidence": supported, "unsupported_calculated": unsupported},
			source_documents,
		)
	elif scenario == "calificaciones_discrepantes":
		source_documents = _documents_of_type(all_employee_documents, "evaluacion") + _documents_of_type(all_employee_documents, "registro_sistema")
		scores = _scores_from_documents(source_documents, {"evaluacion", "registro_sistema"})
		passed = len(set(scores)) >= 2 and all(f"{score:.1f}/5" in answer for score in scores)
		record("ambas_calificaciones_fuente_aparecen_en_respuesta", passed, {"source_scores": scores}, source_documents)
		audit_documents = _documents_of_type(all_employee_documents, "registro_auditoria")
		record(
			"conclusion_de_no_conciliacion",
			bool(audit_documents and any(token in answer_lower for token in ("sin conciliación", "no hay conciliación", "no se conciliaron"))),
			{"answer_mentions_unreconciled_values": any(token in answer_lower for token in ("sin conciliación", "no hay conciliación", "no se conciliaron"))},
			audit_documents,
		)
	elif scenario == "cambio_sin_trazabilidad":
		source_documents = _documents_of_type(all_employee_documents, "historial_versiones")
		scores = _scores_from_documents(source_documents, {"historial_versiones"})
		passed = len(set(scores)) >= 2 and all(f"{score:.1f}/5" in answer for score in scores)
		passed = passed and all(token in answer_lower for token in ("autor", "motivo", "aprobación"))
		record("cambio_y_falta_de_trazabilidad", passed, {"history_scores": scores}, source_documents)
	elif scenario == "version_vigente_identificada":
		current_documents = _documents_of_type(all_employee_documents, "evaluacion_corregida")
		scores = _scores_from_documents(current_documents, {"evaluacion_corregida"})
		current_text = " ".join(document.get("content", "") for document in current_documents).lower()
		passed = bool(scores and "versión 2" in current_text and "vigente" in current_text)
		passed = passed and any(f"{score:.1f}/5" in answer for score in scores) and "versión 2" in answer_lower and "vigente" in answer_lower
		passed = passed and any(token in answer_lower for token in ("aprobada", "aprobación"))
		record("version_vigente_y_calificacion", passed, {"current_scores": scores}, current_documents)
	elif scenario == "evidencia_desactualizada":
		source_documents = _documents_of_type(all_employee_documents, "evidencia_desempeno")
		latest_date = max((document.get("document_date", "") for document in source_documents), default="")
		passed = bool(latest_date and latest_date < "2026-04-01") and any(
			token in answer_lower for token in ("no.", "no permite", "no puede")
		)
		record("limite_temporal_de_la_evidencia", passed, {"latest_performance_evidence_date": latest_date}, source_documents)
	elif scenario == "metadatos_insuficientes":
		source_documents = _documents_of_type(all_employee_documents, "evaluacion") + _documents_of_type(all_employee_documents, "evidencia_desempeno")
		missing_field_evidence = {
			"evaluator_identity": any(token in answer_lower for token in ("evaluador", "identidad", "quien evaluó", "quién evaluó")),
			"signature": "firma" in answer_lower,
			"version": "versión" in answer_lower or "version" in answer_lower,
			"criterion_link": "criterio" in answer_lower,
		}
		passed = all(missing_field_evidence.values())
		record("metadatos_ausentes_identificados", passed, missing_field_evidence, source_documents)
	elif scenario == "minimizacion_datos_personales":
		source_documents = _documents_of_type(all_employee_documents, "evaluacion") + _documents_of_type(all_employee_documents, "registro_acceso")
		passed = any(token in answer_lower for token in ("información personal", "datos personales")) and any(
			token in answer_lower for token in ("acceso", "alcance")
		)
		record("riesgo_de_privacidad_y_acceso", passed, {"answer_mentions_sensitive_data_and_access": passed}, source_documents)
	elif scenario in {"comparacion_no_sustentada", "comparabilidad_limitada"}:
		source_documents = _documents_of_type(all_employee_documents, "evidencia_desempeno") + _documents_of_type(all_employee_documents, "marco_criterios") + _documents_of_type(all_employee_documents, "registro_auditoria")
		passed = answer_lower.startswith("no") and any(
			token in answer_lower for token in ("base comparable", "escala común", "criterios", "comparativa")
		)
		record("limite_de_comparacion_respaldado", passed, {"answer_rejects_unsupported_comparison": passed}, source_documents)
	elif scenario in {"respaldo_completo", "trazabilidad_completa"}:
		source_documents = _documents_of_type(all_employee_documents, "evaluacion") + _documents_of_type(all_employee_documents, "evidencia_desempeno")
		content = " ".join(document.get("content", "") for document in source_documents)
		objective_count = _find_objective_total(source_documents)
		passed = bool(objective_count and "objetivos" in content.lower()) and any(
			token in answer_lower for token in ("evidencia", "entregas", "vincula")
		)
		record("objetivos_y_evidencia_completa", passed, {"objective_count_from_sources": objective_count}, source_documents)
	else:
		record("escenario_factual_reconocido", False, {"unhandled_scenario": scenario}, [])

	return checks


def _judge_schema_is_valid(result, allowed_evidence_ids):
	if not isinstance(result, dict):
		return False, "El juez no devolvió un objeto JSON."
	required_boolean_fields = (
		"question_is_answerable",
		"answer_is_supported",
		"evidence_is_relevant",
		"contains_unsupported_claims",
		"contains_contradictions",
	)
	if any(not isinstance(result.get(field), bool) for field in required_boolean_fields):
		return False, "El juez omitió campos booleanos obligatorios."
	if result.get("decision") not in {"ACCEPT", "REJECT"}:
		return False, "La decisión del juez debe ser ACCEPT o REJECT."
	if not isinstance(result.get("reason"), str) or not result["reason"].strip():
		return False, "El juez debe justificar su decisión."
	try:
		score = float(result.get("semantic_score"))
	except (TypeError, ValueError):
		return False, "semantic_score no es numérico."
	if not 0 <= score <= 1:
		return False, "semantic_score debe estar entre 0 y 1."
	claims = result.get("claims")
	if not isinstance(claims, list) or not claims:
		return False, "El juez debe justificar la decisión mediante claims."
	for claim in claims:
		if (
			not isinstance(claim, dict)
			or not isinstance(claim.get("supported"), bool)
			or not isinstance(claim.get("claim"), str)
			or not claim.get("claim", "").strip()
			or not isinstance(claim.get("reason"), str)
			or not claim.get("reason", "").strip()
		):
			return False, "Hay una afirmación sin estado de respaldo válido."
		citations = claim.get("document_ids")
		if not isinstance(citations, list) or any(document_id not in allowed_evidence_ids for document_id in citations):
			return False, "El juez citó una evidencia no esperada o mal formada."
		if claim["supported"] and not citations:
			return False, "Una afirmación respaldada carece de identificadores de evidencia."
	return True, ""


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
		document_context = [
			{"document_id": document["document_id"], "document_type": document["document_type"], "content": document["content"]}
			for document in employee_documents
		]
		prompt = {
			"task": "Evalúa la respuesta de referencia únicamente contra los documentos originales. No la des por correcta solo porque coincida con una conclusión de auditoría; verifica los datos fuente. Evalúa cada afirmación y justifícala con document_ids exactos.",
			"question": question.get("question"),
			"reference_answer": question.get("reference_answer"),
			"scenario": question.get("scenario"),
			"expected_evidence_ids": sorted(expected_ids),
			"documents": document_context,
			"required_output": {
				"question_is_answerable": "boolean",
				"answer_is_supported": "boolean",
				"evidence_is_relevant": "boolean",
				"contains_unsupported_claims": "boolean",
				"contains_contradictions": "boolean",
				"semantic_score": "number 0..1",
				"decision": "ACCEPT or REJECT",
				"reason": "string",
				"claims": [{"claim": "string", "supported": "boolean", "document_ids": ["expected document IDs"], "reason": "string"}],
			},
		}
		response = self.client.chat.completions.create(
			model=self.model,
			temperature=0,
			response_format={"type": "json_object"},
			messages=[
				{"role": "system", "content": "Eres un evaluador independiente de QA para RAG. Devuelve únicamente JSON válido. No uses conocimiento externo."},
				{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
			],
		)
		return json.loads(response.choices[0].message.content)


def validate_question(question, documents_by_id, judge=None):
	question_id = question.get("question_id")
	employee_id = question.get("employee_id")
	scenario = question.get("scenario")
	evidence_ids = question.get("expected_evidence_ids") or []
	all_employee_documents = [
		document for document in documents_by_id.values()
		if document.get("employee_id") == employee_id
	]
	expected_documents = [documents_by_id[document_id] for document_id in evidence_ids if document_id in documents_by_id]
	actual_types = {document.get("document_type") for document in expected_documents}
	expected_types = set(question.get("expected_evidence_types") or [])
	required_types = REQUIRED_EVIDENCE_TYPES.get(scenario, set())
	association_errors = []
	if not evidence_ids:
		association_errors.append("La pregunta no incluye evidencias.")
	if len(evidence_ids) != len(set(evidence_ids)):
		association_errors.append("La lista de evidencias contiene IDs duplicados.")
	for document_id in evidence_ids:
		if document_id not in documents_by_id:
			association_errors.append(f"No existe el documento {document_id}.")
	for document in expected_documents:
		if document.get("employee_id") != employee_id:
			association_errors.append(f"{document['document_id']} pertenece a otro empleado.")
		if document.get("source_metadata", {}).get("audit_scenario") != scenario:
			association_errors.append(f"{document['document_id']} pertenece a otro escenario.")
	if actual_types != expected_types:
		association_errors.append("Los tipos de evidencia reales no coinciden con los declarados.")
	if not required_types or not required_types <= actual_types:
		association_errors.append("Falta al menos un tipo de evidencia indispensable para el escenario.")

	factual_checks = _factual_checks(question, all_employee_documents)
	factual_failures = [check for check in factual_checks if not check["passed"]]
	semantic_judge = {"status": "NOT_RUN", "result": None, "error": None}
	unsupported_claim_count = None
	decision = "NEEDS_REVIEW"
	reasons = list(association_errors) + [check["check"] + ": " + str(check["detail"]) for check in factual_failures]

	if association_errors or factual_failures:
		decision = "REJECT"
	elif judge is None:
		reasons.append("Juez semántico independiente no ejecutado; requiere revisión.")
	elif judge is not None:
		try:
			judge_result = judge(question, all_employee_documents, expected_documents)
			is_valid, schema_error = _judge_schema_is_valid(judge_result, set(evidence_ids))
			if not is_valid:
				semantic_judge = {"status": "INVALID", "result": judge_result, "error": schema_error}
				reasons.append(schema_error)
			else:
				unsupported_claim_count = sum(not claim["supported"] for claim in judge_result["claims"])
				judge_accepts = (
					judge_result["decision"] == "ACCEPT"
					and judge_result["question_is_answerable"]
					and judge_result["answer_is_supported"]
					and judge_result["evidence_is_relevant"]
					and not judge_result["contains_unsupported_claims"]
					and not judge_result["contains_contradictions"]
					and unsupported_claim_count == 0
					and float(judge_result["semantic_score"]) >= 0.8
				)
				semantic_judge = {"status": "PASS" if judge_accepts else "REJECT", "result": judge_result, "error": None}
				decision = "ACCEPT" if judge_accepts else "REJECT"
				if not judge_accepts:
					reasons.append(judge_result.get("reason", "El juez semántico rechazó la respuesta."))
		except Exception as error:
			semantic_judge = {"status": "ERROR", "result": None, "error": str(error)}
			reasons.append("Falló el evaluador semántico; requiere revisión.")

	return {
		"question_id": question_id,
		"employee_id": employee_id,
		"scenario": scenario,
		"question": question.get("question"),
		"reference_answer": question.get("reference_answer"),
		"expected_evidence_ids": evidence_ids,
		"decision": decision,
		"verified": decision == "ACCEPT",
		"reasons": reasons,
		"factual_checks": factual_checks,
		"evidence_association": {
			"passed": not association_errors,
			"errors": association_errors,
			"actual_types": sorted(str(value) for value in actual_types),
			"required_types": sorted(required_types),
		},
		"semantic_judge": semantic_judge,
		"unsupported_claims": unsupported_claim_count,
	}


def build_report(questions, results, judge_enabled):
	by_question_id = {result["question_id"]: result for result in results}
	verified = sum(result["decision"] == "ACCEPT" for result in results)
	rejected = sum(result["decision"] == "REJECT" for result in results)
	unsupported_claims = None
	if judge_enabled and all(result["unsupported_claims"] is not None for result in results):
		unsupported_claims = sum(result["unsupported_claims"] for result in results)
	return {
		"total_questions": len(questions),
		"verified_questions": verified,
		"rejected_questions": rejected,
		"pending_questions": len(questions) - verified - rejected,
		"questions_without_evidence": sum(not question.get("expected_evidence_ids") for question in questions),
		"inconsistent_answers": sum(
			any(not check["passed"] for check in result["factual_checks"])
			or result["semantic_judge"]["status"] == "REJECT"
			for result in results
		),
		"incorrectly_associated_evidence": sum(not result["evidence_association"]["passed"] for result in results),
		"unsupported_claims": unsupported_claims,
		"question_type_distribution": dict(sorted(Counter(question.get("question_type", "unclassified") for question in questions).items())),
		"difficulty_distribution": dict(sorted(Counter(question.get("difficulty", "unclassified") for question in questions).items())),
		"semantic_judge_enabled": judge_enabled,
		"manual_review_status": "pending",
		"validation_issues": [
			{"question_id": result["question_id"], "decision": result["decision"], "reasons": result["reasons"]}
			for result in results
			if result["decision"] != "ACCEPT"
		],
		"duplicate_question_ids": len(questions) - len(by_question_id),
	}


def validate_corpus(documents_path, questions_path, results_path, report_path, judge=None):
	documents = read_jsonl(documents_path)
	questions = read_jsonl(questions_path)
	documents_by_id = {document["document_id"]: document for document in documents}
	if len(documents_by_id) != len(documents):
		raise ValueError("El corpus contiene document_id duplicados.")
	results = [validate_question(question, documents_by_id, judge=judge) for question in questions]
	report = build_report(questions, results, judge_enabled=judge is not None)

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
	parser = argparse.ArgumentParser(description="Valida preguntas de auditoría sin ejecutar el generador.")
	parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
	parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
	parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
	parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
	parser.add_argument("--llm-judge", action="store_true", help="Activa el juez semántico OpenAI-compatible.")
	parser.add_argument("--model", default=None, help="Sobrescribe SEMANTIC_VALIDATION_MODEL.")
	args = parser.parse_args()

	judge = OpenAISemanticJudge(model=args.model) if args.llm_judge else None
	results, report = validate_corpus(args.documents, args.questions, args.results, args.report, judge=judge)
	print(json.dumps(report, ensure_ascii=False, indent=2))
	print(f"\nValidación individual: {args.results}")
	print(f"Informe: {args.report}")
	print(f"Preguntas evaluadas: {len(results)}")
	if report["rejected_questions"]:
		raise SystemExit(1)


if __name__ == "__main__":
	main()