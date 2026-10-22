from __future__ import annotations

from datetime import date
from importlib import import_module
from math import isfinite
from threading import RLock
from typing import Any


# Rule rationale, source mapping, and applicability limits:
# docs/fundamento_reglas_auditoria_desempeno.md
PROLOG_RULES = (
	"auditoria_hallazgo(E, P, evaluador_no_identificado, [Doc]) :- "
	"evaluacion(E, P, Doc), \\+ evaluador_identificado(E, P).",
	"auditoria_hallazgo(E, P, evaluacion_sin_firma, [Doc]) :- "
	"evaluacion(E, P, Doc), \\+ evaluacion_firmada(E, P).",
	"auditoria_hallazgo(E, P, version_no_identificada, [Doc]) :- "
	"evaluacion(E, P, Doc), \\+ version_identificada(E, P).",
	"auditoria_hallazgo(E, P, objetivo_sin_evidencia(O), [ObjectiveDoc]) :- "
	"objetivo_esperado(E, P, O, ObjectiveDoc), "
	"\\+ evidencia_verificada(E, P, O, _, _).",
	"auditoria_hallazgo(E, P, calificacion_discrepante, [EvalDoc, SystemDoc]) :- "
	"calificacion_evaluacion(E, P, EvalScore, EvalDoc), "
	"calificacion_sistema(E, P, SystemScore, SystemDoc), "
	"Difference is abs(EvalScore - SystemScore), Difference > 0.000001.",
	"auditoria_hallazgo(E, P, evidencia_no_cubre_periodo, [EndDoc, LatestDoc]) :- "
	"fin_periodo(E, P, EndDate, EndDoc), "
	"ultima_evidencia(E, P, LatestDate, LatestDoc), LatestDate @< EndDate.",
	"auditoria_hallazgo(E, P, promedio_inconsistente, [AverageDoc|IndicatorDocs]) :- "
	"promedio_declarado(E, P, Reported, AverageDoc), "
	"findall(Value, indicador(E, P, Value, _), Values), Values \\= [], "
	"sum_list(Values, Total), length(Values, Count), Calculated is Total / Count, "
	"Difference is abs(Reported - Calculated), Difference > 0.000001, "
	"findall(Doc, indicador(E, P, _, Doc), IndicatorDocs).",
	"auditoria_hallazgo(E, P, dato_sensible_no_necesario, [Doc]) :- "
	"dato_sensible_no_necesario(E, P, Doc).",
	"auditoria_hallazgo(E, P, acceso_excesivo, [Doc]) :- "
	"acceso_excesivo(E, P, Doc).",
	"auditoria_hallazgo(E, P, cambio_sin_aprobacion, [Doc]) :- "
	"cambio_version(E, P, Doc), \\+ cambio_aprobado(E, P).",
	"auditoria_hallazgo(E, P, comparacion_global_sin_base, [ClaimDoc, ReviewDoc]) :- "
	"afirmacion_comparacion_global(E, P, ClaimDoc), "
	"evaluacion_comparabilidad(E, P, ReviewDoc), \\+ base_comparacion_valida(E, P).",
)

FACT_RESET_QUERIES = (
	"retractall(evaluacion(_, _, _))",
	"retractall(evaluador_identificado(_, _))",
	"retractall(evaluacion_firmada(_, _))",
	"retractall(version_identificada(_, _))",
	"retractall(objetivo_esperado(_, _, _, _))",
	"retractall(evidencia_verificada(_, _, _, _, _))",
	"retractall(calificacion_evaluacion(_, _, _, _))",
	"retractall(calificacion_sistema(_, _, _, _))",
	"retractall(fin_periodo(_, _, _, _))",
	"retractall(ultima_evidencia(_, _, _, _))",
	"retractall(promedio_declarado(_, _, _, _))",
	"retractall(indicador(_, _, _, _))",
	"retractall(dato_sensible_no_necesario(_, _, _))",
	"retractall(acceso_excesivo(_, _, _))",
	"retractall(cambio_version(_, _, _))",
	"retractall(cambio_aprobado(_, _))",
	"retractall(afirmacion_comparacion_global(_, _, _))",
	"retractall(evaluacion_comparabilidad(_, _, _))",
	"retractall(base_comparacion_valida(_, _))",
	"retractall(audit_scenario(_, _, _))",
	"retractall(audit_document(_, _, _, _, _))",
)

CORPUS_SCENARIO_RULES = (
	("trazabilidad_completa", "trazabilidad_suficiente", "control_satisfecho", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("respaldo_completo", "objetivos_respaldados", "control_satisfecho", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("evidencia_incompleta", "cobertura_de_evidencia_insuficiente", "hallazgo", ("evaluacion", "registro_auditoria")),
	("objetivos_parcialmente_respaldados", "cobertura_parcial_de_objetivos", "hallazgo", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("calificaciones_discrepantes", "conflicto_entre_calificaciones", "hallazgo", ("evaluacion", "registro_auditoria")),
	("evidencia_desactualizada", "cobertura_temporal_insuficiente", "hallazgo", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("metadatos_insuficientes", "trazabilidad_de_autoria_insuficiente", "hallazgo", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("error_de_calculo", "calculo_no_reconciliado", "hallazgo", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("minimizacion_datos_personales", "riesgo_de_minimizacion_de_datos", "hallazgo", ("evaluacion", "registro_auditoria")),
	("minimizacion_datos_personales", "riesgo_de_acceso_excesivo", "hallazgo", ("evaluacion", "registro_auditoria")),
	("cambio_sin_trazabilidad", "cambio_de_evaluacion_sin_trazabilidad", "hallazgo", ("evaluacion", "registro_auditoria")),
	("comparacion_no_sustentada", "comparacion_global_no_sustentada", "hallazgo", ("evaluacion", "evidencia_desempeno", "registro_auditoria")),
	("version_vigente_identificada", "version_vigente_trazable", "control_satisfecho", ("evaluacion", "evaluacion_corregida", "registro_auditoria")),
	("comparabilidad_limitada", "limite_de_comparabilidad", "limitacion_documentada", ("evaluacion", "marco_criterios", "registro_auditoria")),
)

CORPUS_SCENARIO_RULE = (
	"auditoria_resultado(E, P, S, Rule, Status, EvidenceIds) :- "
	"audit_scenario(E, P, S), corpus_scenario_rule(S, Rule, Status, Types), "
	"forall(member(Type, Types), audit_document(E, P, S, Type, _)), "
	"findall(Doc, (member(Type, Types), audit_document(E, P, S, Type, Doc)), RawIds), "
	"sort(RawIds, EvidenceIds)."
)


class DeterministicWorkflowPrologRules:
	"""Evaluate normalized performance-audit facts with explicit Prolog rules.

	Input records are normalized facts, not raw narrative documents. Supported
	``kind`` values are ``evaluation``, ``objective``, ``evidence``,
	``system_score``, ``period_end``, ``latest_evidence``, ``reported_average``,
	``indicator``, ``sensitive_data``, ``access_scope``, ``version_change``, and
	``comparison_assessment``. Each record must include the employee, period,
	and source document ID needed to trace a finding.

	For ``objective_without_evidence`` to be meaningful, callers must provide
	the complete expected-objective inventory for that employee and period.
	Absence of a retrieved text chunk alone is not treated as proof of missing
	evidence.

	``evaluate_corpus_documents`` applies the controlled scenario labels already
	stored in ``source_metadata.audit_scenario``. This is a gold-label benchmark
	path, not independent inference from document prose.
	"""

	def __init__(self, prolog: Any | None = None) -> None:
		if prolog is None:
			try:
				prolog_module = import_module("pyswip")
			except ImportError as exc:
				raise RuntimeError(
					"PySwip is required. Install pyswip and SWI-Prolog."
				) from exc

			try:
				prolog = prolog_module.Prolog()
			except Exception as exc:
				raise RuntimeError(
					"Could not initialize SWI-Prolog. Install SWI-Prolog "
					"and ensure its executable is available."
				) from exc

		self.prolog = prolog
		self._lock = RLock()
		self._registered_facts: set[str] = set()
		for rule in PROLOG_RULES:
			self.prolog.assertz(rule)
		self.prolog.assertz(CORPUS_SCENARIO_RULE)
		for scenario, rule_id, status, document_types in CORPUS_SCENARIO_RULES:
			types = ", ".join(self._atom(document_type) for document_type in document_types)
			self.prolog.assertz(
				"corpus_scenario_rule("
				f"{self._atom(scenario)}, {self._atom(rule_id)}, "
				f"{self._atom(status)}, [{types}])"
			)

	def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
		"""Load normalized audit facts and return traceable rule findings."""

		with self._lock:
			self._clear_facts()
			loaded_records = 0
			for index, record in enumerate(records):
				if not isinstance(record, dict):
					raise TypeError(f"records[{index}] must be a dictionary.")
				self._load_record(record, index)
				loaded_records += 1

			findings = []
			for result in self.prolog.query(
				"auditoria_hallazgo(Employee, Period, Rule, EvidenceIds)"
			):
				findings.append(
					{
						"employee_id": self._decode(result["Employee"]),
						"period_id": self._decode(result["Period"]),
						"rule": self._decode(result["Rule"]),
						"evidence_ids": [
							self._decode(document_id)
							for document_id in result["EvidenceIds"]
						],
					}
				)

			return {
				"facts_loaded": loaded_records > 0,
				"record_count": loaded_records,
				"findings": findings,
			}

	def evaluate_corpus_documents(
		self,
		documents: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Apply the corpus's explicit scenario labels and return cited results.

		This method evaluates the controlled labels written by the corpus
		generator. It is suitable for reproducing the scenario benchmark, but it
		must not be reported as detecting the scenario from document content.
		"""

		with self._lock:
			self._clear_facts()
			case_scenarios: dict[tuple[str, str], str] = {}
			case_document_types: dict[tuple[str, str], set[str]] = {}
			document_ids: set[str] = set()
			loaded_documents = 0

			for index, document in enumerate(documents):
				if not isinstance(document, dict):
					raise TypeError(f"documents[{index}] must be a dictionary.")
				metadata = document.get("source_metadata")
				if not isinstance(metadata, dict):
					raise ValueError(
						f"documents[{index}] requires source_metadata."
					)

				employee = self._required_text(document, "employee_id", index)
				period = self._required_text(document, "evaluation_period", index)
				document_id = self._required_text(document, "document_id", index)
				document_type = self._required_text(document, "document_type", index)
				scenario = self._required_text(metadata, "audit_scenario", index)
				case_key = (employee, period)

				if document_id in document_ids:
					raise ValueError(f"Duplicate document_id: {document_id}")
				document_ids.add(document_id)

				previous_scenario = case_scenarios.setdefault(case_key, scenario)
				if previous_scenario != scenario:
					raise ValueError(
					f"Conflicting audit scenarios for {employee}/{period}: "
					f"{previous_scenario!r} and {scenario!r}."
					)
				case_document_types.setdefault(case_key, set()).add(document_type)

				self._assert_fact(
					"audit_scenario("
					f"{self._atom(employee)}, {self._atom(period)}, "
					f"{self._atom(scenario)})"
				)
				self._assert_fact(
					"audit_document("
					f"{self._atom(employee)}, {self._atom(period)}, "
					f"{self._atom(scenario)}, {self._atom(document_type)}, "
					f"{self._atom(document_id)})"
				)
				loaded_documents += 1

			for (employee, period), scenario in case_scenarios.items():
				defined_types = case_document_types[(employee, period)]
				required_types = {
					document_type
					for rule_scenario, _, _, document_types in CORPUS_SCENARIO_RULES
					if rule_scenario == scenario
					for document_type in document_types
				}
				if not required_types:
					raise ValueError(f"Unsupported audit scenario: {scenario!r}")
				missing_types = required_types - defined_types
				if missing_types:
					raise ValueError(
						f"{employee}/{period} scenario {scenario!r} is missing "
						f"document types: {sorted(missing_types)}"
					)

			results = []
			for result in self.prolog.query(
				"auditoria_resultado(Employee, Period, Scenario, Rule, Status, EvidenceIds)"
			):
				results.append(
					{
						"employee_id": self._decode(result["Employee"]),
						"period_id": self._decode(result["Period"]),
						"scenario": self._decode(result["Scenario"]),
						"rule_id": self._decode(result["Rule"]),
						"status": self._decode(result["Status"]),
						"evidence_ids": [
							self._decode(document_id)
							for document_id in result["EvidenceIds"]
						],
					}
				)

			return {
				"facts_loaded": loaded_documents > 0,
				"document_count": loaded_documents,
				"case_count": len(case_scenarios),
				"results": results,
			}

	def _load_record(self, record: dict[str, Any], index: int) -> None:
		kind = record.get("kind")
		if kind not in {
			"evaluation",
			"objective",
			"evidence",
			"system_score",
			"period_end",
			"latest_evidence",
			"reported_average",
			"indicator",
			"sensitive_data",
			"access_scope",
			"version_change",
			"comparison_assessment",
		}:
			raise ValueError(f"records[{index}] has unsupported kind: {kind!r}")

		employee = self._required_text(record, "employee_id", index)
		period = self._required_text(record, "period_id", index)
		document_id = self._required_text(record, "document_id", index)
		e = self._atom(employee)
		p = self._atom(period)
		doc = self._atom(document_id)

		if kind == "evaluation":
			self._assert_fact(f"evaluacion({e}, {p}, {doc})")
			self._assert_flag(record, "evaluator_id", "evaluador_identificado", e, p)
			if self._required_boolean(record, "signed", index):
				self._assert_fact(f"evaluacion_firmada({e}, {p})")
			self._assert_flag(record, "version", "version_identificada", e, p)
			if "score" in record:
				score = self._number(record["score"], f"records[{index}].score")
				self._assert_fact(
					f"calificacion_evaluacion({e}, {p}, {score}, {doc})"
				)
			if record.get("claims_global_comparison") is True:
				self._assert_fact(
					f"afirmacion_comparacion_global({e}, {p}, {doc})"
				)
		elif kind == "objective":
			objective = self._atom(
				self._required_text(record, "objective_id", index)
			)
			self._assert_fact(
				f"objetivo_esperado({e}, {p}, {objective}, {doc})"
			)
		elif kind == "evidence":
			objective = self._atom(
				self._required_text(record, "objective_id", index)
			)
			if self._required_boolean(record, "verified", index):
				self._assert_fact(
					f"evidencia_verificada({e}, {p}, {objective}, {doc}, verificada)"
				)
		elif kind == "system_score":
			score = self._number(record.get("score"), f"records[{index}].score")
			self._assert_fact(f"calificacion_sistema({e}, {p}, {score}, {doc})")
		elif kind == "period_end":
			end_date = self._iso_date(record.get("date"), f"records[{index}].date")
			self._assert_fact(f"fin_periodo({e}, {p}, {self._atom(end_date)}, {doc})")
		elif kind == "latest_evidence":
			last_date = self._iso_date(record.get("date"), f"records[{index}].date")
			self._assert_fact(
				f"ultima_evidencia({e}, {p}, {self._atom(last_date)}, {doc})"
			)
		elif kind == "reported_average":
			value = self._number(record.get("value"), f"records[{index}].value")
			self._assert_fact(f"promedio_declarado({e}, {p}, {value}, {doc})")
		elif kind == "indicator":
			value = self._number(record.get("value"), f"records[{index}].value")
			self._assert_fact(f"indicador({e}, {p}, {value}, {doc})")
		elif kind == "sensitive_data":
			if self._required_boolean(record, "unnecessary", index):
				self._assert_fact(
					f"dato_sensible_no_necesario({e}, {p}, {doc})"
				)
		elif kind == "access_scope":
			if self._required_boolean(record, "excessive", index):
				self._assert_fact(f"acceso_excesivo({e}, {p}, {doc})")
		elif kind == "version_change":
			self._assert_fact(f"cambio_version({e}, {p}, {doc})")
			if self._required_boolean(record, "approved", index):
				self._assert_fact(f"cambio_aprobado({e}, {p})")
		elif kind == "comparison_assessment":
			self._assert_fact(f"evaluacion_comparabilidad({e}, {p}, {doc})")
			if self._required_boolean(record, "common_basis", index):
				self._assert_fact(f"base_comparacion_valida({e}, {p})")

	def _clear_facts(self) -> None:
		for query in FACT_RESET_QUERIES:
			list(self.prolog.query(query))
		self._registered_facts.clear()

	def _assert_fact(self, fact: str) -> None:
		if fact in self._registered_facts:
			return
		self.prolog.assertz(fact)
		self._registered_facts.add(fact)

	def _assert_flag(
		self,
		record: dict[str, Any],
		field: str,
		predicate: str,
		employee: str,
		period: str,
	) -> None:
		value = record.get(field)
		if value is not None and str(value).strip():
			self._assert_fact(
				f"{predicate}({employee}, {period})"
			)

	@staticmethod
	def _required_text(record: dict[str, Any], key: str, index: int) -> str:
		value = record.get(key)
		if not isinstance(value, str) or not value.strip():
			raise ValueError(
				f"records[{index}] requires a non-empty string '{key}'."
			)
		return value.strip()

	@staticmethod
	def _required_boolean(record: dict[str, Any], key: str, index: int) -> bool:
		value = record.get(key)
		if not isinstance(value, bool):
			raise ValueError(f"records[{index}] requires a boolean '{key}'.")
		return value

	@staticmethod
	def _number(value: Any, field: str) -> str:
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			raise ValueError(f"{field} must be numeric.")
		if not isfinite(value):
			raise ValueError(f"{field} must be finite.")
		return repr(value)

	@staticmethod
	def _iso_date(value: Any, field: str) -> str:
		if not isinstance(value, str):
			raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD).")
		try:
			return date.fromisoformat(value).isoformat()
		except ValueError as exc:
			raise ValueError(
				f"{field} must be an ISO date (YYYY-MM-DD)."
			) from exc

	@staticmethod
	def _atom(value: str) -> str:
		escaped = (
			value.replace("\\", "\\\\")
			.replace("'", "\\'")
			.replace("\n", "\\n")
			.replace("\r", "\\r")
		)
		return f"'{escaped}'"

	@staticmethod
	def _decode(value: Any) -> str:
		if isinstance(value, bytes):
			return value.decode("utf-8")
		return str(value)
