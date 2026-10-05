from __future__ import annotations

import unicodedata
from importlib import import_module
from threading import RLock
from typing import Any


class PrologKnowledgeManager:
	"""Load employee/task facts and evaluate deterministic Prolog rules."""

	def __init__(
		self,
		prolog: Any | None = None,
		overload_threshold: int = 3,
	) -> None:
		if overload_threshold < 0:
			raise ValueError("overload_threshold cannot be negative.")

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
		self.overload_threshold = overload_threshold
		self._registered_facts: set[str] = set()
		self._lock = RLock()
		self._initialize_base_rules()

	def _initialize_base_rules(self) -> None:
		rules = (
			"sobrecargado(Empleado) :- "
			"empleado(Empleado, _), "
			"findall(Id, tarea(Id, Empleado, 'pendiente'), Tareas), "
			"length(Tareas, Cantidad), "
			f"Cantidad > {self.overload_threshold}",
			"requiere_asistencia(Empleado) :- "
			"tarea(_, Empleado, 'retrasada')",
		)

		for rule in rules:
			self.prolog.assertz(rule)

	def limpiar_base_conocimiento(self) -> None:
		"""Remove facts while retaining the inference rules."""

		for query in (
			"retractall(empleado(_, _))",
			"retractall(tarea(_, _, _))",
			"retractall(cumplimiento(_, _, _))",
		):
			list(self.prolog.query(query))

		self._registered_facts.clear()

	def registrar_empleado(self, nombre: Any, rol: Any) -> None:
		self._assert_fact(
			f"empleado({self._atom(nombre)}, {self._atom(rol)})"
		)

	def registrar_tarea(
		self,
		identificador: Any,
		empleado: Any,
		estado: Any,
	) -> None:
		normalized_state = self._normalize_status(estado)
		self._assert_fact(
			"tarea("
			f"{self._atom(identificador)}, "
			f"{self._atom(empleado)}, "
			f"{self._atom(normalized_state)})"
		)

	def registrar_cumplimiento(
		self,
		identificador_tarea: Any,
		tipo: Any,
		valor: Any,
	) -> None:
		self._assert_fact(
			"cumplimiento("
			f"{self._atom(identificador_tarea)}, "
			f"{self._atom(tipo)}, "
			f"{self._atom(valor)})"
		)

	def consultar(self, query: str) -> list[dict[str, Any]]:
		return list(self.prolog.query(query))

	def cargar_datos(self, raw_data: list[dict[str, Any]]) -> int:
		"""Replace the current facts with supported employee/task records."""

		self.limpiar_base_conocimiento()
		loaded_records = 0

		for index, item in enumerate(raw_data):
			if not isinstance(item, dict):
				raise TypeError(f"raw_data[{index}] must be a dictionary.")

			kind = item.get("tipo")

			if kind == "empleado":
				self.registrar_empleado(
					self._required_value(item, "nombre", index),
					self._required_value(item, "rol", index),
				)
				loaded_records += 1
			elif kind == "tarea":
				task_id = self._required_value(item, "id", index)
				self.registrar_tarea(
					task_id,
					self._required_value(item, "empleado", index),
					self._required_value(item, "estado", index),
				)
				if "limite" in item:
					self.registrar_cumplimiento(
						task_id,
						"limite",
						item["limite"],
					)
				loaded_records += 1

		return loaded_records

	def inferir(self) -> dict[str, list[str]]:
		"""Return the employees matched by each base rule."""

		return {
			"empleados_sobrecargados": self._query_names(
				"sobrecargado(X)"
			),
			"requieren_asistencia": self._query_names(
				"requiere_asistencia(X)"
			),
		}

	def evaluate(
		self,
		raw_data: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Load a dataset and evaluate it as one serialized operation."""

		with self._lock:
			fact_count = self.cargar_datos(raw_data)
			return {
				"facts_loaded": fact_count > 0,
				"fact_count": fact_count,
				"prolog_results": self.inferir(),
			}

	def _assert_fact(self, fact: str) -> None:
		if fact in self._registered_facts:
			return

		self.prolog.assertz(fact)
		self._registered_facts.add(fact)

	def _query_names(self, query: str) -> list[str]:
		values = {
			str(result["X"])
			for result in self.consultar(query)
		}
		return sorted(values, key=str.casefold)

	@staticmethod
	def _atom(value: Any) -> str:
		if value is None:
			raise ValueError("Prolog fact values cannot be None.")

		text = str(value)
		escaped = (
			text.replace("\\", "\\\\")
			.replace("'", "\\'")
			.replace("\n", "\\n")
			.replace("\r", "\\r")
		)
		return f"'{escaped}'"

	@staticmethod
	def _normalize_status(status: Any) -> str:
		text = str(status).strip().casefold()
		text = "".join(
			character
			for character in unicodedata.normalize("NFD", text)
			if unicodedata.category(character) != "Mn"
		)

		if text in {"pendiente", "pending", "open", "abierta", "abierto"}:
			return "pendiente"
		if text in {"retrasada", "atrasada", "overdue", "delayed"}:
			return "retrasada"
		return text

	@staticmethod
	def _required_value(
		item: dict[str, Any],
		key: str,
		index: int,
	) -> Any:
		value = item.get(key)
		if value is None or value == "":
			raise ValueError(
				f"raw_data[{index}] requires a non-empty '{key}' value."
			)
		return value
