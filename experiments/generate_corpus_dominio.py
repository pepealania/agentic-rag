import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "auditoria_desempeno"
PROCESSED = ROOT / "data" / "processed" / "auditoria_desempeno"
EVALUATION = ROOT / "data" / "evaluation"

for directory in (RAW, PROCESSED, EVALUATION):
	directory.mkdir(parents=True, exist_ok=True)


# Synthetic cases designed to audit the information used in employee reviews.
CASES = [
	{
		"employee_id": "EMP-AUD-001",
		"area": "Desarrollo",
		"scenario": "trazabilidad_completa",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación semestral asigna 4.0/5 a EMP-AUD-001 para el periodo 2026-S1. La persona evaluadora identificada como EVAL-014 firmó la versión 1 el 2026-06-30."),
			("evidencia_desempeno", "2026-06-28", "El registro de entregas lista ocho objetivos acordados y ocho entregas aceptadas durante 2026-S1. Cada entrega tiene fecha, responsable y enlace a su registro de proyecto."),
			("registro_auditoria", "2026-07-02", "La auditoría vinculó la evaluación versión 1 con los ocho registros de entrega. Identificó coincidencia de periodo, empleado y objetivos; no encontró modificaciones posteriores."),
		],
		"question": "¿Qué elementos permiten rastrear la calificación de EMP-AUD-001 hasta la evidencia del periodo?",
		"answer": "La evaluación identifica a EVAL-014, fecha y versión; el registro de auditoría la vincula con ocho entregas fechadas del periodo 2026-S1.",
		"evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "integracion_multifuente",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-002",
		"area": "Operaciones",
		"scenario": "evidencia_incompleta",
		"objective_count": 10,
		"supported_count": 6,
		"unsupported_count": 4,
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-002 afirma que cumplió diez objetivos y asigna 5.0/5 para 2026-S1."),
			("evidencia_desempeno", "2026-06-25", "El expediente contiene seis registros verificables de objetivos cumplidos para EMP-AUD-002. No se localizaron registros de los otros cuatro objetivos mencionados en la evaluación."),
			("registro_auditoria", "2026-07-03", "La auditoría confirmó seis evidencias vinculables y marcó cuatro objetivos sin respaldo documental. La calificación no incluye una explicación de esta diferencia."),
		],
		"question": "¿Cuántos de los diez objetivos de EMP-AUD-002 tienen respaldo documental localizado?",
		"answer": "Se localizaron seis respaldos; cuatro objetivos mencionados en la evaluación no tienen evidencia localizada.",
		"evidence_types": ["evidencia_desempeno", "registro_auditoria"],
		"question_type": "busqueda_directa",
		"difficulty": "baja",
	},
	{
		"employee_id": "EMP-AUD-003",
		"area": "Ventas",
		"scenario": "calificaciones_discrepantes",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación firmada por la jefatura de EMP-AUD-003 asigna 4.5/5 para 2026-S1."),
			("registro_sistema", "2026-06-30", "El sistema de evaluación conserva una calificación de 2.5/5 para EMP-AUD-003 en 2026-S1. El registro no incluye nota que explique la diferencia respecto de la evaluación firmada."),
			("registro_auditoria", "2026-07-02", "La auditoría detectó dos calificaciones distintas para la misma persona y periodo. No encontró conciliación, aprobación de cambio ni identificación de cuál valor debe considerarse definitivo."),
		],
		"question": "¿Qué conclusión permite sostener la auditoría sobre las calificaciones de EMP-AUD-003?",
		"answer": "Hay una discrepancia entre 4.5/5 en la evaluación firmada y 2.5/5 en el sistema; no hay conciliación que permita elegir una como definitiva.",
		"evidence_types": ["evaluacion", "registro_sistema", "registro_auditoria"],
		"question_type": "deteccion_contradiccion",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-004",
		"area": "Administración",
		"scenario": "evidencia_desactualizada",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-004 cubre el periodo 2026-S1 y afirma cumplimiento estable durante los seis meses."),
			("evidencia_desempeno", "2026-03-31", "Los registros de actividad disponibles para EMP-AUD-004 cubren enero, febrero y marzo de 2026. No se adjuntaron registros de abril, mayo ni junio."),
			("registro_auditoria", "2026-07-04", "La auditoría observó que la evidencia disponible termina el 2026-03-31, aunque la evaluación cubre hasta el 2026-06-30. El estado de los objetivos del segundo trimestre no puede verificarse con este expediente."),
		],
		"question": "¿Puede verificarse con este expediente el desempeño de EMP-AUD-004 durante abril-junio?",
		"answer": "No. La evidencia disponible termina el 31 de marzo y la auditoría no puede verificar el segundo trimestre.",
		"evidence_types": ["evidencia_desempeno", "registro_auditoria"],
		"question_type": "suficiencia_evidencia",
		"difficulty": "baja",
	},
	{
		"employee_id": "EMP-AUD-005",
		"area": "Desarrollo",
		"scenario": "metadatos_insuficientes",
		"documents": [
			("evaluacion", "2026-06-30", "El formulario de EMP-AUD-005 asigna 3.8/5. El archivo no identifica a la persona evaluadora, no contiene firma y no indica número de versión."),
			("evidencia_desempeno", "2026-06-20", "Se adjuntan tres informes de proyecto de EMP-AUD-005, pero ninguno indica a qué criterio de evaluación corresponde."),
			("registro_auditoria", "2026-07-01", "La auditoría no pudo reconstruir quién aprobó la calificación ni vincular de forma inequívoca los tres informes con los criterios evaluados. Faltan identidad de evaluador, firma y versión del formulario."),
		],
		"question": "¿Qué datos faltan para reconstruir la autoría y el fundamento de la evaluación de EMP-AUD-005?",
		"answer": "Faltan la identidad y firma de quien evaluó, la versión del formulario y la vinculación de cada informe con un criterio.",
		"evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "integracion_multifuente",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-006",
		"area": "Operaciones",
		"scenario": "error_de_calculo",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-006 informa un promedio de 90 % en dos indicadores para 2026-S1."),
			("evidencia_desempeno", "2026-06-28", "Los indicadores registrados son 8 de 10 y 6 de 10. La hoja de cálculo indica que el promedio simple de ambos porcentajes es 90 %."),
			("registro_auditoria", "2026-07-02", "La auditoría recalculó el promedio simple: (80 % + 60 %) / 2 = 70 %. El 90 % informado no coincide con los valores fuente ni con la fórmula declarada."),
		],
		"question": "¿Qué promedio resulta de los dos indicadores de EMP-AUD-006 y qué discrepancia encontró la auditoría?",
		"answer": "El promedio simple es 70 %; la evaluación informa 90 %, que no coincide con los valores fuente ni con la fórmula declarada.",
		"evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "verificacion_calculo",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-007",
		"area": "Administración",
		"scenario": "minimizacion_datos_personales",
		"documents": [
			("evaluacion", "2026-06-30", "El anexo de evaluación de EMP-AUD-007 incorpora información personal sensible que no está vinculada con los criterios de desempeño revisados."),
			("registro_acceso", "2026-07-01", "El anexo de EMP-AUD-007 estuvo disponible para todos los responsables del área, aunque el expediente de evaluación solo requería acceso de la jefatura y Recursos Humanos."),
			("registro_auditoria", "2026-07-05", "La auditoría identificó información personal sensible no necesaria para evaluar el desempeño y un alcance de acceso más amplio que el requerido. El informe no reproduce el contenido sensible."),
		],
		"question": "¿Qué dos riesgos de manejo de información identificó la auditoría en el expediente de EMP-AUD-007?",
		"answer": "Se incluyó información personal sensible no necesaria y el anexo tuvo un alcance de acceso mayor al requerido.",
		"evidence_types": ["evaluacion", "registro_acceso", "registro_auditoria"],
		"question_type": "integracion_multifuente",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-008",
		"area": "Ventas",
		"scenario": "cambio_sin_trazabilidad",
		"documents": [
			("evaluacion", "2026-06-30", "La copia final de la evaluación de EMP-AUD-008 asigna 4.0/5 para 2026-S1."),
			("historial_versiones", "2026-06-30", "El repositorio conserva una copia anterior de la evaluación de EMP-AUD-008 con calificación 3.0/5 y una copia posterior con 4.0/5. No registra quién realizó el cambio ni su motivo."),
			("registro_auditoria", "2026-07-03", "La auditoría confirmó el cambio de 3.0/5 a 4.0/5, pero no encontró autor, justificación ni aprobación asociada. No puede determinar si la modificación fue autorizada."),
		],
		"question": "¿Qué puede confirmarse y qué permanece sin trazabilidad en el cambio de calificación de EMP-AUD-008?",
		"answer": "Se confirma un cambio de 3.0/5 a 4.0/5; no constan autor, motivo ni aprobación, por lo que su autorización no puede determinarse.",
		"evidence_types": ["evaluacion", "historial_versiones", "registro_auditoria"],
		"question_type": "integracion_multifuente",
		"difficulty": "alta",
	},
	{
		"employee_id": "EMP-AUD-009",
		"area": "Desarrollo",
		"scenario": "comparacion_no_sustentada",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-009 describe su resultado como superior al de todo el equipo."),
			("evidencia_desempeno", "2026-06-28", "El expediente contiene únicamente métricas individuales de EMP-AUD-009. No incluye resultados de otras personas ni criterios comunes de comparación."),
			("registro_auditoria", "2026-07-02", "La auditoría no encontró datos comparables del equipo ni un método de clasificación. La afirmación de superioridad global no puede verificarse con la información disponible."),
		],
		"question": "¿Está sustentada la afirmación de que EMP-AUD-009 superó a todo el equipo?",
		"answer": "No. Solo hay métricas individuales y no existe una base comparable ni un método de clasificación del equipo.",
		"evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "suficiencia_evidencia",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-010",
		"area": "Operaciones",
		"scenario": "objetivos_parcialmente_respaldados",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-010 afirma que cumplió los tres objetivos acordados en 2026-S1."),
			("evidencia_desempeno", "2026-06-27", "El expediente incluye evidencia verificable para los objetivos OBJ-01 y OBJ-02 de EMP-AUD-010. No se localizó evidencia para OBJ-03."),
			("registro_auditoria", "2026-07-03", "La auditoría vinculó dos de los tres objetivos con evidencia. El expediente no permite confirmar ni descartar el cumplimiento de OBJ-03."),
		],
		"question": "¿Qué parte de la afirmación de cumplimiento de EMP-AUD-010 puede verificarse?",
		"answer": "Puede verificarse el respaldo de OBJ-01 y OBJ-02; el cumplimiento de OBJ-03 queda sin determinar porque no se localizó evidencia.",
		"evidence_types": ["evaluacion", "evidencia_desempeno", "registro_auditoria"],
		"question_type": "suficiencia_evidencia",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-011",
		"area": "Administración",
		"scenario": "version_vigente_identificada",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación versión 1 de EMP-AUD-011 asigna 3.6/5. El documento está marcado como reemplazado."),
			("evaluacion_corregida", "2026-07-05", "La evaluación versión 2 de EMP-AUD-011 asigna 4.0/5, incluye motivo de corrección y firma de la jefatura. El documento está marcado como vigente."),
			("registro_auditoria", "2026-07-06", "La auditoría verificó que la versión 2 reemplaza a la versión 1, registra la corrección y tiene aprobación de la jefatura. La versión 1 se conserva para trazabilidad, no como resultado vigente."),
		],
		"question": "¿Qué calificación debe tratarse como vigente para EMP-AUD-011 y por qué?",
		"answer": "La calificación vigente es 4.0/5 de la versión 2, aprobada y documentada; la versión 1 de 3.6/5 se conserva como antecedente reemplazado.",
		"evidence_types": ["evaluacion_corregida", "registro_auditoria"],
		"question_type": "resolucion_versiones",
		"difficulty": "media",
	},
	{
		"employee_id": "EMP-AUD-012",
		"area": "Ventas",
		"scenario": "comparabilidad_limitada",
		"documents": [
			("evaluacion", "2026-06-30", "La evaluación de EMP-AUD-012 usa indicadores de captación comercial correspondientes a su rol."),
			("marco_criterios", "2026-01-15", "El marco de evaluación asigna indicadores diferentes según el rol. No define una escala común ni ponderaciones para comparar resultados entre áreas."),
			("registro_auditoria", "2026-07-04", "La auditoría concluyó que los datos permiten revisar el expediente de EMP-AUD-012 según sus criterios, pero no establecer una clasificación válida frente a empleados de otras áreas."),
		],
		"question": "¿Permite la información comparar globalmente la evaluación de EMP-AUD-012 con empleados de otras áreas?",
		"answer": "No. Los criterios dependen del rol y no existe una escala común ni ponderaciones para una comparación global válida.",
		"evidence_types": ["marco_criterios", "registro_auditoria"],
		"question_type": "limite_comparabilidad",
		"difficulty": "media",
	},
]


DOCUMENTS = []
QUESTIONS = []

for case in CASES:
	case_documents = []
	for document_type, document_date, content in case["documents"]:
		document_id = f"AUD-DOC-{len(DOCUMENTS) + 1:03d}"
		document = {
			"document_id": document_id,
			"employee_id": case["employee_id"],
			"evaluation_period": "2026-S1",
			"area": case["area"],
			"document_type": document_type,
			"document_date": document_date,
			"content": content,
			"source_metadata": {
				"document_type": document_type,
				"evaluation_period": "2026-S1",
				"synthetic": True,
				"source": "auditoria_desempeno_laboral",
				"audit_scenario": case["scenario"],
			},
		}
		DOCUMENTS.append(document)
		case_documents.append(document)

	evidence_types = set(case["evidence_types"])
	QUESTIONS.append(
		{
			"question_id": f"QAUD-{len(QUESTIONS) + 1:02d}",
			"employee_id": case["employee_id"],
			"question": case["question"],
			"reference_answer": case["answer"],
			"expected_evidence_ids": [
				document["document_id"]
				for document in case_documents
				if document["document_type"] in evidence_types
			],
			"expected_evidence_types": case["evidence_types"],
			**({
				"objective_count": case["objective_count"],
				"supported_count": case["supported_count"],
				"unsupported_count": case["unsupported_count"],
			} if "objective_count" in case else {}),
			"scenario": case["scenario"],
			"question_type": case["question_type"],
			"difficulty": case["difficulty"],
		}
	)

AUDIT_SCENARIOS = [
	"respaldo_completo",
	"evidencia_incompleta",
	"calificaciones_discrepantes",
	"evidencia_desactualizada",
	"metadatos_insuficientes",
	"error_de_calculo",
	"minimizacion_datos_personales",
	"cambio_sin_trazabilidad",
	"comparacion_no_sustentada",
	"objetivos_parcialmente_respaldados",
	"version_vigente_identificada",
	"comparabilidad_limitada",
]
AREAS = ["Desarrollo", "Operaciones", "Ventas", "Administración"]
BULK_EMPLOYEE_NUMBERS = tuple(range(13, 1001))
SCENARIO_ASSIGNMENT_SEED = 20261030
base_scenario_count, extra_scenario_count = divmod(len(BULK_EMPLOYEE_NUMBERS), len(AUDIT_SCENARIOS))
SCENARIO_DISTRIBUTION = {
	scenario: base_scenario_count + (index < extra_scenario_count)
	for index, scenario in enumerate(AUDIT_SCENARIOS)
}
scenario_pool = [
	scenario
	for scenario, count in SCENARIO_DISTRIBUTION.items()
	for _ in range(count)
]
random.Random(SCENARIO_ASSIGNMENT_SEED).shuffle(scenario_pool)
SCENARIO_BY_EMPLOYEE = dict(zip(BULK_EMPLOYEE_NUMBERS, scenario_pool))


def build_scenario_documents(employee_id, area, scenario, objective_count, supported_count, score, system_score):
	if scenario == "respaldo_completo":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} para 2026-S1 asigna {score:.1f}/5 y declara {objective_count} objetivos cerrados con evidencia verificable."),
			("evidencia_desempeno", "2026-06-28", f"El expediente de {employee_id} contiene {objective_count} registros de entrega con fechas, responsables y enlaces a los proyectos de 2026-S1."),
			("registro_auditoria", "2026-07-02", f"La auditoría revisó {employee_id} y corroboró que cada objetivo está vinculado a la evidencia, sin cambios no registrados ni inconsistencias."),
		]
	if scenario == "evidencia_incompleta":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} para 2026-S1 afirma cumplimiento de {objective_count} objetivos y asigna {score:.1f}/5."),
			("evidencia_desempeno", "2026-06-25", f"El expediente de {employee_id} contiene {supported_count} registros verificables; faltan {objective_count - supported_count} evidencias para completar la valoración."),
			("registro_auditoria", "2026-07-03", f"La auditoría verificó {supported_count} objetivos con respaldo documental y dejó pendientes {objective_count - supported_count} sin soporte."),
		]
	if scenario == "calificaciones_discrepantes":
		return [
			("evaluacion", "2026-06-30", f"La evaluación firmada de {employee_id} asigna {score:.1f}/5 para 2026-S1."),
			("registro_sistema", "2026-06-30", f"El sistema de {employee_id} conserva una calificación de {system_score:.1f}/5 en 2026-S1; no incluye nota de ajuste ni aprobación."),
			("registro_auditoria", "2026-07-02", f"La auditoría detectó dos calificaciones distintas para {employee_id} y no encontró conciliación ni aprobación del cambio."),
		]
	if scenario == "evidencia_desactualizada":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} cubre el periodo 2026-S1 y declara cumplimiento estable durante los seis meses."),
			("evidencia_desempeno", "2026-03-31", f"Los registros disponibles de {employee_id} finalizan el 2026-03-31; no hay evidencias de abril, mayo ni junio."),
			("registro_auditoria", "2026-07-04", f"La auditoría concluyó que la evidencia disponible no cubre el segundo trimestre de {employee_id}; el desempeño de abril-junio no puede verificarse."),
		]
	if scenario == "metadatos_insuficientes":
		return [
			("evaluacion", "2026-06-30", f"El formulario de {employee_id} asigna {score:.1f}/5 pero no identifica evaluador, firma ni versión del documento."),
			("evidencia_desempeno", "2026-06-20", f"Se adjuntan varios informes de {employee_id}, pero ninguno indica a qué criterio o objetivo corresponde."),
			("registro_auditoria", "2026-07-01", f"La auditoría no pudo reconstruir autoría, firma ni criterio asociado para {employee_id}; faltan metadatos esenciales."),
		]
	if scenario == "error_de_calculo":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} informa un promedio de 90 % en dos indicadores del periodo 2026-S1."),
			("evidencia_desempeno", "2026-06-28", f"Los indicadores registrados de {employee_id} son 8 de 10 y 6 de 10. La hoja de cálculo indica que el promedio simple es 70 %."),
			("registro_auditoria", "2026-07-02", f"La auditoría recalculó {employee_id}: (80 % + 60 %) / 2 = 70 %. El 90 % informado no coincide con los valores fuente ni con la fórmula declarada."),
		]
	if scenario == "minimizacion_datos_personales":
		return [
			("evaluacion", "2026-06-30", f"El anexo de evaluación de {employee_id} incorpora información personal sensible no relacionada con los criterios revisados."),
			("registro_acceso", "2026-07-01", f"El anexo de {employee_id} estuvo disponible para todos los responsables del área, aunque la evaluación solo requería acceso restringido."),
			("registro_auditoria", "2026-07-05", f"La auditoría detectó información personal sensible innecesaria y un alcance de acceso más amplio que el requerido para {employee_id}."),
		]
	if scenario == "cambio_sin_trazabilidad":
		return [
			("evaluacion", "2026-06-30", f"La copia final de la evaluación de {employee_id} asigna {score:.1f}/5 para 2026-S1."),
			("historial_versiones", "2026-06-30", f"El repositorio conserva una versión anterior de {employee_id} con {system_score:.1f}/5 y una posterior con {score:.1f}/5, pero no registra autor ni motivo del cambio."),
			("registro_auditoria", "2026-07-03", f"La auditoría confirmó el cambio de {system_score:.1f}/5 a {score:.1f}/5 en {employee_id}; no encontró autor, justificación ni aprobación asociada."),
		]
	if scenario == "objetivos_parcialmente_respaldados":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} afirma que cumplió {objective_count} objetivos acordados en 2026-S1."),
			("evidencia_desempeno", "2026-06-27", f"El expediente de {employee_id} contiene evidencia verificable para {supported_count} de {objective_count} objetivos; no se localizó respaldo para {objective_count - supported_count}."),
			("registro_auditoria", "2026-07-03", f"La auditoría vinculó {supported_count} de {objective_count} objetivos de {employee_id}; el cumplimiento de los {objective_count - supported_count} restantes no puede determinarse."),
		]
	if scenario == "version_vigente_identificada":
		return [
			("evaluacion", "2026-06-30", f"La evaluación versión 1 de {employee_id} asigna {system_score:.1f}/5 y está marcada como reemplazada."),
			("evaluacion_corregida", "2026-07-05", f"La evaluación versión 2 de {employee_id} asigna {score:.1f}/5, incluye motivo de corrección y firma de la jefatura. Está marcada como vigente."),
			("registro_auditoria", "2026-07-06", f"La auditoría verificó que la versión 2 de {employee_id} reemplaza a la versión 1, documenta la corrección y tiene aprobación de la jefatura."),
		]
	if scenario == "comparabilidad_limitada":
		return [
			("evaluacion", "2026-06-30", f"La evaluación de {employee_id} utiliza indicadores específicos para el rol y el área de {area}."),
			("marco_criterios", "2026-01-15", f"El marco asigna indicadores según el rol, pero no define una escala común ni ponderaciones para comparar resultados entre áreas."),
			("registro_auditoria", "2026-07-04", f"La auditoría permite revisar el expediente de {employee_id} según sus criterios, pero no establecer una clasificación válida frente a otras áreas."),
		]
	return [
		("evaluacion", "2026-06-30", f"La evaluación de {employee_id} describe un resultado superior al del equipo en 2026-S1."),
		("evidencia_desempeno", "2026-06-28", f"El expediente de {employee_id} contiene solo métricas individuales; no incluye resultados comparables del equipo ni criterios comunes."),
		("registro_auditoria", "2026-07-02", f"La auditoría concluyó que la afirmación de superioridad global de {employee_id} no puede verificarse con la información disponible."),
	]


for employee_number in BULK_EMPLOYEE_NUMBERS:
	employee_id = f"EMP-AUD-{employee_number:03d}"
	area = AREAS[(employee_number - 1) % len(AREAS)]
	scenario = SCENARIO_BY_EMPLOYEE[employee_number]
	objective_count = 4 + employee_number % 5
	supported_count = objective_count - 2
	score = 2.5 + (employee_number % 6) * 0.5
	system_score = score + 0.5 if score < 5.0 else score - 0.5

	for document_type, document_date, content in build_scenario_documents(
		employee_id,
		area,
		scenario,
		objective_count,
		supported_count,
		score,
		system_score,
	):
		DOCUMENTS.append(
			{
				"document_id": f"AUD-DOC-{len(DOCUMENTS) + 1:03d}",
				"employee_id": employee_id,
				"evaluation_period": "2026-S1",
				"area": area,
				"document_type": document_type,
				"document_date": document_date,
				"content": content,
				"source_metadata": {
					"document_type": document_type,
					"evaluation_period": "2026-S1",
					"synthetic": True,
					"source": "auditoria_desempeno_laboral",
					"audit_scenario": scenario,
					"objective_count": objective_count,
					"supported_count": supported_count,
					"unsupported_count": objective_count - supported_count,
				},
			}
		)


TARGET_QUESTION_COUNT = 100
SEMANTIC_VERIFICATION_NOTE = (
	"La validación semántica aplica comprobaciones de coherencia de la pregunta, la respuesta de referencia "
	"y la cobertura de evidencias por escenario; no sustituye una evaluación humana ni una comparación literal exacta."
)
SPANISH_NUMBERS = {
	0: "cero",
	1: "uno",
	2: "dos",
	3: "tres",
	4: "cuatro",
	5: "cinco",
	6: "seis",
	7: "siete",
	8: "ocho",
	9: "nueve",
	10: "diez",
	11: "once",
	12: "doce",
	13: "trece",
	14: "catorce",
	15: "quince",
	16: "dieciséis",
	17: "diecisiete",
	18: "dieciocho",
	19: "diecinueve",
	20: "veinte",
}


def validate_semantic_question(question, documents_by_id):
	question_text = question["question"].strip()
	answer_text = question["reference_answer"].strip()
	scenario = question["scenario"]

	assert question_text, f"Pregunta vacía: {question}"
	assert answer_text, f"Respuesta vacía: {question}"
	assert question_text.endswith("?"), f"La pregunta no es interrogativa: {question_text}"
	assert len(answer_text) >= 20, f"Respuesta demasiado corta para {question_text}"
	assert question["expected_evidence_ids"], f"Sin evidencias: {question_text}"

	related_docs = [documents_by_id[evidence_id] for evidence_id in question["expected_evidence_ids"]]
	evidence_text = "\n".join(doc["content"] for doc in related_docs)
	question_lc = question_text.lower()
	answer_lc = answer_text.lower()

	if scenario in {"respaldo_completo", "trazabilidad_completa"}:
		assert "evidencia" in question_lc or "verificar" in question_lc or "rastre" in question_lc
		assert any(token in answer_lc for token in ("coinciden", "verificar", "evidencia", "auditoría", "vincula"))
	elif scenario == "evidencia_incompleta":
		assert "objetivo" in question_lc or "respaldo" in question_lc or "evidencia" in question_lc
		objective_count = int(question.get("objective_count", 0))
		supported_count = int(question.get("supported_count", 0))
		unsupported_count = int(question.get("unsupported_count", max(objective_count - supported_count, 0)))
		assert objective_count > 0, f"El escenario evidencia incompleta requiere un total de objetivos: {question_text}"
		assert supported_count >= 0 and unsupported_count >= 0, f"Los números de evidencia deben ser no negativos: {question_text}"
		assert objective_count == supported_count + unsupported_count, (
			f"La suma no coincide: objective_count={objective_count}, supported_count={supported_count}, unsupported_count={unsupported_count}"
		)
		assert supported_count < objective_count, f"Debe haber objetivos sin respaldo documental: {question_text}"
		count_tokens = {
			str(supported_count),
			str(unsupported_count),
			SPANISH_NUMBERS.get(supported_count, ""),
			SPANISH_NUMBERS.get(unsupported_count, ""),
			"resto",
		}
		assert any(token in answer_lc for token in count_tokens if token), (
			f"No se detectó la cantidad de objetivos respaldados o pendientes en la respuesta: {answer_text}"
		)
		assert any(token in answer_lc for token in ("localizaron", "sin evidencia", "sin respaldo", "no cuentan", "resto"))
		assert any(token in answer_lc for token in ("objetivo", "evidencia", "respaldo"))
	elif scenario == "calificaciones_discrepantes":
		assert "calific" in question_lc or "discrep" in question_lc
		values = sorted({match.group(0) for match in re.finditer(r"\d+(?:\.\d+)?/5", evidence_text)})
		assert len(values) >= 2, f"No se detectaron dos calificaciones para {question_text}"
		assert any(value in answer_text for value in values)
		assert any(token in answer_lc for token in ("distinta", "discrep", "conciliación", "no coincide"))
		assert any(ch.isdigit() for ch in answer_text)
	elif scenario == "evidencia_desactualizada":
		assert "segundo trimestre" in question_lc or "abril" in question_lc or "junio" in question_lc
		assert "2026-03-31" in evidence_text or "marzo" in evidence_text.lower()
		assert any(token in answer_lc for token in ("no.", "termina", "marzo", "abril", "junio"))
		assert "2026-03-31" in evidence_text or "31 de marzo" in answer_text.lower()
	elif scenario == "metadatos_insuficientes":
		assert "datos" in question_lc or "faltan" in question_lc or "reconstruir" in question_lc
		assert any(token in answer_lc for token in ("faltan", "identidad", "firma", "versión", "criterio"))
	elif scenario == "error_de_calculo":
		assert "promedio" in question_lc or "indicador" in question_lc or "discrep" in question_lc

		source_match = re.search(
			r"valores fuente son\s*(\d+(?:\.\d+)?)\s*%\s*(?:y|e)\s*(\d+(?:\.\d+)?)\s*%",
			evidence_text,
			flags=re.IGNORECASE,
		)
		if not source_match:
			source_match = re.search(
				r"(\d+)\s*de\s*10\s*(?:y|e)\s*(\d+)\s*de\s*10",
				evidence_text,
				flags=re.IGNORECASE,
			)
		if not source_match:
			source_match = re.search(
				r"\((\d+(?:\.\d+)?)\s*%\s*\+\s*(\d+(?:\.\d+)?)\s*%\s*\)\s*/\s*2",
				evidence_text,
				flags=re.IGNORECASE,
			)
		assert source_match, f"No se encontraron indicadores fuente: {question_text}"

		if source_match.re.pattern.startswith("valores fuente son"):
			value_1, value_2 = map(float, source_match.groups())
		elif source_match.re.pattern.startswith("(\\d+)\\s*de\\s*10"):
			value_1, value_2 = (float(source_match.group(1)) * 10, float(source_match.group(2)) * 10)
		else:
			value_1, value_2 = map(float, source_match.groups())
		expected_average = (value_1 + value_2) / 2

		answer_match = re.search(
			r"promedio(?:\s+simple|\s+real)?(?:\s+de\s+[^;,.]+)?\s+es\s*(\d+(?:\.\d+)?)\s*%",
			answer_lc,
			flags=re.IGNORECASE,
		)
		assert answer_match, f"No se encontró el promedio en la respuesta: {answer_text}"

		reported_average = float(answer_match.group(1))
		assert abs(reported_average - expected_average) < 0.001, (
			f"Promedio incorrecto: esperado {expected_average}, respuesta {reported_average}"
		)
		assert "no coincide" in answer_lc or "90" in answer_text or "70" in answer_text
	elif scenario == "minimizacion_datos_personales":
		assert "riesgo" in question_lc or "información" in question_lc or "auditoría" in question_lc
		assert any(token in answer_lc for token in ("información personal", "acceso", "riesgo", "sensib"))
	elif scenario == "comparacion_no_sustentada":
		assert "compar" in question_lc or "equipo" in question_lc or "áreas" in question_lc
		assert any(token in answer_lc for token in ("no.", "no existe", "base comparable", "clasificación"))
	elif scenario in {"comparabilidad_limitada", "limite_comparabilidad"}:
		assert "compar" in question_lc or "áreas" in question_lc or "rol" in question_lc
		assert any(token in answer_lc for token in ("no", "criterios", "rol", "comparativa", "escala común"))
	elif scenario in {"cambio_sin_trazabilidad", "version_vigente_identificada"}:
		assert "cambio" in question_lc or "versión" in question_lc or "vigente" in question_lc
		assert any(token in answer_lc for token in ("cambio", "vigente", "versión", "aprobación", "autor"))
		if scenario == "version_vigente_identificada":
			assert "versión 2" in evidence_text.lower() or "vigente" in evidence_text.lower()
			assert "versión 2" in answer_lc or "vigente" in answer_lc
			assert "aprobada" in answer_lc or "aprobación" in answer_lc
	elif scenario == "objetivos_parcialmente_respaldados":
		assert "objetivo" in question_lc or "cumplimiento" in question_lc or "verific" in question_lc
		assert any(token in answer_lc for token in ("objetivo", "verificarse", "sin determinar", "evidencia"))
	else:
		assert "compar" in question_lc or "áreas" in question_lc or "datos" in question_lc
		assert any(token in answer_lc for token in ("no", "criterios", "rol", "comparativa", "datos"))

	return True


def build_validation_report(questions, documents_by_id):
	question_type_distribution = Counter(
		question.get("question_type", "sin_clasificar") for question in questions
	)
	difficulty_distribution = Counter(
		question.get("difficulty", "sin_clasificar") for question in questions
	)
	verified_questions = 0
	rejected_questions = 0
	questions_without_evidence = 0
	inconsistent_answers = 0
	incorrectly_associated_evidence = 0
	validation_issues = []

	for question in questions:
		issues = []
		evidence_ids = question.get("expected_evidence_ids", [])
		if not evidence_ids:
			questions_without_evidence += 1
			issues.append("sin_evidencias")

		related_docs = []
		for evidence_id in evidence_ids:
			document = documents_by_id.get(evidence_id)
			if document is None:
				issues.append(f"evidencia_inexistente:{evidence_id}")
			else:
				related_docs.append(document)

		expected_types = set(question.get("expected_evidence_types", []))
		actual_types = {document["document_type"] for document in related_docs}
		association_is_invalid = (
			len(evidence_ids) != len(set(evidence_ids))
			or not question.get("employee_id")
			or not expected_types
			or actual_types != expected_types
			or any(
				document["employee_id"] != question.get("employee_id")
				or document["source_metadata"].get("audit_scenario") != question.get("scenario")
				for document in related_docs
			)
		)
		if association_is_invalid:
			incorrectly_associated_evidence += 1
			issues.append("evidencias_incorrectamente_asociadas")

		try:
			validate_semantic_question(question, documents_by_id)
		except (AssertionError, KeyError, TypeError, ValueError) as error:
			inconsistent_answers += 1
			issues.append(f"inconsistencia_semantica:{error}")

		if issues:
			rejected_questions += 1
			validation_issues.append(
				{"question_id": question.get("question_id"), "issues": issues}
			)
		else:
			verified_questions += 1

	return {
		"total_questions": len(questions),
		"verified_questions": verified_questions,
		"rejected_questions": rejected_questions,
		"questions_without_evidence": questions_without_evidence,
		"inconsistent_answers": inconsistent_answers,
		"incorrectly_associated_evidence": incorrectly_associated_evidence,
		"questions_with_answers": sum(bool(question.get("reference_answer")) for question in questions),
		"questions_with_evidence": sum(bool(question.get("expected_evidence_ids")) for question in questions),
		"question_type_distribution": dict(sorted(question_type_distribution.items())),
		"difficulty_distribution": dict(sorted(difficulty_distribution.items())),
		"validation_issues": validation_issues,
	}


for employee_number in range(13, 101):
	employee_id = f"EMP-AUD-{employee_number:03d}"
	scenario = SCENARIO_BY_EMPLOYEE[employee_number]
	objective_count = 4 + employee_number % 5
	supported_count = objective_count - 2
	score = 2.5 + (employee_number % 6) * 0.5
	system_score = score + 0.5 if score < 5.0 else score - 0.5

	if scenario == "respaldo_completo":
		question = (
			f"¿Qué evidencia permite verificar el desempeño de {employee_id} "
			"durante 2026-S1?"
		)
		answer = (
			f"La evaluación, la evidencia de desempeño y la auditoría de {employee_id} "
			"se alinean con el mismo periodo y permiten verificar el cumplimiento."
		)
	elif scenario == "evidencia_incompleta":
		question = (
			f"¿Cuántos objetivos de {employee_id} tienen respaldo documental localizado?"
		)
		answer = (
			f"Se localizaron {supported_count} respaldos; el resto de los objetivos "
			"no cuentan con evidencia documental suficiente."
		)
	elif scenario == "calificaciones_discrepantes":
		question = (
			f"¿Qué discrepancia detecta la auditoría en la calificación de {employee_id}?"
		)
		answer = (
			f"La evaluación y el sistema muestran calificaciones distintas para {employee_id}; "
			f"se registran {score:.1f}/5 y {system_score:.1f}/5 sin conciliación."
		)
	elif scenario == "evidencia_desactualizada":
		question = (
			f"¿Puede verificarse el desempeño de {employee_id} en el segundo trimestre con "
			"el expediente disponible?"
		)
		answer = (
			f"No. La evidencia disponible de {employee_id} termina en marzo y no permite "
			"verificar abril-junio."
		)
	elif scenario == "metadatos_insuficientes":
		question = (
			f"¿Qué datos faltan para reconstruir la evaluación de {employee_id}?"
		)
		answer = (
			f"Faltan la identidad del evaluador, la firma, la versión y la vinculación "
			f"de la evidencia con los criterios de {employee_id}."
		)
	elif scenario == "error_de_calculo":
		question = (
			f"¿Cuál es el promedio real de los indicadores de {employee_id} y qué discrepancia "
			"encontró la auditoría?"
		)
		answer = (
			f"El promedio real de {employee_id} es 70 %; la evaluación informa 90 %, que no "
			"coincide con los valores fuente ni con la fórmula."
		)
	elif scenario == "minimizacion_datos_personales":
		question = (
			f"¿Qué riesgo de manejo de información detectó la auditoría en {employee_id}?"
		)
		answer = (
			f"Se detectó información personal sensible no necesaria y un alcance de acceso "
			f"más amplio que el requerido para {employee_id}."
		)
	elif scenario == "cambio_sin_trazabilidad":
		question = (
			f"¿Qué cambio de calificación se confirmó para {employee_id} y qué datos de trazabilidad faltan?"
		)
		answer = (
			f"La calificación cambió de {system_score:.1f}/5 a {score:.1f}/5; no constan autor, "
			"motivo ni aprobación del cambio."
		)
	elif scenario == "objetivos_parcialmente_respaldados":
		question = (
			f"¿Cuántos objetivos de {employee_id} tienen respaldo y cuántos quedan sin determinar?"
		)
		answer = (
			f"Hay respaldo para {supported_count} de {objective_count} objetivos; "
			f"los {objective_count - supported_count} restantes no pueden determinarse."
		)
	elif scenario == "version_vigente_identificada":
		question = (
			f"¿Qué versión y calificación deben considerarse vigentes para {employee_id}?"
		)
		answer = (
			f"La versión vigente es la versión 2, con {score:.1f}/5, aprobada y documentada; "
			"la versión 1 está reemplazada."
		)
	elif scenario == "comparabilidad_limitada":
		question = (
			f"¿Puede compararse globalmente la evaluación de {employee_id} con otras áreas?"
		)
		answer = (
			"No. Los criterios dependen del rol y no existe una escala común ni ponderaciones "
			"para una comparación global válida."
		)
	else:
		question = (
			f"¿Puede compararse globalmente la evaluación de {employee_id} con otras áreas?"
		)
		answer = (
			f"No. Los criterios de {employee_id} dependen del rol y no existe una base "
			"comparativa común para una clasificación global válida."
		)

	employee_docs = [
		doc for doc in DOCUMENTS if doc["employee_id"] == employee_id
	]
	QUESTIONS.append(
		{
			"question_id": f"QAUD-{len(QUESTIONS) + 1:02d}",
			"employee_id": employee_id,
			"question": question,
			"reference_answer": answer,
			"expected_evidence_ids": [
				doc["document_id"] for doc in employee_docs
			],
			"expected_evidence_types": [
				doc["document_type"] for doc in employee_docs
			],
			"objective_count": objective_count,
			"supported_count": supported_count,
			"unsupported_count": objective_count - supported_count,
			"scenario": scenario,
			"question_type": (
				"integracion_multifuente"
				if scenario in {"respaldo_completo", "metadatos_insuficientes", "minimizacion_datos_personales", "cambio_sin_trazabilidad", "version_vigente_identificada"}
				else "suficiencia_evidencia"
				if scenario in {"evidencia_incompleta", "evidencia_desactualizada", "comparacion_no_sustentada", "objetivos_parcialmente_respaldados"}
				else "deteccion_contradiccion"
				if scenario == "calificaciones_discrepantes"
				else "verificacion_calculo"
				if scenario == "error_de_calculo"
				else "limite_comparabilidad"
			),
			"difficulty": "media" if scenario in {"respaldo_completo", "metadatos_insuficientes", "calificaciones_discrepantes", "comparacion_no_sustentada", "cambio_sin_trazabilidad", "version_vigente_identificada"} else "baja" if scenario in {"evidencia_incompleta", "evidencia_desactualizada"} else "media",
		}
	)

assert len(CASES) == 12
assert len(DOCUMENTS) == 3000
assert len(QUESTIONS) == TARGET_QUESTION_COUNT
assert len({doc["document_id"] for doc in DOCUMENTS}) == len(DOCUMENTS)
assert len({doc["employee_id"] for doc in DOCUMENTS}) == 1000

documents_by_id = {doc["document_id"]: doc for doc in DOCUMENTS}
VALIDATION_REPORT = build_validation_report(QUESTIONS, documents_by_id)


raw_path = RAW / "corpus_auditoria_desempeno.jsonl"
processed_path = PROCESSED / "corpus_auditoria_desempeno.jsonl"
questions_path = EVALUATION / "questions_auditoria_desempeno.jsonl"
validation_report_path = EVALUATION / "validation_report_auditoria_desempeno.json"

with validation_report_path.open("w", encoding="utf-8") as report_file:
	json.dump(VALIDATION_REPORT, report_file, ensure_ascii=False, indent=2)

print("\nREPORTE DE VALIDACIÓN")
print(f"Preguntas totales: {VALIDATION_REPORT['total_questions']}")
print(f"Preguntas verificadas: {VALIDATION_REPORT['verified_questions']}")
print(f"Preguntas rechazadas: {VALIDATION_REPORT['rejected_questions']}")
print(f"Preguntas sin evidencias: {VALIDATION_REPORT['questions_without_evidence']}")
print(f"Respuestas inconsistentes: {VALIDATION_REPORT['inconsistent_answers']}")
print(f"Evidencias incorrectamente asociadas: {VALIDATION_REPORT['incorrectly_associated_evidence']}")

print("\nDISTRIBUCIÓN DE PREGUNTAS")
print("\nPor tipo de consulta:")
for question_type, count in VALIDATION_REPORT["question_type_distribution"].items():
	print(f"{question_type}: {count}")

print("\nPor nivel de dificultad:")
for difficulty, count in VALIDATION_REPORT["difficulty_distribution"].items():
	print(f"{difficulty}: {count}")

print("\nTOTAL:")
print(f"Preguntas: {VALIDATION_REPORT['total_questions']}")
print(f"Preguntas con respuesta: {VALIDATION_REPORT['questions_with_answers']}")
print(f"Preguntas con evidencias: {VALIDATION_REPORT['questions_with_evidence']}")
print(f"Reporte JSON: {validation_report_path}")

if VALIDATION_REPORT["rejected_questions"]:
	raise ValueError(
		f"Se rechazaron {VALIDATION_REPORT['rejected_questions']} preguntas; "
		f"consulta el reporte {validation_report_path}"
	)

for output_path, records in (
	(raw_path, DOCUMENTS),
	(processed_path, DOCUMENTS),
	(questions_path, QUESTIONS),
):
	with output_path.open("w", encoding="utf-8") as output_file:
		for record in records:
			output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


print("CORPUS DE AUDITORÍA DE DESEMPEÑO GENERADO")
print(f"Documentos: {len(DOCUMENTS)}")
print(f"Expedientes: {len({doc['employee_id'] for doc in DOCUMENTS})}")
print(f"Preguntas: {len(QUESTIONS)}")
print("VALIDACIÓN SEMÁNTICA: comprobación de coherencia por escenario y evidencia.")
print(SEMANTIC_VERIFICATION_NOTE)
print(f"RAW: {raw_path}")
print(f"PROCESSED: {processed_path}")
print(f"EVALUATION: {questions_path}")
