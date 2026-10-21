import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
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
			"question": case["question"],
			"reference_answer": case["answer"],
			"expected_evidence_ids": [
				document["document_id"]
				for document in case_documents
				if document["document_type"] in evidence_types
			],
			"scenario": case["scenario"],
			"question_type": case["question_type"],
			"difficulty": case["difficulty"],
		}
	)


assert len(CASES) == 12
assert len(DOCUMENTS) == 36
assert len(QUESTIONS) == 12
assert len({doc["document_id"] for doc in DOCUMENTS}) == len(DOCUMENTS)

document_ids = {doc["document_id"] for doc in DOCUMENTS}
for question in QUESTIONS:
	assert question["expected_evidence_ids"]
	assert set(question["expected_evidence_ids"]) <= document_ids


raw_path = RAW / "corpus_auditoria_desempeno.jsonl"
processed_path = PROCESSED / "corpus_auditoria_desempeno.jsonl"
questions_path = EVALUATION / "questions_auditoria_desempeno.jsonl"

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
print(f"Expedientes: {len(CASES)}")
print(f"Preguntas: {len(QUESTIONS)}")
print(f"RAW: {raw_path}")
print(f"PROCESSED: {processed_path}")
print(f"EVALUATION: {questions_path}")
