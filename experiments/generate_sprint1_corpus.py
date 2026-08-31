import json
import random
from pathlib import Path
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "evaluation"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

AREAS = [
    ("AREA-01", "Desarrollo"),
    ("AREA-02", "Operaciones"),
    ("AREA-03", "Ventas"),
    ("AREA-04", "Administración"),
]

PERIODS = [
    (f"M{i:02d}", date(2026, i, 1), date(2026, i, 28 if i == 2 else 30))
    for i in range(1, 7)
]

EMPLOYEES = [
    {
        "employee_id": f"EMP-{i:03d}",
        "area_id": AREAS[(i - 1) % 4][0],
        "area_name": AREAS[(i - 1) % 4][1],
        "project_id": f"PROJ-{((i - 1) % 6) + 1:03d}",
        "project_name": f"Proyecto {((i - 1) % 6) + 1:03d}",
    }
    for i in range(1, 25)
]

DOCUMENTS = []
QUESTIONS = []

doc_counter = 1
task_counter = 1

def add_doc(
    employee,
    period_id,
    document_type,
    content,
    task_id=None,
    document_date=None,
    extra=None,
):
    global doc_counter

    doc = {
        "document_id": f"DOC-{doc_counter:03d}",
        "employee_id": employee["employee_id"],
        "project_id": employee["project_id"],
        "task_id": task_id,
        "period_id": period_id,
        "document_type": document_type,
        "document_date": str(document_date or date(2026, 1, 15)),
        "content": content,
        "source_metadata": {
            "document_type": document_type,
            "period_id": period_id,
            "synthetic": True,
            "source": "sprint1_corpus",
        },
    }

    if extra:
        doc["source_metadata"].update(extra)

    DOCUMENTS.append(doc)
    doc_counter += 1
    return doc


# -------------------------------------------------------------------
# 1. Base corpus: 24 employees × (6 tasks + 3 reports + 2 feedback +
#    1 evaluation) = 288 documents
# -------------------------------------------------------------------

employee_tasks = {}

for emp in EMPLOYEES:
    eid = emp["employee_id"]
    tasks = []

    for task_index in range(1, 7):
        tid = f"TASK-{task_counter:03d}"
        task_counter += 1

        period_id = f"M{task_index:02d}"

        task = {
            "task_id": tid,
            "employee_id": eid,
            "project_id": emp["project_id"],
            "period_id": period_id,
            "status": "completed",
            "due_date": f"2026-{task_index:02d}-20",
            "completion_date": f"2026-{task_index:02d}-18",
        }

        tasks.append(task)

        add_doc(
            emp,
            period_id,
            "task",
            (
                f"{eid} tuvo asignada la tarea {tid} en {emp['project_id']} "
                f"durante {period_id}. La tarea fue completada dentro del "
                f"periodo documentado. Fecha límite: {task['due_date']}. "
                f"Fecha de finalización: {task['completion_date']}."
            ),
            task_id=tid,
            document_date=date(2026, task_index, 18),
        )

    employee_tasks[eid] = tasks

    # 3 progress reports
    for report_index, period_id in enumerate(["M02", "M04", "M06"], start=1):
        add_doc(
            emp,
            period_id,
            "progress_report",
            (
                f"Reporte de avance de {eid} correspondiente a {period_id}. "
                f"El empleado mantiene seguimiento de las tareas asignadas "
                f"en {emp['project_id']}. El avance documentado es consistente "
                f"con las tareas registradas para el periodo."
            ),
            document_date=date(2026, int(period_id[1:]), 22),
        )

    # 2 feedback
    for period_id in ["M03", "M05"]:
        add_doc(
            emp,
            period_id,
            "feedback",
            (
                f"Retroalimentación de {eid} correspondiente a {period_id}. "
                f"Se registra seguimiento del trabajo realizado y observaciones "
                f"sobre las tareas del proyecto {emp['project_id']}."
            ),
            document_date=date(2026, int(period_id[1:]), 25),
        )

    # 1 evaluation
    add_doc(
        emp,
        "M06",
        "evaluation",
        (
            f"Evaluación periódica de {eid}. El documento resume la evidencia "
            f"registrada durante los seis meses y describe el comportamiento "
            f"documentado del empleado en {emp['project_id']}."
        ),
        document_date=date(2026, 6, 28),
    )


# -------------------------------------------------------------------
# 2. Controlled scenarios
# -------------------------------------------------------------------

def find_docs(employee_id, document_type=None, period_id=None, task_id=None):
    return [
        d for d in DOCUMENTS
        if d["employee_id"] == employee_id
        and (document_type is None or d["document_type"] == document_type)
        and (period_id is None or d["period_id"] == period_id)
        and (task_id is None or d["task_id"] == task_id)
    ]


# Scenario: sustained compliance EMP-001 ... EMP-005
for i in range(1, 6):
    eid = f"EMP-{i:03d}"

    for d in DOCUMENTS:
        if d["employee_id"] == eid:
            d["content"] += (
                f" La evidencia disponible para {eid} indica cumplimiento "
                f"consistente durante el periodo documentado."
            )


# Scenario: overload/delay EMP-006 ... EMP-010
for i in range(6, 11):
    eid = f"EMP-{i:03d}"
    tasks = employee_tasks[eid]

    # Make first two tasks delayed.
    for task in tasks[:2]:
        task["status"] = "delayed"
        task["completion_date"] = None

        docs = find_docs(eid, task_id=task["task_id"])

        for d in docs:
            d["content"] = (
                f"{eid} tuvo la tarea {task['task_id']} asignada durante "
                f"{task['period_id']}. La tarea presentó retraso respecto "
                f"a la fecha prevista y no se documentó finalización dentro "
                f"del periodo."
            )

    for d in DOCUMENTS:
        if d["employee_id"] == eid and d["document_type"] == "progress_report":
            d["content"] += (
                " Durante M03 y M04 se registró un aumento de tareas "
                "simultáneas. El incremento de carga coincidió con retrasos "
                "en algunas tareas."
            )

    for d in DOCUMENTS:
        if d["employee_id"] == eid and d["document_type"] == "feedback":
            d["content"] += (
                " La retroalimentación registra que algunos retrasos "
                "coincidieron con una carga de trabajo elevada."
            )


# EMP-009: one delay with documented overload and another without causal evidence
eid = "EMP-009"
tasks = employee_tasks[eid]

for d in DOCUMENTS:
    if d["employee_id"] == eid and d["document_type"] == "feedback":
        d["content"] += (
            " Para una de las tareas retrasadas existe evidencia de carga "
            "elevada; para otra tarea retrasada no se documenta una causa."
        )


# EMP-010: overload does NOT explain every delay
eid = "EMP-010"
for d in DOCUMENTS:
    if d["employee_id"] == eid and d["document_type"] == "feedback":
        d["content"] += (
            " La sobrecarga explica algunos retrasos documentados, pero "
            "no existe evidencia suficiente para atribuir todos los retrasos "
            "a dicha causa."
        )


# Scenario: contradictions EMP-011 ... EMP-015
contradiction_pairs = {
    "EMP-011": 3,
    "EMP-012": 3,
    "EMP-013": 4,
    "EMP-014": 5,
    "EMP-015": 5,
}

for eid, month in contradiction_pairs.items():
    tasks = employee_tasks[eid]
    task = tasks[(month - 1) % 6]

    reports = find_docs(
        eid,
        document_type="progress_report"
    )

    feedback = find_docs(
        eid,
        document_type="feedback"
    )

    if reports:
        reports[0]["content"] += (
            f" Para {task['task_id']}, el reporte registra 100 % de avance "
            f"y cumplimiento de la tarea."
        )

    if feedback:
        feedback[0]["content"] += (
            f" Para {task['task_id']}, la retroalimentación registra "
            f"avance incompleto e incumplimiento parcial."
        )


# EMP-013 evaluation contradiction
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-013" and d["document_type"] == "evaluation":
        d["content"] += (
            " La evaluación periódica presenta una valoración diferente "
            "respecto al reporte de avance, por lo que existe evidencia "
            "contradictoria."
        )


# EMP-014: recent evidence in M06
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-014":
        if d["period_id"] == "M06":
            d["content"] += (
                " Esta evidencia corresponde al periodo final y describe "
                "el estado más reciente documentado de la tarea relacionada."
            )


# EMP-015: explicitly unresolved contradiction
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-015":
        d["content"] += (
            " Las fuentes disponibles no permiten determinar concluyentemente "
            "el cumplimiento de la tarea debido a evidencia contradictoria."
        )


# Scenario: missing evidence EMP-016 ... EMP-019
for i in range(16, 20):
    eid = f"EMP-{i:03d}"

    for d in DOCUMENTS:
        if d["employee_id"] == eid:
            d["content"] += (
                " El corpus no contiene información adicional fuera de "
                "los hechos explícitamente documentados en este conjunto."
            )


# EMP-016: no peer excellence data
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-016":
        d["content"] += (
            " No se registran evaluaciones de compañeros ni porcentajes "
            "de calificaciones otorgadas por pares."
        )


# EMP-017: delay but no personal cause
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-017":
        d["content"] += (
            " Se documentan retrasos, pero no se registra una causa personal "
            "que permita explicar dichos retrasos."
        )


# EMP-018: no intention to leave
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-018":
        d["content"] += (
            " No existe documentación sobre intención de abandonar el proyecto."
        )


# EMP-019: no global comparison
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-019":
        d["content"] += (
            " La documentación disponible se limita al empleado y proyecto "
            "correspondiente y no permite establecer una comparación global "
            "con todos los empleados."
        )


# Scenario: temporal change EMP-020 ... EMP-024
for i in range(20, 25):
    eid = f"EMP-{i:03d}"

    for d in DOCUMENTS:
        if d["employee_id"] == eid:
            month = int(d["period_id"][1:])

            if month <= 2:
                d["content"] += (
                    " Durante los primeros meses se registraron algunos "
                    "retrasos y un cumplimiento menos estable."
                )
            elif month >= 4:
                d["content"] += (
                    " Durante la segunda mitad del periodo se observa "
                    "mayor estabilidad y reducción de retrasos."
                )


# EMP-020 explicit improvement
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-020":
        d["content"] += (
            " El patrón documentado muestra mejora desde M04, con cumplimiento "
            "más estable hacia M05 y M06."
        )


# EMP-021 decrease in delays
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-021":
        d["content"] += (
            " Los retrasos son más frecuentes en M01-M03 y disminuyen "
            "en M04-M06."
        )


# EMP-022 progressive improvement
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-022":
        d["content"] += (
            " La evidencia registra una mejora progresiva hacia M05-M06."
        )


# EMP-023 change from M04
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-023":
        month = int(d["period_id"][1:])
        if month >= 4:
            d["content"] += (
                " A partir de M04 se observa un cambio hacia menor cantidad "
                "de retrasos y mayor cumplimiento."
            )


# EMP-024 final improvement
for d in DOCUMENTS:
    if d["employee_id"] == "EMP-024":
        if d["period_id"] in ("M05", "M06"):
            d["content"] += (
                " La evidencia de M05-M06 muestra mayor cumplimiento y "
                "menos retrasos que la evidencia de M01-M02."
            )


# -------------------------------------------------------------------
# 3. Build questions with evidence IDs
# -------------------------------------------------------------------

def ids_for(eid, types=None, periods=None):
    return [
        d["document_id"]
        for d in DOCUMENTS
        if d["employee_id"] == eid
        and (types is None or d["document_type"] in types)
        and (periods is None or d["period_id"] in periods)
    ]


def task_doc_ids(eid, task_numbers):
    tasks = employee_tasks[eid]
    wanted = {
        tasks[n - 1]["task_id"]
        for n in task_numbers
    }

    return [
        d["document_id"]
        for d in DOCUMENTS
        if d["employee_id"] == eid
        and d["task_id"] in wanted
    ]


def q(qid, question, answer, evidence, scenario):
    QUESTIONS.append({
        "question_id": qid,
        "question": question,
        "reference_answer": answer,
        "expected_evidence_ids": evidence,
        "scenario": scenario,
    })


# Q01
q(
    "Q01",
    "¿Qué tareas asignadas a EMP-001 fueron completadas durante los seis meses?",
    "EMP-001 completó las tareas TASK-001 a TASK-006.",
    task_doc_ids("EMP-001", [1, 2, 3, 4, 5, 6]),
    "cumplimiento_sostenido",
)

q(
    "Q02",
    "¿En qué meses EMP-002 mantuvo cumplimiento de las tareas asignadas?",
    "EMP-002 mantuvo cumplimiento en M01, M02, M03, M04, M05 y M06.",
    ids_for("EMP-002"),
    "cumplimiento_sostenido",
)

q(
    "Q03",
    "¿Qué evidencia documental respalda el cumplimiento sostenido de EMP-003?",
    "Los reportes de avance y la evaluación periódica registran cumplimiento sostenido.",
    ids_for("EMP-003", ["progress_report", "evaluation"]),
    "cumplimiento_sostenido",
)

q(
    "Q04",
    "¿Qué proyecto muestra cumplimiento consistente por parte de EMP-004 durante el periodo?",
    "PROJ-004 muestra cumplimiento consistente de las tareas asignadas a EMP-004.",
    ids_for("EMP-004"),
    "cumplimiento_sostenido",
)

q(
    "Q05",
    "¿Existe evidencia suficiente para afirmar que EMP-005 mantuvo un cumplimiento sostenido durante los seis meses?",
    "Sí, la evidencia disponible permite sostener el cumplimiento durante el periodo.",
    ids_for("EMP-005"),
    "cumplimiento_sostenido",
)

q(
    "Q06",
    "¿Qué tareas de EMP-006 presentaron retrasos durante el periodo?",
    "Las dos primeras tareas de EMP-006 presentaron retrasos durante el periodo.",
    task_doc_ids("EMP-006", [1, 2]),
    "retraso_sobrecarga",
)

q(
    "Q07",
    "¿Qué evidencia indica que los retrasos de EMP-007 estuvieron relacionados con una mayor carga de trabajo?",
    "Los reportes registran aumento de tareas simultáneas coincidente con los retrasos.",
    ids_for("EMP-007", ["progress_report", "feedback"]),
    "retraso_sobrecarga",
)

q(
    "Q08",
    "¿Cómo cambió la carga de trabajo de EMP-008 durante los meses en que aumentaron sus retrasos?",
    "La carga aumentó durante el periodo de mayor retraso y coincidió con un incremento de retrasos.",
    ids_for("EMP-008"),
    "retraso_sobrecarga",
)

q(
    "Q09",
    "¿Qué documentos permiten diferenciar un retraso asociado a sobrecarga de un retraso sin causa documentada en EMP-009?",
    "Un retraso tiene evidencia de aumento de carga; otro no tiene una causa documentada.",
    ids_for("EMP-009"),
    "retraso_sobrecarga",
)

q(
    "Q10",
    "¿Puede afirmarse que la sobrecarga explica todos los retrasos de EMP-010?",
    "No. La sobrecarga está documentada solo para algunos retrasos.",
    ids_for("EMP-010"),
    "retraso_sobrecarga",
)

q(
    "Q11",
    "¿Existen documentos contradictorios sobre el cumplimiento de EMP-011 en M03?",
    "Sí. Un reporte registra cumplimiento y una retroalimentación registra incumplimiento parcial.",
    ids_for("EMP-011", ["progress_report", "feedback"]),
    "evidencia_contradictoria",
)

q(
    "Q12",
    "¿Qué documentos presentan versiones diferentes sobre el avance de una tarea de EMP-012?",
    "El reporte indica 100 % de avance y la retroalimentación registra avance incompleto.",
    ids_for("EMP-012", ["progress_report", "feedback"]),
    "evidencia_contradictoria",
)

q(
    "Q13",
    "¿Cómo debe describirse el cumplimiento de EMP-013 cuando el reporte y la evaluación difieren?",
    "Como evidencia contradictoria, sin afirmar cumplimiento o incumplimiento total.",
    ids_for("EMP-013", ["progress_report", "evaluation", "feedback"]),
    "evidencia_contradictoria",
)

q(
    "Q14",
    "¿Qué evidencia tiene mayor respaldo temporal para explicar el estado final de EMP-014?",
    "La evidencia más reciente de M06 describe el estado final, señalando contradicciones previas si existen.",
    ids_for("EMP-014", periods=["M06"]),
    "evidencia_contradictoria",
)

q(
    "Q15",
    "¿Puede determinarse concluyentemente si EMP-015 cumplió una tarea cuando existen evidencias contradictorias?",
    "No. Debe informarse la contradicción y citar ambas fuentes.",
    ids_for("EMP-015", ["progress_report", "feedback", "evaluation"]),
    "evidencia_contradictoria",
)

q(
    "Q16",
    "¿Qué porcentaje de las tareas de EMP-016 fueron consideradas excelentes por sus compañeros?",
    "No respondible: no existe evidencia para calcular ese porcentaje.",
    [],
    "ausencia_evidencia",
)

q(
    "Q17",
    "¿Cuál fue la causa personal del retraso de EMP-017 durante M04?",
    "No respondible: el retraso está documentado, pero no su causa personal.",
    ids_for("EMP-017"),
    "ausencia_evidencia",
)

q(
    "Q18",
    "¿Qué documento demuestra que EMP-018 tenía intención de abandonar el proyecto?",
    "Ningún documento lo demuestra; la pregunta es no respondible.",
    [],
    "ausencia_evidencia",
)

q(
    "Q19",
    "¿Puede determinarse si EMP-019 tuvo un desempeño superior al de todos los demás empleados?",
    "No. No existe evidencia comparativa suficiente.",
    ids_for("EMP-019"),
    "ausencia_evidencia",
)

q(
    "Q20",
    "¿Cómo evolucionó el cumplimiento de tareas de EMP-020 entre M01 y M06?",
    "Mejoró: presentó retrasos iniciales y cumplimiento más estable desde M04.",
    ids_for("EMP-020"),
    "cambio_temporal",
)

q(
    "Q21",
    "¿Qué cambios se observan en los retrasos de EMP-021 entre la primera y segunda mitad del periodo?",
    "Los retrasos disminuyeron durante M04-M06 respecto de M01-M03.",
    ids_for("EMP-021"),
    "cambio_temporal",
)

q(
    "Q22",
    "¿Qué evidencia indica una mejora o deterioro en los indicadores documentados de EMP-022?",
    "La evidencia indica una mejora progresiva hacia M05-M06.",
    ids_for("EMP-022"),
    "cambio_temporal",
)

q(
    "Q23",
    "¿En qué momento cambió el comportamiento documentado de EMP-023 y qué documentos lo sustentan?",
    "El cambio se observa a partir de M04, con menos retrasos y mayor cumplimiento.",
    ids_for("EMP-023"),
    "cambio_temporal",
)

q(
    "Q24",
    "¿La evidencia permite afirmar que el desempeño documentado de EMP-024 mejoró hacia el final del periodo?",
    "Sí. M05-M06 muestran mayor cumplimiento y menos retrasos que M01-M02.",
    ids_for("EMP-024"),
    "cambio_temporal",
)


# -------------------------------------------------------------------
# 4. Validation
# -------------------------------------------------------------------

assert len(DOCUMENTS) == 288, len(DOCUMENTS)
assert len(QUESTIONS) == 24, len(QUESTIONS)

assert len({
    d["employee_id"]
    for d in DOCUMENTS
}) == 24

assert len({
    d["period_id"]
    for d in DOCUMENTS
}) == 6

for eid in [f"EMP-{i:03d}" for i in range(1, 25)]:
    docs = [
        d for d in DOCUMENTS
        if d["employee_id"] == eid
    ]

    assert len(docs) == 12, (eid, len(docs))

    counts = {}
    for d in docs:
        counts[d["document_type"]] = counts.get(
            d["document_type"], 0
        ) + 1

    assert counts["task"] == 6
    assert counts["progress_report"] == 3
    assert counts["feedback"] == 2
    assert counts["evaluation"] == 1

document_ids = {
    d["document_id"]
    for d in DOCUMENTS
}

for question in QUESTIONS:
    for evidence_id in question["expected_evidence_ids"]:
        assert evidence_id in document_ids, (
            question["question_id"],
            evidence_id,
        )

# Contradiction checks
for eid in ["EMP-011", "EMP-012", "EMP-013", "EMP-014", "EMP-015"]:
    text = " ".join(
        d["content"]
        for d in DOCUMENTS
        if d["employee_id"] == eid
    )

    assert "100 % de avance" in text or "evidencia contradictoria" in text
    assert "avance incompleto" in text or "evidencia contradictoria" in text


# -------------------------------------------------------------------
# 5. Save
# -------------------------------------------------------------------

raw_path = RAW / "corpus_sprint1.jsonl"
processed_path = PROCESSED / "corpus_sprint1.jsonl"
questions_path = EVAL / "questions.jsonl"

with raw_path.open("w", encoding="utf-8") as f:
    for doc in DOCUMENTS:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

with processed_path.open("w", encoding="utf-8") as f:
    for doc in DOCUMENTS:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

with questions_path.open("w", encoding="utf-8") as f:
    for question in QUESTIONS:
        f.write(json.dumps(question, ensure_ascii=False) + "\n")


print("=" * 60)
print("SPRINT 1 CORPUS GENERATED")
print("=" * 60)
print(f"Documents : {len(DOCUMENTS)}")
print(f"Employees : 24")
print(f"Periods   : 6")
print(f"Questions : {len(QUESTIONS)}")
print()
print(f"RAW       : {raw_path}")
print(f"PROCESSED : {processed_path}")
print(f"EVALUATION: {questions_path}")
print()
print("Validation: PASSED")
print("=" * 60)
