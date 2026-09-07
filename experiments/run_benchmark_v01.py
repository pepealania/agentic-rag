from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config
from src.rag_baseline import RAGBaseline

from agentic_rag.chunking import chunk_documents
from agentic_rag.embeddings import EmbeddingModel
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.ingestion import load_documents
from agentic_rag.retrieval import VectorStore
from agentic_rag.schemas import Question
from agentic_rag.workflow import AgenticRAGWorkflow


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = Path("configs/default.yaml")
OUTPUT_DIR = Path("outputs/benchmark_v0.1")

# Repetitions required by Sprint 2.
# One complete run for every question plus a small repeated sample.
REPEAT_QUESTION_IDS = ["Q05", "Q10", "Q15", "Q20", "Q24"]


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any):
    """Serialize project objects safely."""
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                data,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_questions(path: Path) -> list[dict]:
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            questions.append(json.loads(line))

    return questions


def ensure_clean_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in [
        "rag_results.jsonl",
        "agentic_results.jsonl",
        "execution_log.csv",
        "error_log.csv",
    ]:
        path = OUTPUT_DIR / filename

        if path.exists():
            path.unlink()


# ============================================================
# LOGGING
# ============================================================

EXECUTION_FIELDS = [
    "run_id",
    "question_id",
    "scenario",
    "configuration",
    "replicate",
    "started_at",
    "finished_at",
    "latency_seconds",
    "retrieved_count",
    "retrieved_document_ids",
    "retrieved_chunk_ids",
    "retrieved_scores",
    "query_used",
    "answer",
    "citations",
    "json_valid",
    "citation_validity",
    "retry_count",
    "step_count",
    "decision_log",
    "tool_results",
    "validation_errors",
    "error",
]


ERROR_FIELDS = [
    "run_id",
    "question_id",
    "scenario",
    "configuration",
    "replicate",
    "error_stage",
    "error_type",
    "error_message",
]


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                field: row.get(field, "")
                for field in fields
            }
        )


def log_error(
    *,
    run_id: str,
    question: dict,
    configuration: str,
    replicate: int,
    stage: str,
    error: Exception | str,
) -> None:

    append_csv(
        OUTPUT_DIR / "error_log.csv",
        ERROR_FIELDS,
        {
            "run_id": run_id,
            "question_id": question["question_id"],
            "scenario": question.get("scenario", ""),
            "configuration": configuration,
            "replicate": replicate,
            "error_stage": stage,
            "error_type": (
                type(error).__name__
                if isinstance(error, Exception)
                else "RecordedError"
            ),
            "error_message": str(error),
        },
    )


# ============================================================
# BASELINE RAG
# ============================================================

def build_baseline(config: dict) -> RAGBaseline:
    rag = RAGBaseline(config)

    rag.load_documents(
        config["paths"]["raw_data"]
    )

    rag.build_chunks()
    rag.build_index()

    return rag


def run_baseline_question(
    rag: RAGBaseline,
    question: dict,
    run_id: str,
    replicate: int,
) -> dict:

    started_at = utc_now()
    start = time.perf_counter()

    error = None
    result = {}

    try:

        result = rag.run(
            question["question_id"],
            question["question"],
        )

    except Exception as exc:

        error = str(exc)

        log_error(
            run_id=run_id,
            question=question,
            configuration="rag",
            replicate=replicate,
            stage="generation",
            error=exc,
        )

    latency = time.perf_counter() - start
    finished_at = utc_now()

    retrieved = result.get(
        "retrieved_documents",
        [],
    )

    row = {
        "run_id": run_id,
        "question_id": question["question_id"],
        "scenario": question.get("scenario", ""),
        "configuration": "rag",
        "replicate": replicate,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_seconds": latency,
        "retrieved_count": len(retrieved),
        "retrieved_document_ids": json.dumps(
            [
                x.get("document_id")
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "retrieved_chunk_ids": json.dumps(
            [
                x.get("chunk_id")
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "retrieved_scores": json.dumps(
            [
                x.get("score")
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "query_used": question["question"],
        "answer": result.get("answer", ""),
        "citations": "",
        "json_valid": result.get("json_valid", False),
        "citation_validity": result.get(
            "citation_validity",
            0.0,
        ),
        "retry_count": 0,
        "step_count": 1,
        "decision_log": json.dumps(
            [
                {
                    "agent": "baseline",
                    "decision": "retrieve_then_generate",
                }
            ],
            ensure_ascii=False,
        ),
        "tool_results": "",
        "validation_errors": "",
        "error": error or result.get("error"),
    }

    return row


# ============================================================
# AGENTIC RAG
# ============================================================

def build_agentic(config: dict) -> AgenticRAGWorkflow:
    data_dir = Path(
        config["paths"]["raw_data"]
    )

    documents = load_documents(
        data_dir
    )

    chunks = chunk_documents(
        documents=documents,
        chunk_size=config["retrieval"]["chunk_size"],
        chunk_overlap=config["retrieval"]["chunk_overlap"],
    )

    embedding_model = EmbeddingModel(
        config["embeddings"]["name"]
    )

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.build(
        chunks
    )

    evidence_agent = EvidenceAgent(
        vector_store=vector_store,
        top_k=config["retrieval"]["top_k"],
    )

    workflow = AgenticRAGWorkflow(
        evidence_agent=evidence_agent,
        max_steps=config["pipeline"].get(
            "max_steps",
            8,
        ),
    )

    return workflow


def run_agentic_question(
    workflow: AgenticRAGWorkflow,
    question: dict,
    run_id: str,
    replicate: int,
) -> dict:

    started_at = utc_now()
    start = time.perf_counter()

    error = None
    state = {}

    question_model = Question(
        question_id=question["question_id"],
        question=question["question"],
    )

    initial_state = {
        "question": question_model,
        "retry_count": 0,
        "step_count": 0,
        "decision_log": [],
        "tool_results": [],
        "validation_errors": [],
    }

    try:

        state = workflow.run(
            initial_state
        )

    except Exception as exc:

        error = str(exc)

        log_error(
            run_id=run_id,
            question=question,
            configuration="agentic_rag",
            replicate=replicate,
            stage="decision",
            error=exc,
        )

    latency = time.perf_counter() - start
    finished_at = utc_now()

    retrieved = state.get(
        "retrieved_chunks",
        [],
    )

    answer = state.get(
        "answer"
    )

    if answer is not None:

        answer_dict = (
            answer.model_dump()
            if hasattr(answer, "model_dump")
            else dict(answer)
        )

    else:

        answer_dict = {}

    citations = answer_dict.get(
        "citations",
        [],
    )

    row = {
        "run_id": run_id,
        "question_id": question["question_id"],
        "scenario": question.get("scenario", ""),
        "configuration": "agentic_rag",
        "replicate": replicate,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_seconds": latency,
        "retrieved_count": len(retrieved),
        "retrieved_document_ids": json.dumps(
            [
                x.document_id
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "retrieved_chunk_ids": json.dumps(
            [
                x.chunk_id
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "retrieved_scores": json.dumps(
            [
                x.score
                for x in retrieved
            ],
            ensure_ascii=False,
        ),
        "query_used": state.get(
            "query_used",
            question["question"],
        ),
        "answer": answer_dict.get(
            "answer",
            "",
        ),
        "citations": json.dumps(
            citations,
            ensure_ascii=False,
            default=json_default,
        ),
        "json_valid": answer is not None,
        "citation_validity": calculate_agentic_citation_validity(
            citations,
            retrieved,
        ),
        "retry_count": state.get(
            "retry_count",
            0,
        ),
        "step_count": state.get(
            "step_count",
            0,
        ),
        "decision_log": json.dumps(
            state.get(
                "decision_log",
                [],
            ),
            ensure_ascii=False,
        ),
        "tool_results": json.dumps(
            state.get(
                "tool_results",
                [],
            ),
            ensure_ascii=False,
        ),
        "validation_errors": json.dumps(
            state.get(
                "validation_errors",
                [],
            ),
            ensure_ascii=False,
        ),
        "error": error,
    }

    return row


def calculate_agentic_citation_validity(
    citations: list,
    retrieved: list,
) -> float:

    if not citations:
        return 0.0

    valid_pairs = {
        (
            chunk.document_id,
            chunk.chunk_id,
        )
        for chunk in retrieved
    }

    valid = 0

    for citation in citations:

        if hasattr(citation, "document_id"):

            pair = (
                citation.document_id,
                citation.chunk_id,
            )

        else:

            pair = (
                citation.get("document_id"),
                citation.get("chunk_id"),
            )

        if pair in valid_pairs:
            valid += 1

    return valid / len(citations)


# ============================================================
# BENCHMARK
# ============================================================

def main() -> None:

    print("=" * 70)
    print("SPRINT 2 — BENCHMARK v0.1")
    print("=" * 70)

    config = load_config(
        CONFIG_PATH
    )

    questions_path = Path(
        config["evaluation"]["questions_path"]
    )

    questions = load_questions(
        questions_path
    )

    if not questions:
        raise RuntimeError(
            "No evaluation questions found."
        )

    print(
        f"Preguntas cargadas: {len(questions)}"
    )

    print(
        "Preguntas de referencia:",
        ", ".join(
            q["question_id"]
            for q in questions
        ),
    )

    ensure_clean_output_directory()

    # --------------------------------------------------------
    # Freeze benchmark metadata
    # --------------------------------------------------------

    manifest = {
        "benchmark": "v0.1",
        "created_at": utc_now(),
        "question_count": len(questions),
        "question_ids": [
            q["question_id"]
            for q in questions
        ],
        "question_file": str(
            questions_path
        ),
        "question_file_sha256": sha256_file(
            questions_path
        ),
        "repeat_sample": REPEAT_QUESTION_IDS,
        "config": config,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    write_json(
        OUTPUT_DIR / "benchmark_manifest.json",
        manifest,
    )

    # --------------------------------------------------------
    # Build systems once.
    # --------------------------------------------------------

    print("\n[1/2] Construyendo RAG convencional...")

    rag = build_baseline(
        config
    )

    print(
        f"  Documentos: {len(rag.documents)}"
    )

    print(
        f"  Chunks: {len(rag.chunks)}"
    )

    print(
        "\n[2/2] Construyendo Agentic RAG..."
    )

    agentic = build_agentic(
        config
    )

    print("  Workflow listo.")

    # --------------------------------------------------------
    # Determine executions.
    # --------------------------------------------------------

    execution_plan = []

    for question in questions:

        execution_plan.append(
            (
                question,
                1,
            )
        )

        if question["question_id"] in REPEAT_QUESTION_IDS:

            execution_plan.append(
                (
                    question,
                    2,
                )
            )

    total_executions = (
        len(execution_plan) * 2
    )

    current_execution = 0

    print(
        f"\nTotal ejecuciones: {total_executions}"
    )

    # --------------------------------------------------------
    # Run benchmark.
    # --------------------------------------------------------

    for question, replicate in execution_plan:

        question_id = question[
            "question_id"
        ]

        # -------------------------
        # RAG
        # -------------------------

        current_execution += 1

        run_id = (
            f"benchmark_v01_"
            f"rag_"
            f"{question_id}_"
            f"r{replicate}"
        )

        print(
            f"\n[{current_execution}/{total_executions}] "
            f"RAG | {question_id} | "
            f"replicate={replicate}"
        )

        result = run_baseline_question(
            rag=rag,
            question=question,
            run_id=run_id,
            replicate=replicate,
        )

        append_jsonl(
            OUTPUT_DIR / "rag_results.jsonl",
            result,
        )

        append_csv(
            OUTPUT_DIR / "execution_log.csv",
            EXECUTION_FIELDS,
            result,
        )

        print(
            f"  latency={result['latency_seconds']:.3f}s "
            f"retrieved={result['retrieved_count']} "
            f"citation_validity={result['citation_validity']}"
        )

        # -------------------------
        # Agentic RAG
        # -------------------------

        current_execution += 1

        run_id = (
            f"benchmark_v01_"
            f"agentic_rag_"
            f"{question_id}_"
            f"r{replicate}"
        )

        print(
            f"\n[{current_execution}/{total_executions}] "
            f"Agentic RAG | {question_id} | "
            f"replicate={replicate}"
        )

        result = run_agentic_question(
            workflow=agentic,
            question=question,
            run_id=run_id,
            replicate=replicate,
        )

        append_jsonl(
            OUTPUT_DIR / "agentic_results.jsonl",
            result,
        )

        append_csv(
            OUTPUT_DIR / "execution_log.csv",
            EXECUTION_FIELDS,
            result,
        )

        print(
            f"  latency={result['latency_seconds']:.3f}s "
            f"retrieved={result['retrieved_count']} "
            f"steps={result['step_count']} "
            f"retries={result['retry_count']} "
            f"citation_validity={result['citation_validity']}"
        )

    # --------------------------------------------------------
    # Completion manifest
    # --------------------------------------------------------

    completed = {
        **manifest,
        "completed_at": utc_now(),
        "status": "completed",
        "total_question_runs": len(
            execution_plan
        ),
        "total_configuration_runs": total_executions,
        "rag_runs": len(
            execution_plan
        ),
        "agentic_rag_runs": len(
            execution_plan
        ),
        "required_sprint_2_artifacts": [
            "benchmark_manifest.json",
            "rag_results.jsonl",
            "agentic_results.jsonl",
            "execution_log.csv",
            "error_log.csv",
        ],
    }

    write_json(
        OUTPUT_DIR / "benchmark_manifest.json",
        completed,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK v0.1 FINALIZADO")
    print("=" * 70)

    print(
        f"Preguntas: {len(questions)}"
    )

    print(
        f"Ejecuciones por configuración: "
        f"{len(execution_plan)}"
    )

    print(
        f"RAG: "
        f"{OUTPUT_DIR / 'rag_results.jsonl'}"
    )

    print(
        f"Agentic RAG: "
        f"{OUTPUT_DIR / 'agentic_results.jsonl'}"
    )

    print(
        f"Execution log: "
        f"{OUTPUT_DIR / 'execution_log.csv'}"
    )

    print(
        f"Error log: "
        f"{OUTPUT_DIR / 'error_log.csv'}"
    )


if __name__ == "__main__":
    main()
