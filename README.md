# Agentic RAG

Reproducible research repository for the implementation and evaluation of an Agentic Retrieval-Augmented Generation (RAG) system.

The repository is organized to support controlled experiments, reproducible execution, and comparison between a baseline RAG pipeline and an agentic RAG pipeline.

## Repository structure

```text
agentic-rag/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── agentic_rag/
├── configs/
│   └── default.yaml
├── experiments/
├── tests/
├── outputs/
├── docs/
├── pyproject.toml
├── README.md
└── .gitignore
```

## Requirements

- Python 3.12
- Git

Python 3.12 is the required Python version for this project.

## Installation

### Windows

#### 1. Clone the repository

```powershell
git clone https://github.com/pepealania/agentic-rag.git
cd agentic-rag
```

#### 2. Create a Python 3.12 virtual environment

```powershell
py -3.12 -m venv .venv
```

#### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should display:

```text
(.venv)
```

#### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

#### 5. Install the project

```powershell
pip install -e .
```

The project and its dependencies are installed according to the versions declared in `pyproject.toml`.

To install the full runner and test dependencies listed in `requirements.txt`
(including `pandas` and `pytest`), run:

```powershell
pip install -r requirements.txt
```

### Linux / macOS

#### 1. Clone the repository

```bash
git clone https://github.com/pepealania/agentic-rag.git
cd agentic-rag
```

#### 2. Create a Python 3.12 virtual environment

```bash
python3.12 -m venv .venv
```

#### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

#### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

#### 5. Install the project

```bash
pip install -e .
```

For the full runner and test dependencies, also install `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration

Experiment parameters are centralized in:

```text
configs/default.yaml
```

The configuration defines the parameters required to reproduce an experiment, including:

- experiment name;
- random seed;
- LLM provider and model;
- embedding provider and model;
- generation temperature;
- maximum number of generated tokens;
- retrieval `top_k`;
- pipeline iteration limit;
- evaluation dataset path;
- data paths;
- experiment output paths.

Example:

```yaml
experiment:
  name: "baseline"
  seed: 42

model:
  provider: "openai"
  name: "REPLACE_WITH_MODEL"
  temperature: 0.0
  max_tokens: 1024

embeddings:
  provider: "openai"
  name: "REPLACE_WITH_EMBEDDING_MODEL"

retrieval:
  top_k: 5

pipeline:
  max_iterations: 1

evaluation:
  questions_path: "data/evaluation/questions.jsonl"

paths:
  data: "data"
  raw_data: "data/raw"
  processed_data: "data/processed"
  outputs: "outputs"
  experiments: "experiments"
  logs: "outputs/logs"
```

Parameters should be modified through the configuration file rather than being duplicated throughout the source code.

## Performance-audit corpus

The repository includes a synthetic corpus for auditing information used in
employee performance evaluations. It contains 3,000 documents covering 1,000
employee-period records and 12 evaluation questions. The scenarios cover
traceability, evidence coverage, score conflicts, temporal coverage,
calculation checks, privacy, version changes, and comparability.

Generate or regenerate the corpus from the repository root:

```powershell
python experiments\generate_corpus_dominio.py
```

The generator writes:

- `data/raw/auditoria_desempeno/corpus_auditoria_desempeno.jsonl`
- `data/processed/auditoria_desempeno/corpus_auditoria_desempeno.jsonl`
- `data/evaluation/questions_auditoria_desempeno.jsonl`

Run the classic RAG baseline for this corpus from the repository root:

```powershell
python experiments\run_rag_baseline_dominio.py
```

The runner reads `configs/default_desempeno.yaml` and writes
`results.json` and `results.csv` under `outputs/rag_baseline_dominio/`. Set
`OPENAI_API_KEY` in the shell before running. The FAISS index is built in
memory for each run; it is not currently persisted to the configured
`index_path`.

### Audit-rule prototype

`src/agentic_rag/deterministic_workflow_prolog_rules.py` contains a Prolog rule
catalog for the corpus scenarios. Its `evaluate_corpus_documents()` method
uses the generator-provided `source_metadata.audit_scenario` label. This is
useful for evaluating the rule mapping against controlled cases, but it is not
independent inference from document text; using that label as input would leak
the expected scenario in an inference evaluation. The engine is not yet wired
into `DeterministicAgenticRAGWorkflow`.

Running the rules requires SWI-Prolog and PySwip. The rule rationale, source
mapping, and methodological limits are documented in
[`docs/fundamento_reglas_auditoria_desempeno.md`](docs/fundamento_reglas_auditoria_desempeno.md).

## Prolog deterministic workflow

The Prolog-backed workflow reuses the deterministic retrieve, analyze, validate, and finalize graph. In Colab, install SWI-Prolog and the optional Python binding:

```python
!apt-get update -qq && apt-get install -y -qq swi-prolog
!pip install -q -e ".[prolog]"
```

Set `OPENAI_API_KEY` in the Colab environment before running the default `AnalystAgent`. Then provide employee/task records in `raw_data`:

```python
from agentic_rag.prolog_workflow import PrologDeterministicAgenticRAGWorkflow
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.schemas import Question

vector_evidence_agent = EvidenceAgent(
  vector_store=faiss_vector_store,  # índice FAISS construido desde tus chunks
  top_k=5,
)
workflow = PrologDeterministicAgenticRAGWorkflow(
  vector_evidence_agent=vector_evidence_agent,
  max_retries=1,
)
result = workflow.run({
  "question": Question(
    question_id="q1",
    question="¿Quién está sobrecargado o requiere asistencia?",
  ),
  "raw_data": [
    {"tipo": "empleado", "nombre": "Ana", "rol": "analista"},
    {"tipo": "tarea", "id": "t1", "empleado": "Ana", "estado": "pendiente"},
    {"tipo": "tarea", "id": "t2", "empleado": "Ana", "estado": "pendiente"},
    {"tipo": "tarea", "id": "t3", "empleado": "Ana", "estado": "pendiente"},
    {"tipo": "tarea", "id": "t4", "empleado": "Ana", "estado": "pendiente"},
    {"tipo": "tarea", "id": "t5", "empleado": "Ana", "estado": "retrasada"},
  ],
})

print(result["answer"].answer)
print(result["prolog_results"])
print(result["validation_passed"])
```

By default, Prolog marks an employee as overloaded with more than three pending tasks and as requiring assistance when at least one task is delayed. The Prolog deductions are passed to the analyst as a citable evidence chunk.

## Running and recording experiments

`agentic_rag.run_experiment` validates a YAML file and creates a directory with
the configuration and environment metadata. **It does not run retrieval,
indexing, or answer generation.** From the repository root, run:

```powershell
python -m agentic_rag.run_experiment
```

A successful invocation produces an experiment identifier and a dedicated
output directory.

Example:

```text
Experiment created successfully.
Output directory: outputs\20260821_205426_baseline
```

The generated directory contains:

```text
outputs/
└── 20260821_205426_baseline/
    ├── config.yaml
    └── environment.json
```

### Experiment identifier

Each execution receives a unique identifier based on the execution timestamp and experiment name.

Example:

```text
20260821_205426_baseline
```

The identifier is used to associate the artifacts produced by that execution.

### Saved configuration

The `config.yaml` file inside the experiment directory contains the configuration used for that specific execution.

This allows an experiment to be traced back to the parameters that produced its results.

### Execution environment

The `environment.json` file records information about the Python interpreter and operating system used during execution.

This provides additional information for reproducing the experiment environment.

### RAG execution entry points

Run the classic baseline with `configs/default.yaml`:

```powershell
python -m experiments.run_rag_baseline
```

This runner loads JSONL files recursively from `paths.raw_data`, reads the
questions from `evaluation.questions_path`, builds a FAISS index in memory,
and writes `results.json` and `results.csv` to `outputs/rag_baseline/`. The
index and chunks are not persisted, even though the YAML has `index_path` and
`chunks_path` fields.

Run the performance-audit baseline with `configs/default_desempeno.yaml`:

```powershell
python -m experiments.run_rag_baseline_dominio
```

This runner uses the domain corpus and questions configured there and writes
results to `outputs/rag_baseline_dominio/`. It reuses the classic baseline
algorithm; the domain-specific difference is its data, questions, and
configuration.

The v0.1 benchmark runs the classic baseline and `AgenticRAGWorkflow`:

```powershell
python -m experiments.run_benchmark_v01
```

It currently uses `configs/default.yaml` and writes benchmark artifacts to
`outputs/benchmark_v0.1/`. Its Agentic RAG ingestion path accepts `.txt` and
`.md` files through `agentic_rag.ingestion`; it does **not** currently ingest
the performance-audit JSONL corpus. The classic baseline loader accepts JSONL.

### Agentic workflow variants

- `AgenticRAGWorkflow` uses the fixed retrieve → analyze → validate path with
  rule-based retry/finalize routing. The analyst is LLM-backed; the coordinator
  does not dynamically plan tools.
- `DeterministicAgenticRAGWorkflow` uses the same predefined graph and exposes
  a configurable retry budget. It is implemented, but the current v0.1
  benchmark does not instantiate it.
- `AdaptiveAgenticRAGWorkflow` selects retrieve, analyze, validate, retry, or
  finalize actions from state using `AdaptiveCoordinator` thresholds and
  validation results. This is state-dependent control, not an LLM planner. It
  is implemented, but the current benchmark does not instantiate it.

The deterministic Prolog audit-rule prototype is separate from those runners;
it is not yet wired into `DeterministicAgenticRAGWorkflow`. Its domain
assumptions and the scenario-label leakage caveat are documented in
[`docs/fundamento_reglas_auditoria_desempeno.md`](docs/fundamento_reglas_auditoria_desempeno.md).

## Reproducibility workflow

The complete workflow is:

```text
Clone repository
       |
       v
Create Python 3.12 virtual environment
       |
       v
Activate virtual environment
       |
       v
pip install -e .
       |
       v
Configure configs/default.yaml
       |
       v
python -m agentic_rag.run_experiment
       |
       v
outputs/<experiment_id>/
       |
       +── config.yaml
       |
       └── environment.json
```

The executable project logic is implemented in `src/`.

Notebook execution is not required to run the project.

Notebooks under `experiments/` may be used for exploratory analysis or documentation, but they are not the only implementation of the system.

## Dependency management

Project dependencies are declared in:

```text
pyproject.toml
```

Dependency versions are explicitly specified to provide a consistent execution environment.

The project targets:

```text
Python 3.12
```

Install the project and its dependencies with:

```powershell
pip install -e .
```

`requirements.txt` also includes runner and test dependencies such as `pandas`
and `pytest`.

## Secrets and temporary files

Secrets must never be committed to the repository.

The `.gitignore` file excludes local and generated files including:

- `.env` files;
- virtual environments;
- Python caches;
- test caches;
- Jupyter checkpoints;
- generated experiment outputs;
- local raw data;
- local processed data;
- build artifacts;
- IDE files;
- operating-system temporary files.

Generated experiment outputs are stored under:

```text
outputs/
```

and are excluded from version control.

Local datasets are stored under:

```text
data/raw/
data/processed/
```

and are excluded from version control.

The synthetic audit corpus has an explicit exception in `.gitignore` so it can
be versioned. Check `.gitignore` before adding other local datasets.

## Development

Source code:

```text
src/agentic_rag/
```

Tests:

```text
tests/
```

Configuration:

```text
configs/
```

Experiments:

```text
experiments/
```

Documentation:

```text
docs/
```

The executable implementation should remain independent of notebook execution.

## Research workflow

The project is developed in the following stages:

1. Reproducible repository setup.
2. Baseline RAG implementation.
3. Minimum Agentic RAG implementation.
4. Experimental evaluation.
5. Comparison and analysis.

The baseline RAG is implemented independently from the agentic workflow so that the contribution of the agentic components can be evaluated against a controlled reference.

## Current status

### 3.1 Reproducible repository setup — ✅ Complete

The repository provides:

- reproducible Python 3.12 environment;
- centralized experiment configuration;
- versioned project dependencies;
- controlled random seed and generation temperature;
- unique experiment identifiers;
- saved experiment configurations;
- execution environment information;
- separation between source code, experiments, tests, data, and generated outputs.

### 3.2 Baseline RAG — 🟢 Implemented

The baseline RAG pipeline is implemented and provides the reference system for the Agentic RAG experiments.

The current implementation includes:

- document ingestion;
- document and chunk metadata;
- configurable chunking;
- configurable embeddings;
- vector indexing;
- top-k retrieval;
- context-constrained answer generation;
- document and chunk citations;
- retrieval scores;
- structured outputs;
- evidence/citation validation;
- baseline experiment execution;
- JSONL loading that requires `document_id` and `content` in each record.

The FAISS index and chunks are held in memory for the run; `index_path` and
`chunks_path` are not currently used to persist them. The general runner is
`experiments/run_rag_baseline.py`; the performance-audit runner reuses the
baseline engine with a domain-specific configuration and inputs.

The main baseline implementation is located in:

```text
src/rag_baseline.py
```
## 3.3 Deterministic Agentic RAG — 🟡 Implemented

`DeterministicAgenticRAGWorkflow` is a LangGraph workflow with a predefined
retrieve → analyze → validate path and bounded, rule-based retry/finalize
routing. It uses an `EvidenceAgent` and an LLM-backed `AnalystAgent`, but its
coordinator does not plan tools or actions dynamically. It is a controlled
workflow variant, not an autonomous planner.

### Current implementation

The current implementation includes:

- bounded retrieval, analysis, and validation;
- configurable maximum steps and retry budget;
- explicit, rule-based control flow;
- document and chunk citations;
- retrieval scores;
- structured intermediate and final outputs;
- evidence/citation validation;
- experiment-level artifact tracking.

The current `run_benchmark_v01.py` does not instantiate this class; it
instantiates `AgenticRAGWorkflow`. There is not yet a deterministic workflow
benchmark runner for the performance-audit corpus.

### Implementation

The main implementation is located in:

```text
src/agentic_rag/
```

## 3.4 Adaptive Agentic RAG — 🟡 Implemented

`AdaptiveAgenticRAGWorkflow` uses `AdaptiveCoordinator` to choose the next
predefined action from current state, evidence counts, validation results, and
step/retry budgets. The policy is state-dependent, but it is implemented as
code rules rather than an LLM planner selecting arbitrary tools.

### Possible actions

Available actions include:

- accepting the current evidence and generating a final answer;
- performing an additional retrieval step;
- retrying retrieval;
- retrying analysis;
- finalizing after validation or when a configured budget is exhausted.

### Current implementation

The implementation includes:

- iterative retrieval;
- adaptive control decisions;
- evidence sufficiency thresholds;
- configurable step and retry limits;
- state-dependent routing;
- document and chunk citations;
- retrieval scores;
- structured intermediate and final outputs;
- evidence/citation validation;
- experiment-level artifact tracking.

### Implementation

The main implementation is located in:

```text
src/agentic_rag/
```

No current experiment runner instantiates `AdaptiveAgenticRAGWorkflow`; the
v0.1 benchmark uses `AgenticRAGWorkflow` instead.
