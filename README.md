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

## Running an experiment

The executable implementation is located under:

```text
src/agentic_rag/
```

From the repository root, with the virtual environment activated, run:

```powershell
python -m agentic_rag.run_experiment
```

A successful execution produces an experiment identifier and a dedicated output directory.

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

Individual dependencies should not need to be installed manually.

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
3. Agentic RAG implementation.
4. Experimental evaluation.
5. Comparison and analysis.

The baseline RAG is implemented independently from the agentic workflow so that the contribution of the agentic components can be evaluated against a controlled reference.

## Current status

The repository currently provides:

- reproducible project structure;
- centralized experiment configuration;
- versioned dependencies;
- reproducible experiment execution;
- unique experiment identifiers;
- saved experiment configurations;
- execution environment information.

The next implementation stage is the baseline RAG pipeline, including:

- document ingestion;
- persistent document metadata;
- document chunking;
- embeddings;
- vector indexing;
- top-k retrieval;
- context-constrained generation;
- document and chunk citations;
- retrieval scores;
- execution traces;
- evaluation questions and metrics.