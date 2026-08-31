from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def validate_config(config: dict) -> None:
    required_sections = [
        "experiment",
        "model",
        "embeddings",
        "retrieval",
        "pipeline",
        "generation",        
        "evaluation",
        "paths",
    ]

    missing = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing:
        raise ValueError(
            f"Missing configuration sections: {', '.join(missing)}"
        )

    if config["retrieval"]["top_k"] <= 0:
        raise ValueError("retrieval.top_k must be greater than zero.")

    if config["pipeline"]["max_iterations"] <= 0:
        raise ValueError(
            "pipeline.max_iterations must be greater than zero."
        )

    temperature = config["model"]["temperature"]

    if not 0.0 <= temperature <= 2.0:
        raise ValueError(
            "model.temperature must be between 0.0 and 2.0."
        )


def create_experiment_id(name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{name}"


def save_environment(output_dir: Path) -> None:
    environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }

    path = output_dir / "environment.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2)


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    validate_config(config)

    experiment_name = config["experiment"]["name"]
    experiment_id = create_experiment_id(experiment_name)

    output_root = Path(config["paths"]["outputs"])
    output_dir = output_root / experiment_id

    output_dir.mkdir(parents=True, exist_ok=False)

    config["experiment"]["id"] = experiment_id

    saved_config = output_dir / "config.yaml"

    with saved_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    save_environment(output_dir)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible experiment run."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the experiment configuration.",
    )

    args = parser.parse_args()

    output_dir = run(args.config)

    print("Experiment created successfully.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()