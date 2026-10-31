from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.rag_baseline import RAGBaseline as _RAGBaseline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default_desempeno.yaml"


def load_domain_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the performance-audit domain configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")

    return config


class RAGBaseline(_RAGBaseline):
    """Thin domain-specific wrapper around the generic baseline RAG implementation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_domain_config()
        super().__init__(config)


__all__ = ["RAGBaseline", "load_domain_config"]
