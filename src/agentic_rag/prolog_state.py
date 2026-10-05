from __future__ import annotations

from typing import Any

from agentic_rag.state import AgentState


class PrologAgentState(AgentState, total=False):
    """State extensions owned only by the Prolog workflow."""

    raw_data: list[dict[str, Any]]
    facts_loaded: bool
    prolog_results: dict[str, list[str]]