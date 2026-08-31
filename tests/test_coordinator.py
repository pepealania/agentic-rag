from agentic_rag.coordinator import Coordinator


def test_coordinator_retries_once_after_validation_failure():
    coordinator = Coordinator(max_steps=8)

    state = {
        "step_count": 3,
        "retry_count": 0,
        "validation_errors": ["citations_missing"],
        "decision_log": [],
    }

    route = coordinator.route_after_validation(state)

    assert route == "retry"
    assert state["retry_count"] == 1
    assert state["decision_log"][-1]["decision"] == "retry"


def test_coordinator_finalizes_after_retry_is_exhausted():
    coordinator = Coordinator(max_steps=8)

    state = {
        "step_count": 6,
        "retry_count": 1,
        "validation_errors": ["citations_missing"],
        "decision_log": [],
    }

    route = coordinator.route_after_validation(state)

    assert route == "finalize"
    assert state["retry_count"] == 1
    assert state["decision_log"][-1]["decision"] == "finalize"


def test_coordinator_finalizes_when_validation_passes():
    coordinator = Coordinator(max_steps=8)

    state = {
        "step_count": 3,
        "retry_count": 0,
        "validation_errors": [],
        "decision_log": [],
    }

    route = coordinator.route_after_validation(state)

    assert route == "finalize"
    assert state["decision_log"][-1]["decision"] == "finalize"
