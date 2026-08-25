"""Focused tests for Example 21 Gate 4 controls."""

from __future__ import annotations

import time

import pytest

from examples.pp_prop.example21_gate4 import (
    BackendProbe,
    benchmark_decoder,
    measure_probe,
    select_backend,
    validate_temporary_proof,
)


def test_select_backend_uses_lower_valid_median_and_cpu_tie() -> None:
    cpu = BackendProbe("cpu", (4.0, 2.0, 3.0), True, b"p")
    gpu = BackendProbe("gpu", (3.0, 3.0, 3.0), True, b"p")
    assert select_backend(cpu, gpu) == "cpu"
    assert select_backend(BackendProbe("cpu", (3.0, 3.0, 3.0), True, b"p"), gpu) == "cpu"


def test_select_backend_ignores_invalid_probe() -> None:
    assert select_backend(BackendProbe("cpu", (1.0,), False, b""), BackendProbe("gpu", (2.0,), True, b"p")) == "gpu"
    with pytest.raises(RuntimeError, match="no valid"):
        select_backend(BackendProbe("cpu", (), False, b""), BackendProbe("gpu", (), False, b""))


def test_measure_probe_warms_and_records_three_calls() -> None:
    calls = []
    probe = measure_probe(lambda: calls.append(1), backend="cpu", prediction_bytes=b"p", finite=True)
    assert len(calls) == 4
    assert len(probe.times_ms) == 3


def test_decoder_records_five_calls_per_request() -> None:
    result = benchmark_decoder(lambda request: request, [1, 2])
    assert result["calls_per_request"] == 5
    assert len(result["requests"]) == 2
    assert all(len(record["calls_ms"]) == 5 for record in result["requests"])


def test_decoder_rejects_invalid_count_and_slow_call() -> None:
    with pytest.raises(ValueError, match="positive"):
        benchmark_decoder(lambda value: value, [1], calls_per_request=0)

    def slow(value):
        time.sleep(0.002)
        return value

    with pytest.raises(RuntimeError, match="exceeded"):
        benchmark_decoder(slow, [1], limit_ms=0.1)


def test_temporary_proof_enforces_isolation_and_direct_evidence() -> None:
    evidence = validate_temporary_proof(
        training_task="d631b094",
        validation_task="46f33fce",
        update_tasks=["d631b094"] * 8,
        validation_state_before=b"same",
        validation_state_after=b"same",
        prediction_before=b"before",
        prediction_after=b"after",
        interventions={name: {"changed": name != "null"} for name in ("voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null")},
        elapsed_seconds=1.0,
    )
    assert evidence["update_count"] == 8
    assert evidence["prediction_changed"] is True


@pytest.mark.parametrize("field", ["tasks", "state", "prediction", "interventions", "time"])
def test_temporary_proof_rejects_gate_failures(field: str) -> None:
    kwargs = dict(
        training_task="d631b094",
        validation_task="46f33fce",
        update_tasks=["d631b094"] * 8,
        validation_state_before=b"same",
        validation_state_after=b"same",
        prediction_before=b"before",
        prediction_after=b"after",
        interventions={name: True for name in ("voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null")},
        elapsed_seconds=1.0,
    )
    if field == "tasks":
        kwargs["update_tasks"] = ["46f33fce"] * 8
    elif field == "state":
        kwargs["validation_state_after"] = b"changed"
    elif field == "prediction":
        kwargs["prediction_after"] = b"before"
    elif field == "interventions":
        kwargs["interventions"].pop("null")
    else:
        kwargs["elapsed_seconds"] = 181.0
    with pytest.raises(RuntimeError):
        validate_temporary_proof(**kwargs)
