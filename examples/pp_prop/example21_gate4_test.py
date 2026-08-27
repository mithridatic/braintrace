"""Focused tests for Example 21 Gate 4 controls."""

from __future__ import annotations

import time
import json
import subprocess

import pytest

from examples.pp_prop.example21_gate4 import (
    BackendProbe,
    benchmark_decoder,
    measure_probe,
    select_backend,
    validate_backend_probes,
    validate_temporary_proof,
    backend_command,
    run_gate4,
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


def test_backend_validation_requires_stable_prediction_and_finite_timing() -> None:
    cpu = BackendProbe("cpu", (1.0, 2.0, 3.0), True, b"same")
    gpu = BackendProbe("gpu", (float("inf"),), True, b"same")
    evidence = validate_backend_probes(cpu, gpu)
    assert evidence["selected_backend"] == "cpu"
    assert evidence["prediction_bytes_stable"] is True
    with pytest.raises(RuntimeError, match="different predictions"):
        validate_backend_probes(cpu, BackendProbe("gpu", (1.0,), True, b"other"))


def test_backend_validation_rejects_nan_probe() -> None:
    assert not BackendProbe("cpu", (float("nan"),), True, b"p").valid


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
        targets=[[[1]]],
        loss_components={"pre": {"shape": 1.0, "rows": 2.0}, "post": {"shape": 0.5, "rows": 1.0}},
        recurrent_weight_movement=1.0,
        decoded_predictions={"pre": [[1]], "post": [[2]]},
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
        targets=[[[1]]],
        loss_components={"pre": {"shape": 1.0, "rows": 2.0}, "post": {"shape": 0.5, "rows": 1.0}},
        recurrent_weight_movement=1.0,
        decoded_predictions={"pre": [[1]], "post": [[2]]},
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


def test_temporary_proof_requires_changed_state_observations() -> None:
    kwargs = dict(
        training_task="d631b094",
        validation_task="46f33fce",
        update_tasks=["d631b094"] * 8,
        validation_state_before=b"same",
        validation_state_after=b"same",
        prediction_before=b"before",
        prediction_after=b"after",
            interventions={name: {"changed": False} for name in ("voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null")},
            elapsed_seconds=1.0,
            targets=[[[1]]],
            loss_components={"pre": {"shape": 1.0, "rows": 2.0}, "post": {"shape": 0.5, "rows": 1.0}},
            recurrent_weight_movement=1.0,
            decoded_predictions={"pre": [[1]], "post": [[2]]},
    )
    with pytest.raises(RuntimeError, match="no direct change"):
        validate_temporary_proof(**kwargs)
    kwargs["interventions"]["null"]["changed"] = True
    with pytest.raises(RuntimeError, match="null intervention"):
        validate_temporary_proof(**kwargs)


def test_backend_command_isolated_and_rejects_unknown_backend(tmp_path) -> None:
    command = backend_command("cpu", tmp_path, tmp_path / "cpu.json")
    assert "--child" in command
    assert command[command.index("--backend") + 1] == "cpu"
    full = backend_command("gpu", tmp_path, tmp_path / "gpu.json", measure_decoder=True, probe_only=False)
    assert "--measure-decoder" in full
    assert "--probe-only" not in full
    with pytest.raises(ValueError, match="cpu or gpu"):
        backend_command("tpu", tmp_path, tmp_path / "x.json")


def test_data_root_accepts_arc_root_and_training_directory(tmp_path) -> None:
    from examples.pp_prop import example21_gate4 as gate4

    arc_root = tmp_path / "raw"
    training = arc_root / "data" / "training"
    assert gate4._normalize_data_root(arc_root) == arc_root
    assert gate4._normalize_data_root(training) == arc_root


def test_run_backend_process_enforces_parent_deadline(monkeypatch, tmp_path) -> None:
    from examples.pp_prop import example21_gate4 as gate4

    captured = {}

    def fake_run(*args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(gate4.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="180 seconds"):
        gate4.run_backend_process(
            "cpu", tmp_path, tmp_path / "scratch", deadline=time.perf_counter() + 5.0
        )
    assert 0.0 < captured["timeout"] <= 5.0


def test_process_evidence_rejects_missing_or_invalid_payload(tmp_path) -> None:
    from examples.pp_prop import example21_gate4 as gate4

    path = tmp_path / "evidence.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        gate4._read_process_evidence(path)
    path.write_text(json.dumps({"backend": "cpu", "prediction_hex": "70", "times_ms": [], "finite": True}))
    with pytest.raises(ValueError, match="invalid"):
        gate4._read_process_evidence(path)


def test_process_runner_rejects_expired_and_failed_children(monkeypatch, tmp_path) -> None:
    from examples.pp_prop import example21_gate4 as gate4

    with pytest.raises(RuntimeError, match="180 seconds"):
        gate4.run_backend_process("cpu", tmp_path, tmp_path, deadline=time.perf_counter() - 1)

    class Completed:
        returncode = 1
        stderr = "failure"

    monkeypatch.setattr(gate4.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="failed"):
        gate4.run_backend_process("cpu", tmp_path, tmp_path)


def test_internal_serializers_and_synchronizer(monkeypatch) -> None:
    from examples.pp_prop import example21_gate4 as gate4
    import jax

    assert gate4._json_safe({"x": (b"a", 1)}) == {"x": [{"encoding": "hex", "value": "61"}, 1]}
    assert gate4._decode_proof({"prediction_before": "61"})["prediction_before"] == b"a"
    assert gate4._tree_finite(1.0)
    assert not gate4._tree_finite(())

    class Leaf:
        def __init__(self):
            self.ready = False

        def block_until_ready(self):
            self.ready = True

    leaf = Leaf()
    monkeypatch.setattr(jax.tree_util, "tree_leaves", lambda value: [leaf])
    gate4._synchronize_tree(leaf)
    assert leaf.ready


def test_run_gate4_writes_durable_evidence(monkeypatch, tmp_path) -> None:
    from examples.pp_prop import example21_gate4 as gate4

    def fake_process(backend, root, scratch, **kwargs):
        payload = {
            "backend": backend,
            "prediction_hex": "70",
            "times_ms": [3.0, 2.0, 1.0],
            "finite": True,
            "proof": {
                "training_task": "d631b094",
                "validation_task": "46f33fce",
                "update_tasks": ["d631b094"] * 8,
                "validation_state_before": b"same",
                "validation_state_after": b"same",
                "prediction_before": b"before",
                "prediction_after": b"after",
                "interventions": {name: {"changed": name != "null"} for name in ("voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null")},
                "elapsed_seconds": 0.0,
                "targets": [[[1]]],
                "loss_components": {"pre": {"shape": 1.0, "rows": 2.0}, "post": {"shape": 0.5, "rows": 1.0}},
                "recurrent_weight_movement": 1.0,
                "decoded_predictions": {"pre": [[1]], "post": [[2]]},
            },
        }
        return gate4.ProcessEvidence(backend, b"p", (3.0, 2.0, 1.0), True, payload)

    monkeypatch.setattr(gate4, "run_backend_process", fake_process)
    result = run_gate4(tmp_path, tmp_path / "result.json")
    assert result["backend"]["selected_backend"] == "cpu"
    evidence = tmp_path / "gate4-evidence.json"
    assert json.loads(evidence.read_text())['proof']['update_count'] == 8
    assert (tmp_path / "gate4-evidence.sha256").exists()
