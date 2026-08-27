"""Gate 4 probes and bounded real-data proof validation for Example 21."""

from __future__ import annotations

import math
import argparse
import hashlib
import json
import os
import subprocess
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import brainunit as u


@dataclass(frozen=True)
class BackendProbe:
    """Measured result for one backend process.

    Parameters
    ----------
    backend : str
        Backend name.
    times_ms : tuple[float, ...]
        Synchronized timed calls, in milliseconds.
    finite : bool
        Whether the gradient and prediction outputs were finite.
    prediction_bytes : bytes
        Prediction bytes returned by the probe.
    """

    backend: str
    times_ms: tuple[float, ...]
    finite: bool
    prediction_bytes: bytes

    @property
    def median_ms(self) -> float:
        """Return the measured median in milliseconds."""
        return float(statistics.median(self.times_ms))

    @property
    def valid(self) -> bool:
        """Return whether this probe can participate in selection."""
        return bool(
            self.finite
            and self.times_ms
            and all(math.isfinite(t) and t >= 0.0 for t in self.times_ms)
        )


def validate_backend_probes(cpu: BackendProbe, gpu: BackendProbe) -> dict[str, object]:
    """Validate matched backend outputs and return selection evidence.

    Raises
    ------
    RuntimeError
        If the matched probes produce different predictions.
    """
    if cpu.prediction_bytes != gpu.prediction_bytes:
        raise RuntimeError("matched backend probes produced different predictions")
    selected = select_backend(cpu, gpu)
    return {
        "selected_backend": selected,
        "cpu_median_ms": cpu.median_ms if cpu.valid else None,
        "gpu_median_ms": gpu.median_ms if gpu.valid else None,
        "prediction_bytes_stable": True,
    }


def select_backend(cpu: BackendProbe, gpu: BackendProbe) -> str:
    """Select the lower valid median, with CPU winning an exact tie.

    Raises
    ------
    RuntimeError
        If neither probe is valid.
    """
    valid = [probe for probe in (cpu, gpu) if probe.valid]
    if not valid:
        raise RuntimeError("no valid backend probe")
    winner = min(valid, key=lambda probe: (probe.median_ms, 0 if probe.backend == "cpu" else 1))
    return winner.backend


def _tree_finite(value: Any) -> bool:
    """Return whether every numeric leaf is finite."""
    import jax

    leaves = jax.tree_util.tree_leaves(value)
    return bool(leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves))


def _synchronize_tree(value: Any) -> None:
    """Wait for every asynchronous array leaf."""
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure_probe(
    call: Callable[[], Any],
    *,
    backend: str,
    prediction_bytes: bytes,
    finite: bool | None = None,
    synchronize: Callable[[Any], None] | None = None,
) -> BackendProbe:
    """Warm once and measure three synchronized calls without updating state.

    Parameters
    ----------
    call : callable
        Pure compiled gradient call.
    backend : str
        Backend label.
    prediction_bytes : bytes
        Prediction bytes from the matched call.
    finite : bool
        Finite-gradient and finite-prediction result.
    synchronize : callable, optional
        Function that blocks until a call result is ready.
    """
    synchronize = synchronize or _synchronize_tree
    warmed = call()
    synchronize(warmed)
    measured_finite = _tree_finite(warmed) if finite is None else finite
    samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        synchronize(call())
        samples.append((time.perf_counter() - started) * 1000.0)
    return BackendProbe(backend, tuple(samples), measured_finite, prediction_bytes)


def benchmark_decoder(
    decoder: Callable[[Any], Any],
    requests: Iterable[Any],
    *,
    synchronize: Callable[[Any], None] | None = None,
    calls_per_request: int = 5,
    limit_ms: float = 100.0,
) -> dict[str, object]:
    """Measure five warmed direct decoder calls per request.

    Raises
    ------
    ValueError
        If the call count is not positive.
    RuntimeError
        If any timed call exceeds the limit.
    """
    if calls_per_request <= 0:
        raise ValueError("calls_per_request must be positive")
    synchronize = synchronize or (lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else None)
    records: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        synchronize(decoder(request))
        times: list[float] = []
        for _ in range(calls_per_request):
            started = time.perf_counter()
            synchronize(decoder(request))
            elapsed = (time.perf_counter() - started) * 1000.0
            times.append(elapsed)
            if elapsed > limit_ms:
                raise RuntimeError(f"decoder request {index} exceeded {limit_ms:g} ms")
        records.append({"request": index, "calls_ms": times, "median_ms": float(statistics.median(times))})
    return {"calls_per_request": calls_per_request, "limit_ms": limit_ms, "requests": records}


def validate_temporary_proof(
    *,
    training_task: str,
    validation_task: str,
    update_tasks: Sequence[str],
    validation_state_before: bytes,
    validation_state_after: bytes,
    prediction_before: bytes,
    prediction_after: bytes,
    interventions: Mapping[str, Any],
    elapsed_seconds: float,
    limit_seconds: float = 180.0,
    targets: Sequence[Any] | None = None,
    loss_components: Mapping[str, float] | None = None,
    recurrent_weight_movement: float | None = None,
    decoded_predictions: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Validate the bounded proof's data isolation and direct behavior.

    Raises
    ------
    RuntimeError
        If a Gate 4 condition is not met.
    """
    if training_task != "d631b094" or validation_task != "46f33fce":
        raise RuntimeError("proof tasks do not match the Gate 4 data contract")
    if tuple(update_tasks) != (training_task,) * 8:
        raise RuntimeError("proof must perform exactly eight training-task updates")
    if validation_state_before != validation_state_after:
        raise RuntimeError("validation changed model state")
    if prediction_before == prediction_after:
        raise RuntimeError("training did not change the direct prediction")
    if elapsed_seconds > limit_seconds:
        raise RuntimeError("temporary proof exceeded its time limit")
    if not targets:
        raise RuntimeError("proof must record target grids")
    if not isinstance(loss_components, Mapping) or set(loss_components) != {"pre", "post"}:
        raise RuntimeError("proof must record pre and post loss components")
    for phase in ("pre", "post"):
        values = loss_components[phase]
        if not isinstance(values, Mapping) or not {"shape", "rows"}.issubset(values):
            raise RuntimeError("loss components must include shape and rows for both phases")
        if not all(math.isfinite(float(values[name])) for name in ("shape", "rows")):
            raise RuntimeError("loss components must be finite")
    if recurrent_weight_movement is None or not math.isfinite(float(recurrent_weight_movement)) or recurrent_weight_movement <= 0.0:
        raise RuntimeError("proof must record changed recurrent weights")
    if not isinstance(decoded_predictions, Mapping) or not decoded_predictions.get("pre") or not decoded_predictions.get("post"):
        raise RuntimeError("proof must record decoded pre and post predictions")
    required = {"voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null"}
    missing = required.difference(interventions)
    if missing:
        raise RuntimeError(f"missing state interventions: {sorted(missing)}")
    if any(
        not isinstance(observation, Mapping)
        or not isinstance(observation.get("changed"), bool)
        for observation in interventions.values()
    ):
        raise RuntimeError("interventions must record a Boolean changed field")
    if interventions["null"]["changed"]:
        raise RuntimeError("null intervention must not change the prediction")
    if not any(interventions[name]["changed"] for name in required if name != "null"):
        raise RuntimeError("state interventions produced no direct change")
    return {
        "training_task": training_task,
        "validation_task": validation_task,
        "update_count": len(update_tasks),
        "validation_state_unchanged": True,
        "prediction_changed": True,
        "elapsed_seconds": float(elapsed_seconds),
        "interventions": dict(interventions),
        "targets": list(targets) if targets is not None else [],
        "loss_components": dict(loss_components or {}),
        "recurrent_weight_movement": recurrent_weight_movement,
        "decoded_predictions": dict(decoded_predictions),
    }


@dataclass(frozen=True)
class ProcessEvidence:
    """JSON-safe evidence returned by one backend child process."""

    backend: str
    prediction_bytes: bytes
    times_ms: tuple[float, ...]
    finite: bool
    payload: Mapping[str, object]


def backend_command(
    backend: str,
    root: str | Path,
    evidence: str | Path,
    *,
    measure_decoder: bool = False,
    probe_only: bool = True,
) -> list[str]:
    """Build the isolated child-process command for one JAX backend."""
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be cpu or gpu")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--backend",
        backend,
        "--data-root",
        str(Path(root).resolve()),
        "--evidence",
        str(Path(evidence).resolve()),
    ]
    if measure_decoder:
        command.append("--measure-decoder")
    if probe_only:
        command.append("--probe-only")
    return command


def _read_process_evidence(path: str | Path) -> ProcessEvidence:
    """Read and validate one child result."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"backend", "prediction_hex", "times_ms", "finite"}
    if set(payload) < required:
        raise ValueError("backend evidence is missing required fields")
    prediction = bytes.fromhex(payload["prediction_hex"])
    times = tuple(float(value) for value in payload["times_ms"])
    probe = BackendProbe(payload["backend"], times, bool(payload["finite"]), prediction)
    if not probe.valid:
        raise ValueError("backend evidence is invalid")
    return ProcessEvidence(probe.backend, prediction, times, probe.finite, payload)


def run_backend_process(
    backend: str,
    root: str | Path,
    scratch: str | Path,
    *,
    measure_decoder: bool = False,
    probe_only: bool = True,
    deadline: float | None = None,
) -> ProcessEvidence:
    """Run one CPU or GPU proof process and return its durable evidence."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    evidence = scratch / f"backend-{backend}.json"
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda"
    timeout = None if deadline is None else deadline - time.perf_counter()
    if timeout is not None and timeout <= 0.0:
        raise RuntimeError("Gate 4 proof exceeded 180 seconds")
    try:
        completed = subprocess.run(
            backend_command(
                backend,
                root,
                evidence,
                measure_decoder=measure_decoder,
                probe_only=probe_only,
            ),
            check=False,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Gate 4 proof exceeded 180 seconds") from error
    if completed.returncode:
        raise RuntimeError(f"{backend} proof process failed: {completed.stderr[-2000:]}")
    return _read_process_evidence(evidence)


def write_gate4_evidence(path: str | Path, evidence: Mapping[str, object]) -> None:
    """Write complete Gate 4 evidence outside the bounded result file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(evidence), sort_keys=True, indent=2, allow_nan=False).encode()
    target.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    target.with_suffix(".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    """Convert binary evidence and nested containers to JSON values."""
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _decode_proof(value: Mapping[str, object]) -> dict[str, object]:
    """Restore binary proof fields read from a child JSON document."""
    result = dict(value)
    for name in ("validation_state_before", "validation_state_after", "prediction_before", "prediction_after"):
        item = result.get(name)
        if isinstance(item, str):
            result[name] = bytes.fromhex(item)
    return result


def _normalize_data_root(root: str | Path) -> Path:
    """Accept either the ARC root or its direct training directory."""
    path = Path(root)
    if path.name == "training" and path.parent.name == "data":
        return path.parent.parent
    return path


def _request_readouts(module: Any, model: Any, voltages: Any) -> np.ndarray:  # pragma: no cover
    """Return the 31 shape/row request logits from a 705-event rollout."""
    values = np.asarray(voltages)
    if values.shape[0] != 705:
        raise ValueError("rollout must contain 705 events")
    features = np.tanh((values[[673, *range(675, 705)]] + 65.0) / 20.0)
    return features @ np.asarray(model.readout_weight.value) + np.asarray(model.readout_bias.value)


def run_gate4(root: str | Path, output: str | Path) -> dict[str, object]:
    """Execute the bounded two-process Gate 4 proof against ARC raw data."""
    root = _normalize_data_root(root)
    started = time.perf_counter()
    deadline = started + 180.0
    output = Path(output)
    scratch = output.parent / ".gate4-processes"
    cpu = run_backend_process("cpu", root, scratch, deadline=deadline)
    gpu = run_backend_process("gpu", root, scratch, deadline=deadline)
    probes = (
        BackendProbe(cpu.backend, cpu.times_ms, cpu.finite, cpu.prediction_bytes),
        BackendProbe(gpu.backend, gpu.times_ms, gpu.finite, gpu.prediction_bytes),
    )
    backend = validate_backend_probes(*probes)
    selected = run_backend_process(
        backend["selected_backend"],
        root,
        scratch,
        measure_decoder=True,
        probe_only=False,
        deadline=deadline,
    )
    selected_payload = selected.payload
    proof = _decode_proof(selected_payload.get("proof", {}))
    elapsed = time.perf_counter() - started
    proof["elapsed_seconds"] = elapsed
    validation = validate_temporary_proof(**proof)
    evidence = {
        "backend": backend,
        "processes": {"cpu": cpu.payload, "gpu": gpu.payload},
        "decoder": selected_payload.get("decoder", {}),
        "proof": validation,
        "prediction_sha256": hashlib.sha256(cpu.prediction_bytes).hexdigest(),
    }
    if elapsed > 180.0:
        raise RuntimeError("Gate 4 proof exceeded 180 seconds")
    write_gate4_evidence(output.with_name("gate4-evidence.json"), evidence)
    output.write_text(json.dumps(_json_safe(evidence), sort_keys=True), encoding="utf-8")
    return evidence


def _child_main(arguments: argparse.Namespace) -> None:  # pragma: no cover
    """Execute the real-data work in an isolated backend process."""
    import importlib.util
    import numpy as np

    _ensure_runtime_dependencies()
    module_path = Path(__file__).with_name("21-braincell-arc.py")
    spec = importlib.util.spec_from_file_location("example21_arc", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Example 21 fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_root = _normalize_data_root(arguments.data_root)
    task = module.load_task(data_root, "d631b094", "practice")
    validation_task = module.load_task(data_root, "46f33fce", "practice")
    events, advances = module.encode_episode(task, 0)
    validation_episodes = [
        module.encode_episode(validation_task, index)
        for index in range(len(validation_task.queries))
    ]
    model = module.BrainCellArcModel()
    learner = module.compile_pp_prop_model(model)
    import brainstate
    decoder_runner = brainstate.transform.jit(
        lambda xs, mask: brainstate.transform.for_loop(
            lambda event, advance: model.step(event, advance), xs, mask
        )
    )
    prediction = _child_prediction(module, model, events, advances)
    initial_logits_array = np.asarray(model.readout())
    initial_logits = initial_logits_array.tobytes()

    def call():
        model.reset_episode(learner)
        return learner.etrace_grad(
            events,
            step_fn=lambda event: jnp.sum(learner.etrace_evolve(event[None, :], return_outputs=True)[0]),
            mask=advances,
            reduction="sum",
        )

    import jax.numpy as jnp
    probe = measure_probe(
        call,
        backend=arguments.backend,
        prediction_bytes=prediction,
    )
    if arguments.probe_only:
        Path(arguments.evidence).write_text(
            json.dumps(
                {
                    "backend": probe.backend,
                    "prediction_hex": probe.prediction_bytes.hex(),
                    "times_ms": list(probe.times_ms),
                    "finite": probe.finite,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return
    if arguments.measure_decoder:
        validation_readouts = []
        for encoded in validation_episodes:
            snapshot = _state_snapshot(model)
            model.reset_episode(learner)
            result = decoder_runner(encoded[0], encoded[1])
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            validation_readouts.extend(_request_readouts(module, model, result))
            _restore_state(model, snapshot)
        decoder = benchmark_decoder(
            lambda readout: module.decode_prediction(np.asarray(readout)),
            validation_readouts,
        )
        decoder["request_readout_count"] = 31
        decoder["request_readout_indices"] = [673, *range(675, 705)]
    else:
        decoder = {"calls_per_request": 0, "limit_ms": 100.0, "requests": []}
    validation_state_before = _state_bytes(model)
    _decoder_call(module, model, learner, validation_episodes[0], decoder_runner)
    validation_state_after = _state_bytes(model)
    trainer = module.PPPropEpisodeTrainer(
        learner,
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    recurrent_before = np.asarray(trainer.parameters["recurrent"]).copy()
    def update(_):
        model.reset_episode(learner)
        return trainer.update_episode(
            events,
            step_fn=lambda event: jnp.sum(learner.etrace_evolve(event[None, :], return_outputs=True)[0]),
            loss_mask=advances,
        )
    brainstate.transform.for_loop(update, jnp.arange(8))
    pre_logits = initial_logits_array
    target = np.asarray(next(target for target in task.targets if target is not None))
    pre_prediction = module.decode_prediction(pre_logits)
    pre_loss = _loss_components(module, pre_logits, target)
    post_logits = _episode_logits(module, model, learner, events, advances)
    post_prediction = module.decode_prediction(post_logits)
    loss_components = {"pre": pre_loss, "post": _loss_components(module, post_logits, target)}
    proof = {
        "training_task": "d631b094",
        "validation_task": "46f33fce",
        "update_tasks": ["d631b094"] * 8,
        "validation_state_before": validation_state_before.hex(),
        "validation_state_after": validation_state_after.hex(),
        "prediction_before": pre_logits.tobytes().hex(),
        "prediction_after": np.asarray(post_logits).tobytes().hex(),
        "interventions": _child_interventions(model, events[0]),
        "elapsed_seconds": 0.0,
        "targets": [target.tolist() for target in task.targets if target is not None],
        "loss_components": loss_components,
        "recurrent_weight_movement": float(np.linalg.norm(np.asarray(model.recurrent_weight.value) - recurrent_before)),
        "decoded_predictions": {"pre": pre_prediction.tolist(), "post": post_prediction.tolist()},
    }
    payload = {
        "backend": probe.backend,
        "prediction_hex": probe.prediction_bytes.hex(),
        "times_ms": list(probe.times_ms),
        "finite": probe.finite,
        "decoder": decoder,
        "proof": proof,
    }
    Path(arguments.evidence).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _ensure_runtime_dependencies() -> None:  # pragma: no cover
    """Install the example-only dependency when the proof image omits it."""
    try:
        __import__("braincell")
    except ModuleNotFoundError as error:
        if error.name != "braincell":
            raise
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "braincell==0.1.0"],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _decoder_call(
    module: Any,
    model: Any,
    learner: Any,
    encoded: tuple[Any, Any],
    runner: Callable[[Any, Any], Any] | None = None,
) -> Any:  # pragma: no cover
    """Run one fixed validation decoder request without retaining its state."""
    snapshot = _state_snapshot(model)
    model.reset_episode(learner)
    try:
        if runner is not None:
            result = runner(encoded[0], encoded[1])
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            return np.asarray(module.decode_prediction(np.asarray(model.readout()))).tobytes()
        return _child_prediction(module, model, encoded[0], encoded[1])
    finally:
        _restore_state(model, snapshot)


def _child_prediction(module: Any, model: Any, events: Any, advances: Any) -> bytes:  # pragma: no cover
    """Run one encoded episode and return canonical prediction bytes."""
    module.run_event_sequence(model, events, advances)
    return np.asarray(module.decode_prediction(np.asarray(model.readout()))).tobytes()


def _state_bytes(model: Any) -> bytes:  # pragma: no cover
    """Serialize biological state for the forward-only invariance check."""
    values = [model.cell.V.value.to_decimal(u.mV), model.previous_spikes.value]
    values.extend((model.cell.na.INa.p.value, model.cell.na.INa.q.value, model.cell.k.IK.p.value))
    return b"".join(np.asarray(value).tobytes() for value in values)


def _state_snapshot(model: Any) -> tuple[Any, ...]:  # pragma: no cover
    """Copy the biological state used by causal checks."""
    return (
        np.asarray(model.cell.V.value.to_decimal(u.mV)).copy(),
        np.asarray(model.previous_spikes.value).copy(),
        np.asarray(model.cell.na.INa.p.value).copy(),
        np.asarray(model.cell.na.INa.q.value).copy(),
        np.asarray(model.cell.k.IK.p.value).copy(),
    )


def _restore_state(model: Any, state: tuple[Any, ...]) -> None:  # pragma: no cover
    """Restore a biological-state snapshot."""
    import jax.numpy as jnp

    voltage, spikes, sodium_p, sodium_q, potassium_p = state
    model.cell.V.value = jnp.asarray(voltage) * u.mV
    model.previous_spikes.value = jnp.asarray(spikes)
    model.cell.na.INa.p.value = jnp.asarray(sodium_p)
    model.cell.na.INa.q.value = jnp.asarray(sodium_q)
    model.cell.k.IK.p.value = jnp.asarray(potassium_p)


def _episode_logits(module: Any, model: Any, learner: Any, events: Any, advances: Any) -> np.ndarray:  # pragma: no cover
    """Return direct logits for one reset episode."""
    model.reset_episode(learner)
    module.run_event_sequence(model, events, advances)
    return np.asarray(model.readout())


def _loss_components(module: Any, logits: np.ndarray, target: np.ndarray) -> dict[str, float]:  # pragma: no cover
    """Measure direct shape and valid-row loss components."""
    shape = module.request_loss(logits[:60], np.asarray(target.shape), request="shape")
    row_logits = logits[60:].reshape(30, 10)
    rows = 0.0
    for row in target:
        labels = np.zeros(30, dtype=np.int64)
        labels[: len(row)] = row
        mask = np.arange(30) < len(row)
        rows += module.request_loss(row_logits, labels, request="row", valid_mask=mask)
    return {"shape": float(shape), "rows": float(rows)}


def _child_interventions(model: Any, event: Any) -> dict[str, dict[str, object]]:  # pragma: no cover
    """Apply and measure six causal biological-state interventions."""
    import jax.numpy as jnp

    snapshot = _state_snapshot(model)
    records = {}
    names = ("voltage", "sodium_gates", "potassium_gates", "spikes", "all_state", "null")
    for name in names:
        _restore_state(model, snapshot)
        if name in {"voltage", "all_state"}:
            model.cell.V.value = model.cell.V.value + 20.0 * u.mV
        if name in {"sodium_gates", "all_state"}:
            model.cell.na.INa.p.value = jnp.ones_like(model.cell.na.INa.p.value)
            model.cell.na.INa.q.value = jnp.zeros_like(model.cell.na.INa.q.value)
        if name in {"potassium_gates", "all_state"}:
            model.cell.k.IK.p.value = jnp.ones_like(model.cell.k.IK.p.value)
        if name in {"spikes", "all_state"}:
            model.previous_spikes.value = 1.0 - model.previous_spikes.value
        model._advance(event)
        observed = np.asarray(model.readout()).tobytes()
        records[name] = {"prediction_bytes": observed.hex()}
    baseline = bytes.fromhex(records["null"]["prediction_bytes"])
    for name, record in records.items():
        record["changed"] = bytes.fromhex(record["prediction_bytes"]) != baseline
    _restore_state(model, snapshot)
    return records


def main() -> None:  # pragma: no cover
    """Run a parent or child Gate 4 command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="gate4-result.json")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument("--evidence")
    parser.add_argument("--measure-decoder", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.child:
        if not args.data_root or not args.evidence or not args.backend:
            parser.error("child mode requires --data-root, --evidence, and --backend")
        _child_main(args)
        return
    if not args.data_root:
        parser.error("parent mode requires --data-root")
    result = run_gate4(args.data_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
