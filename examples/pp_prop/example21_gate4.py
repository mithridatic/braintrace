"""Gate 4 probes and bounded real-data proof validation for Example 21."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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


def measure_probe(
    call: Callable[[], Any],
    *,
    backend: str,
    prediction_bytes: bytes,
    finite: bool,
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
    synchronize = synchronize or (lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else None)
    synchronize(call())
    samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        synchronize(call())
        samples.append((time.perf_counter() - started) * 1000.0)
    return BackendProbe(backend, tuple(samples), finite, prediction_bytes)


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
    }
