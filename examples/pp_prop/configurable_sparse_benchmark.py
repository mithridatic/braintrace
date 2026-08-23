"""Command-line orchestration for the configurable sparse pp-prop benchmark."""

from __future__ import annotations

import msgspec_json
import pathlib
import sys
from typing import Sequence

try:
    from .sparse_benchmark_config import (
        SparseBenchmarkConfig,
        config_to_cli_args,
        config_to_dict,
        parse_config,
    )
    from .sparse_benchmark_device import apply_device_selection
    from .sparse_benchmark_supervisor import (
        ResourceLimits,
        SupervisedResult,
        run_supervised,
    )
except ImportError:
    from sparse_benchmark_config import (
        SparseBenchmarkConfig,
        config_to_cli_args,
        config_to_dict,
        parse_config,
    )
    from sparse_benchmark_device import apply_device_selection
    from sparse_benchmark_supervisor import (
        ResourceLimits,
        SupervisedResult,
        run_supervised,
    )


_WORKER_FLAG = "--worker"
_RESULT_SCHEMA_VERSION = 2
_SUPPORTED_RESULT_SCHEMAS = {1, _RESULT_SCHEMA_VERSION}
_TARGET_MISS_EXIT_CODE = 3
_WORKER_STATUSES = {"completed", "target_reached", "target_not_reached"}
_UNMEASURED_DEVICE_MEMORY: dict[str, object] = {
    "device_memory_scope": None,
    "device_peak_bytes": None,
    "device_peak_gib": None,
}


def _host_memory(result: SupervisedResult) -> dict[str, object]:
    return {
        "scope": "cpu_process_tree_rss",
        "peak_rss_bytes": result.peak_rss_bytes,
        "peak_rss_gib": result.peak_rss_bytes / 2**30,
        "peak_rss_is_sampled": True,
        "sampling_interval_seconds": 0.1,
        "guard_status": result.status,
        "guard_reason": result.guard_reason,
    }


def _worker_memory(payload: dict[str, object]) -> dict[str, object]:
    reported = payload.get("memory")
    if not isinstance(reported, dict):
        return dict(_UNMEASURED_DEVICE_MEMORY)
    return {
        name: reported.get(name, value)
        for name, value in _UNMEASURED_DEVICE_MEMORY.items()
    }


def _limits(config: SparseBenchmarkConfig) -> ResourceLimits:
    return ResourceLimits(
        int(config.max_rss_gib * 2**30),
        int(config.min_available_gib * 2**30),
        config.max_wall_seconds,
    )


def _worker_command(config: SparseBenchmarkConfig) -> list[str]:
    script = pathlib.Path(__file__).with_name("16-configurable-sparse-benchmark.py")
    return [sys.executable, str(script), *config_to_cli_args(config), _WORKER_FLAG]


def _failure_payload(
    config: SparseBenchmarkConfig, result: SupervisedResult
) -> dict[str, object]:
    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "status": result.status,
        "config": config_to_dict(config),
        "error_output": result.stdout,
        "memory": {**_UNMEASURED_DEVICE_MEMORY, **_host_memory(result)},
    }


def _parse_worker_payload(output: str) -> tuple[dict[str, object], list[str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        try:
            payload = msgspec_json.loads(lines[index])
        except msgspec_json.JSONDecodeError:
            continue
        schema = payload.get("schema_version") if isinstance(payload, dict) else None
        required = ("status", "config", "metrics", "timings", "environment")
        valid = (
            isinstance(payload, dict)
            and schema in _SUPPORTED_RESULT_SCHEMAS
            and payload.get("status") in _WORKER_STATUSES
            and all(isinstance(payload.get(name), dict) for name in required[1:])
            and (schema == 1 or isinstance(payload.get("topology"), dict))
        )
        if valid:
            return payload, [*lines[:index], *lines[index + 1 :]]
    raise ValueError("Worker did not emit a schema-versioned JSON result. Fix the input condition named in the error, then rerun the operation.")


def _completed_payload(
    config: SparseBenchmarkConfig, result: SupervisedResult
) -> dict[str, object]:
    try:
        payload, progress = _parse_worker_payload(result.stdout)
    except ValueError:
        failed = SupervisedResult(
            1, result.stdout, result.peak_rss_bytes, "failed", None
        )
        return _failure_payload(config, failed)
    if payload["config"] != config_to_dict(config):
        failed = SupervisedResult(
            1, "worker configuration mismatch", result.peak_rss_bytes, "failed", None
        )
        return _failure_payload(config, failed)
    for line in progress:
        print(line, file=sys.stderr)
    payload["memory"] = {**_worker_memory(payload), **_host_memory(result)}
    return payload


def _emit(payload: dict[str, object], config: SparseBenchmarkConfig) -> None:
    serialized = msgspec_json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if config.json_output is not None:
        config.json_output.parent.mkdir(parents=True, exist_ok=True)
        config.json_output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _exit_code(
    config: SparseBenchmarkConfig, result: SupervisedResult, payload: dict[str, object]
) -> int:
    if result.exit_code:
        return result.exit_code
    if payload.get("status") == "failed":
        return 1
    metrics = payload.get("metrics")
    threshold = metrics.get("threshold_updates") if isinstance(metrics, dict) else None
    if config.require_target and threshold is None:
        return _TARGET_MISS_EXIT_CODE
    return 0


def _run_parent(config: SparseBenchmarkConfig) -> int:
    result = run_supervised(_worker_command(config), _limits(config))
    payload = (
        _completed_payload(config, result)
        if result.status == "completed"
        else _failure_payload(config, result)
    )
    _emit(payload, config)
    return _exit_code(config, result, payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark worker or its isolated resource supervisor.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments without the executable name.

    Returns
    -------
    int
        Process exit code.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    worker = _WORKER_FLAG in arguments
    arguments = [argument for argument in arguments if argument != _WORKER_FLAG]
    config = parse_config(arguments)
    if worker:
        apply_device_selection(config.device)
        try:
            from .sparse_benchmark_worker import run_benchmark
        except ImportError:
            from sparse_benchmark_worker import run_benchmark
        print(msgspec_json.dumps(run_benchmark(config), allow_nan=False, sort_keys=True))
        return 0
    return _run_parent(config)
