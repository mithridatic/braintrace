"""Resume, validate, and decide curriculum adoption on development bundles."""

from __future__ import annotations

import msgspec_json
import pathlib
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from temporal_benchmark_config import HORIZONS, config_to_dict
from temporal_benchmark_curriculum_adoption_config import (
    CURRICULUM_ADOPTION_KIND,
    CURRICULUM_ADOPTION_SCHEMA_VERSION,
    CURRICULUM_BUNDLE_KIND,
    CurriculumAdoptionSettings,
    curriculum_adoption_settings_document,
    curriculum_bundle_command,
    expected_curriculum_config,
    selected_config_document,
)
from temporal_benchmark_metrics import BundleValue, hierarchical_paired_interval
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_runner import load_raw_document, run_subprocess
from temporal_benchmark_search_selection import (
    ResumeConfigurationError,
    RunEvidenceError,
    ensure_finite,
)

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class CurriculumBundleEvidence:
    """Store one validated paired development bundle."""

    bundle_id: str
    curriculum_accuracy: float
    direct_long_accuracy: float
    curriculum_time_to_0_80: int | None
    direct_long_time_to_0_80: int | None
    curriculum_total_sample_ticks: int
    direct_long_updates: int
    stable: bool
    raw_path: str
    reused: bool


def _print_progress(message: str) -> None:
    print(message, flush=True)


def raw_curriculum_bundle_path(
    settings: CurriculumAdoptionSettings, bundle_id: str
) -> pathlib.Path:
    """Return the stable raw path for one paired bundle."""
    return settings.output_directory / "raw" / f"{bundle_id}.json"


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RunEvidenceError(f"{location} must be an object")
    return value


def _accuracy(result: Mapping[str, object], location: str) -> float:
    validation = _mapping(
        result.get("final_validation"), f"{location}.final_validation"
    )
    value = validation.get("ensemble_accuracy")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunEvidenceError(f"{location} accuracy must be numeric")
    accuracy = float(value)
    if not 0.0 <= accuracy <= 1.0:
        raise RunEvidenceError(f"{location} accuracy is outside [0, 1]")
    return accuracy


def _optional_tick(value: object, location: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunEvidenceError(f"{location} must be a positive integer or null")
    return value


def _validate_provenance(
    document: Mapping[str, object], settings: CurriculumAdoptionSettings
) -> None:
    environment = _mapping(document.get("environment"), "environment")
    actual = (
        environment.get("source_commit"),
        environment.get("source_dirty"),
        environment.get("container_image_digest"),
    )
    expected = (
        settings.source_commit,
        settings.source_dirty,
        settings.container_image_digest,
    )
    if actual != expected:
        raise ResumeConfigurationError(
            "raw curriculum result provenance does not match the requested source"
        )


def _validate_budget(budget: Mapping[str, object], batch_size: int) -> tuple[int, int]:
    phases = _mapping(budget.get("curriculum_phases"), "sample_tick_budget.phases")
    total = 0
    for horizon in ("short", "medium", "long"):
        if horizon not in phases:
            continue
        phase = _mapping(phases[horizon], f"sample_tick_budget.{horizon}")
        updates = phase.get("updates_completed")
        ticks = phase.get("sample_ticks")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise RunEvidenceError(
                "sample-tick phase updates must be positive integers"
            )
        expected = updates * batch_size * HORIZONS[horizon].total_steps
        if ticks != expected:
            raise RunEvidenceError("sample-tick phase arithmetic does not match")
        total += expected
    direct = _mapping(budget.get("direct_long"), "sample_tick_budget.direct_long")
    updates = direct.get("updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise RunEvidenceError("direct-long updates must be a positive integer")
    direct_total = updates * batch_size * HORIZONS["long"].total_steps
    if (
        budget.get("curriculum_total_sample_ticks") != total
        or direct.get("total_sample_ticks") != direct_total
        or total != direct_total
        or budget.get("exact_match") is not True
    ):
        raise RunEvidenceError("curriculum and direct sample-tick budgets do not match")
    return total, updates


def validate_curriculum_bundle_document(
    document: Mapping[str, object],
    settings: CurriculumAdoptionSettings,
    bundle_id: str,
    raw_path: str,
    *,
    reused: bool,
) -> CurriculumBundleEvidence:
    """Validate exact configuration, provenance, metrics, and tick arithmetic."""
    if (
        document.get("schema_version") != CURRICULUM_ADOPTION_SCHEMA_VERSION
        or document.get("kind") != CURRICULUM_BUNDLE_KIND
        or document.get("development_only") is not True
        or document.get("sealed_test") is not False
    ):
        raise RunEvidenceError("raw curriculum result has an unsupported schema")
    _validate_provenance(document, settings)
    if document.get("selected_config") != selected_config_document(settings):
        raise ResumeConfigurationError(
            "raw curriculum result selected configuration does not exactly match"
        )
    ensure_finite(document)
    result = _mapping(document.get("result"), "result")
    expected = expected_curriculum_config(settings, bundle_id)
    if result.get("base_config") != config_to_dict(expected):
        raise ResumeConfigurationError(
            "raw curriculum result base configuration does not exactly match"
        )
    if result.get("status") != "completed" or result.get("bundle_id") != bundle_id:
        raise RunEvidenceError("paired curriculum bundle did not complete")
    if result.get("sealed_test_metrics") is not None:
        raise RunEvidenceError("development comparison materialized sealed metrics")
    curriculum = _mapping(result.get("curriculum"), "result.curriculum")
    direct = _mapping(result.get("direct_long"), "result.direct_long")
    if curriculum.get("config") != config_to_dict(expected):
        raise ResumeConfigurationError("curriculum child configuration drifted")
    budget = _mapping(result.get("sample_tick_budget"), "result.sample_tick_budget")
    total_ticks, direct_updates = _validate_budget(budget, settings.batch_size)
    expected_direct = replace(expected, curriculum=False, updates=direct_updates)
    if direct.get("config") != config_to_dict(expected_direct):
        raise ResumeConfigurationError("direct-long child configuration drifted")
    if (
        curriculum.get("sealed_test_metrics") is not None
        or direct.get("sealed_test_metrics") is not None
    ):
        raise RunEvidenceError("development comparison materialized sealed metrics")
    times = _mapping(result.get("time_to_0_80"), "result.time_to_0_80")
    stability = _mapping(result.get("stability"), "result.stability")
    if not all(
        isinstance(stability.get(name), bool) for name in ("curriculum", "direct_long")
    ):
        raise RunEvidenceError("paired stability evidence must be boolean")
    curriculum_time = _optional_tick(
        times.get("curriculum_sample_ticks"), "curriculum threshold time"
    )
    direct_time = _optional_tick(
        times.get("direct_long_sample_ticks"), "direct threshold time"
    )
    if any(
        value is not None and value > total_ticks
        for value in (curriculum_time, direct_time)
    ):
        raise RunEvidenceError("threshold time exceeds the matched sample-tick budget")
    return CurriculumBundleEvidence(
        bundle_id=bundle_id,
        curriculum_accuracy=_accuracy(curriculum, "curriculum"),
        direct_long_accuracy=_accuracy(direct, "direct_long"),
        curriculum_time_to_0_80=curriculum_time,
        direct_long_time_to_0_80=direct_time,
        curriculum_total_sample_ticks=total_ticks,
        direct_long_updates=direct_updates,
        stable=stability.get("curriculum") is True
        and stability.get("direct_long") is True,
        raw_path=raw_path,
        reused=reused,
    )


def _failure_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(".failure.json")


def _write_failure(
    settings: CurriculumAdoptionSettings, bundle_id: str, error: Exception
) -> None:
    document: dict[str, object] = {
        "schema_version": CURRICULUM_ADOPTION_SCHEMA_VERSION,
        "kind": "temporal_credit_curriculum_comparison_child_failure",
        "development_only": True,
        "sealed_test": False,
        "bundle_id": bundle_id,
        "selected_config": selected_config_document(settings),
        "provenance": curriculum_adoption_settings_document(settings)["provenance"],
        "error_type": type(error).__name__,
        "message": str(error)[-4000:],
    }
    _write_json(
        _failure_path(raw_curriculum_bundle_path(settings, bundle_id)), document
    )


def obtain_curriculum_bundle_evidence(
    settings: CurriculumAdoptionSettings,
    bundle_id: str,
    runner: CommandRunner,
) -> CurriculumBundleEvidence:
    """Reuse an exact paired run or atomically materialize a fresh one."""
    path = raw_curriculum_bundle_path(settings, bundle_id)
    partial = path.with_suffix(".partial.json")
    partial.unlink(missing_ok=True)
    relative = str(path.relative_to(settings.output_directory))
    if path.is_file():
        return validate_curriculum_bundle_document(
            load_raw_document(path), settings, bundle_id, relative, reused=True
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner(
            curriculum_bundle_command(settings, bundle_id, partial),
            settings.source_root,
        )
        if not partial.is_file():
            raise RunEvidenceError(
                f"paired child did not write staged result: {partial}"
            )
        evidence = validate_curriculum_bundle_document(
            load_raw_document(partial), settings, bundle_id, relative, reused=False
        )
    except (RuntimeError, RunEvidenceError) as error:
        _write_failure(settings, bundle_id, error)
        raise
    partial.replace(path)
    return evidence


def _split_id(bundle_id: str) -> str:
    return bundle_id.split("-", maxsplit=1)[0]


def _decision_document(
    settings: CurriculumAdoptionSettings,
    evidence: Sequence[CurriculumBundleEvidence],
) -> dict[str, object]:
    differences = tuple(
        BundleValue(
            _split_id(item.bundle_id),
            item.bundle_id,
            item.curriculum_accuracy - item.direct_long_accuracy,
        )
        for item in evidence
    )
    interval = hierarchical_paired_interval(differences)
    time_complete = all(
        item.curriculum_time_to_0_80 is not None
        and item.direct_long_time_to_0_80 is not None
        for item in evidence
    )
    time_reduction: float | None = None
    if time_complete:
        curriculum_times = tuple(item.curriculum_time_to_0_80 for item in evidence)
        direct_times = tuple(item.direct_long_time_to_0_80 for item in evidence)
        assert all(value is not None for value in curriculum_times)
        assert all(value is not None for value in direct_times)
        curriculum_mean = statistics.fmean(
            int(value) for value in curriculum_times if value is not None
        )
        direct_mean = statistics.fmean(
            int(value) for value in direct_times if value is not None
        )
        time_reduction = (direct_mean - curriculum_mean) / direct_mean
    stable = all(item.stable for item in evidence)
    static_control = settings.example15_accuracy_change >= -0.01
    time_gate = time_reduction is not None and time_reduction >= 0.20
    accuracy_gate = interval.lower > 0.0
    adoption = stable and static_control and (time_gate or accuracy_gate)
    fixed = curriculum_adoption_settings_document(settings)
    return {
        "schema_version": CURRICULUM_ADOPTION_SCHEMA_VERSION,
        "kind": CURRICULUM_ADOPTION_KIND,
        "status": "completed",
        "development_only": True,
        "sealed_test": False,
        "provenance": fixed["provenance"],
        "development_bundles": fixed["development_bundles"],
        "device": settings.device,
        "neurons": settings.neurons,
        "degree": settings.degree,
        "batch_size": settings.batch_size,
        "evaluation_interval": settings.evaluation_interval,
        "selected_config": selected_config_document(settings),
        "adoption": adoption,
        "decision_evidence": {
            "time_to_0_80_complete": time_complete,
            "time_to_0_80_reduction_fraction": time_reduction,
            "paired_long_accuracy_interval": asdict(interval),
            "example15_accuracy_change": settings.example15_accuracy_change,
            "all_paired_runs_stable": stable,
            "time_gate_passed": time_gate,
            "accuracy_gate_passed": accuracy_gate,
            "static_control_gate_passed": static_control,
        },
        "bundle_evidence": [
            {
                **asdict(item),
                "paired_long_accuracy_difference": (
                    item.curriculum_accuracy - item.direct_long_accuracy
                ),
            }
            for item in evidence
        ],
    }


def run_development_curriculum_adoption(
    settings: CurriculumAdoptionSettings,
    *,
    runner: CommandRunner = run_subprocess,
    progress: ProgressReporter = _print_progress,
) -> dict[str, object]:
    """Run or resume all paired bundles and write the adoption decision."""
    evidence: list[CurriculumBundleEvidence] = []
    for bundle_id in DEVELOPMENT_BUNDLES:
        started = time.perf_counter()
        progress(f"curriculum bundle={bundle_id} start")
        evidence.append(obtain_curriculum_bundle_evidence(settings, bundle_id, runner))
        progress(
            f"curriculum bundle={bundle_id} result=accepted "
            f"wall_seconds={time.perf_counter() - started:.3f}"
        )
    document = _decision_document(settings, evidence)
    _write_json(settings.output_directory / "curriculum-adoption.json", document)
    return document
