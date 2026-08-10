"""Resume, validate, rank, and report trace half-life coordinate runs."""

from __future__ import annotations

import json
import pathlib
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_runner import load_raw_document, run_subprocess
from temporal_benchmark_search_selection import (
    BundleScore,
    ResumeConfigurationError,
    RunEvidenceError,
    StabilityEvidenceError,
    score_raw_document,
)
from temporal_benchmark_trace_search_config import (
    HORIZON_TRACE_GRIDS,
    TRACE_SEARCH_SCHEMA_VERSION,
    CoordinateName,
    HorizonTraceGrid,
    TraceCandidate,
    TraceSearchSettings,
    coordinate_candidates,
    expected_trace_benchmark_config,
    trace_benchmark_command,
    trace_search_settings_document,
)

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class TraceCandidateScore:
    """Store aggregate development evidence for one coordinate value."""

    candidate: TraceCandidate
    bundle_scores: tuple[BundleScore, ...]
    rejection_reasons: tuple[str, ...]
    mean_validation_nll: float | None
    mean_validation_accuracy: float | None
    mean_recurrent_update_ratio_p99: float | None

    @property
    def accepted(self) -> bool:
        """Return whether all three development bundles passed validation."""
        return not self.rejection_reasons


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _value_token(value: float) -> str:
    return format(value, "g").replace(".", "p")


def raw_trace_result_path(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    bundle_id: str,
) -> pathlib.Path:
    """Return a stable path encoding both half-lives for safe resume."""
    x_token = _value_token(candidate.x_half_life)
    f_token = _value_token(candidate.f_half_life)
    pair = f"x-{x_token}-f-{f_token}"
    return (
        settings.output_directory
        / "raw"
        / grid.horizon
        / pair
        / f"{bundle_id}.json"
    )


def _partial_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(".partial.json")


def _failure_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(".failure.json")


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_failure(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    bundle_id: str,
    error: Exception,
) -> None:
    expected = expected_trace_benchmark_config(settings, grid, candidate, bundle_id)
    document: dict[str, object] = {
        "schema_version": TRACE_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_trace_search_child_failure",
        "development_only": True,
        "sealed_test": False,
        "horizon": grid.horizon,
        "coordinate": candidate.coordinate,
        "bundle_id": bundle_id,
        "config": config_to_dict(expected),
        "provenance": {
            "container_image_digest": settings.container_image_digest,
            "source_commit": settings.source_commit,
        },
        "error_type": type(error).__name__,
        "message": str(error)[-4000:],
    }
    _write_json(
        _failure_path(raw_trace_result_path(settings, grid, candidate, bundle_id)),
        document,
    )


def obtain_trace_bundle_score(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    bundle_id: str,
    runner: CommandRunner,
) -> BundleScore:
    """Reuse an exact raw run or atomically materialize a new one."""
    path = raw_trace_result_path(settings, grid, candidate, bundle_id)
    partial = _partial_path(path)
    partial.unlink(missing_ok=True)
    expected = expected_trace_benchmark_config(settings, grid, candidate, bundle_id)
    relative_path = str(path.relative_to(settings.output_directory))
    if path.is_file():
        return score_raw_document(
            load_raw_document(path),
            expected,
            relative_path,
            settings.container_image_digest,
            settings.source_commit,
            reused=True,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner(
            trace_benchmark_command(settings, grid, candidate, bundle_id, partial),
            settings.source_root,
        )
    except RuntimeError as error:
        _write_failure(settings, grid, candidate, bundle_id, error)
        raise RunEvidenceError(f"child run failed: {error}") from error
    if not partial.is_file():
        error = RunEvidenceError(f"Example 17 did not write staged result: {partial}")
        _write_failure(settings, grid, candidate, bundle_id, error)
        raise error
    try:
        score = score_raw_document(
            load_raw_document(partial),
            expected,
            relative_path,
            settings.container_image_digest,
            settings.source_commit,
            reused=False,
        )
    except StabilityEvidenceError:
        partial.replace(path)
        raise
    except (ResumeConfigurationError, RunEvidenceError) as error:
        _write_failure(settings, grid, candidate, bundle_id, error)
        raise
    partial.replace(path)
    return score


def evaluate_trace_candidate(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    runner: CommandRunner,
    progress: ProgressReporter,
) -> TraceCandidateScore:
    """Evaluate every development bundle and reject any incomplete candidate."""
    scores: list[BundleScore] = []
    rejections: list[str] = []
    for bundle_id in DEVELOPMENT_BUNDLES:
        started = time.perf_counter()
        progress(
            f"horizon={grid.horizon} coordinate={candidate.coordinate} "
            f"half_life={candidate.half_life:g} bundle={bundle_id} start"
        )
        try:
            scores.append(
                obtain_trace_bundle_score(
                    settings, grid, candidate, bundle_id, runner
                )
            )
            progress(
                f"horizon={grid.horizon} coordinate={candidate.coordinate} "
                f"half_life={candidate.half_life:g} bundle={bundle_id} "
                f"result=accepted wall_seconds={time.perf_counter() - started:.3f}"
            )
        except RunEvidenceError as error:
            rejections.append(f"{bundle_id}: {error}")
            progress(
                f"horizon={grid.horizon} coordinate={candidate.coordinate} "
                f"half_life={candidate.half_life:g} bundle={bundle_id} "
                f"result=rejected reason={error}"
            )
    if rejections:
        return TraceCandidateScore(
            candidate, tuple(scores), tuple(rejections), None, None, None
        )
    return TraceCandidateScore(
        candidate,
        tuple(scores),
        (),
        statistics.fmean(score.validation_nll for score in scores),
        statistics.fmean(score.validation_accuracy for score in scores),
        statistics.fmean(score.recurrent_update_ratio_p99 for score in scores),
    )


def rank_trace_scores(
    scores: Sequence[TraceCandidateScore],
) -> tuple[TraceCandidateScore, ...]:
    """Rank valid candidates by NLL, accuracy, ratio, then lower half-life."""
    return tuple(
        sorted(
            (score for score in scores if score.accepted),
            key=lambda score: (
                float(score.mean_validation_nll),
                -float(score.mean_validation_accuracy),
                float(score.mean_recurrent_update_ratio_p99),
                score.candidate.half_life,
            ),
        )
    )


def _candidate_document(
    score: TraceCandidateScore, rank: int | None
) -> dict[str, object]:
    return {
        "half_life": score.candidate.half_life,
        "x_half_life": score.candidate.x_half_life,
        "f_half_life": score.candidate.f_half_life,
        "status": "accepted" if score.accepted else "rejected",
        "rejection_reasons": list(score.rejection_reasons),
        "bundle_scores": [asdict(item) for item in score.bundle_scores],
        "mean_validation_nll": score.mean_validation_nll,
        "mean_validation_accuracy": score.mean_validation_accuracy,
        "mean_recurrent_update_ratio_p99": score.mean_recurrent_update_ratio_p99,
        "rank": rank,
    }


def _run_coordinate(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    coordinate: CoordinateName,
    fixed_half_life: float,
    runner: CommandRunner,
    progress: ProgressReporter,
) -> tuple[TraceCandidateScore, pathlib.Path]:
    candidates = coordinate_candidates(grid, coordinate, fixed_half_life)
    scores = tuple(
        evaluate_trace_candidate(settings, grid, item, runner, progress)
        for item in candidates
    )
    ranking = rank_trace_scores(scores)
    ranks = {item.candidate.index: rank + 1 for rank, item in enumerate(ranking)}
    path = settings.output_directory / f"{grid.horizon}-{coordinate}-summary.json"
    document: dict[str, object] = {
        "schema_version": TRACE_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_trace_coordinate_summary",
        "development_only": True,
        "sealed_test": False,
        "horizon": grid.horizon,
        "updates": grid.updates,
        "coordinate": coordinate,
        "fixed_half_life": fixed_half_life,
        "settings": trace_search_settings_document(settings),
        "candidate_half_lives": list(grid.half_lives),
        "ranking": [item.candidate.half_life for item in ranking],
        "selected_half_life": (
            ranking[0].candidate.half_life if ranking else None
        ),
        "candidates": [
            _candidate_document(item, ranks.get(item.candidate.index))
            for item in scores
        ],
    }
    _write_json(path, document)
    if not ranking:
        raise RuntimeError(
            f"trace search failed closed: {grid.horizon} {coordinate} has no "
            "valid candidates"
        )
    return ranking[0], path


def run_development_trace_search(
    settings: TraceSearchSettings,
    *,
    runner: CommandRunner = run_subprocess,
    progress: ProgressReporter = _print_progress,
) -> dict[str, object]:
    """Run X then F coordinate selection independently for every horizon."""
    selections: dict[str, object] = {}
    for grid in HORIZON_TRACE_GRIDS:
        selected_x, x_path = _run_coordinate(
            settings,
            grid,
            "x",
            grid.provisional_f,
            runner,
            progress,
        )
        selected_f, f_path = _run_coordinate(
            settings,
            grid,
            "f",
            selected_x.candidate.half_life,
            runner,
            progress,
        )
        selections[grid.horizon] = {
            "updates": grid.updates,
            "trace_half_life_x_steps": selected_x.candidate.half_life,
            "trace_half_life_f_steps": selected_f.candidate.half_life,
            "x_summary": x_path.name,
            "f_summary": f_path.name,
        }
    final: dict[str, object] = {
        "schema_version": TRACE_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_trace_half_life_selection",
        "development_only": True,
        "sealed_test": False,
        "settings": trace_search_settings_document(settings),
        "selections": selections,
    }
    _write_json(settings.output_directory / "selected-trace-half-lives.json", final)
    return final
