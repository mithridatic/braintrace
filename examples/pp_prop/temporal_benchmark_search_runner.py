"""Subprocess isolation, raw-file resume, and search summary reporting."""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    SEARCH_SCHEMA_VERSION,
    SEARCH_STAGES,
    LearningRateCandidate,
    SearchSettings,
    SearchStage,
    benchmark_command,
    expected_benchmark_config,
    ordered_candidates,
    settings_document,
)
from temporal_benchmark_search_selection import (
    BundleScore,
    CandidateScore,
    ResumeConfigurationError,
    RunEvidenceError,
    StabilityEvidenceError,
    rank_candidate_scores,
    score_raw_document,
)

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
ProgressReporter = Callable[[str], None]


def _print_progress(message: str) -> None:
    print(message, flush=True)


class SubprocessRunError(RuntimeError):
    """Retain concise child-process diagnostics for a failure artifact."""

    def __init__(self, completed: subprocess.CompletedProcess[str]):
        self.returncode = completed.returncode
        self.stdout_tail = completed.stdout.strip()[-2000:]
        self.stderr_tail = completed.stderr.strip()[-2000:]
        detail = self.stderr_tail or self.stdout_tail
        super().__init__(f"exit={self.returncode} detail={detail}")


def run_subprocess(command: Sequence[str], source_root: pathlib.Path) -> None:
    """Run one Example 17 child and retain concise failure diagnostics."""
    completed = subprocess.run(
        command,
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SubprocessRunError(completed)


def _reject_json_constant(value: str) -> None:
    raise RunEvidenceError(f"raw result contains non-finite JSON constant {value}")


def load_raw_document(path: pathlib.Path) -> dict[str, Any]:
    """Read strict JSON while rejecting non-finite constants."""
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RunEvidenceError(f"cannot read raw result {path}: {error}") from error
    if not isinstance(document, dict):
        raise RunEvidenceError("raw result must be a JSON object")
    return document


def raw_result_path(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    bundle_id: str,
) -> pathlib.Path:
    """Return the stable resume path for one candidate/bundle subprocess."""
    return (
        settings.output_directory
        / "raw"
        / f"stage-{stage.number:02d}-updates-{stage.updates:04d}"
        / f"candidate-{candidate.grid_index:02d}"
        / f"{bundle_id}.json"
    )


def failure_result_path(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    bundle_id: str,
) -> pathlib.Path:
    """Return the stable diagnostic path for one failed child run."""
    return raw_result_path(settings, stage, candidate, bundle_id).with_suffix(
        ".failure.json"
    )


def partial_result_path(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    bundle_id: str,
) -> pathlib.Path:
    """Return the staging path that cannot be mistaken for resumable evidence."""
    return raw_result_path(settings, stage, candidate, bundle_id).with_suffix(
        ".partial.json"
    )


def _write_failure_artifact(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    bundle_id: str,
    error: Exception,
) -> None:
    expected = expected_benchmark_config(settings, candidate, bundle_id, stage.updates)
    failure: dict[str, object] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "kind": f"temporal_credit_{settings.search_kind}_search_child_failure",
        "development_only": True,
        "sealed_test": False,
        "stage": stage.number,
        "updates": stage.updates,
        "grid_index": candidate.grid_index,
        "bundle_id": bundle_id,
        "config": config_to_dict(expected),
        "provenance": {
            "container_image_digest": settings.container_image_digest,
            "source_commit": settings.source_commit,
        },
        "error_type": type(error).__name__,
        "message": str(error)[-4000:],
    }
    if isinstance(error, SubprocessRunError):
        failure.update(
            {
                "returncode": error.returncode,
                "stdout_tail": error.stdout_tail,
                "stderr_tail": error.stderr_tail,
            }
        )
    _write_json(failure_result_path(settings, stage, candidate, bundle_id), failure)


def obtain_bundle_score(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    bundle_id: str,
    runner: CommandRunner,
) -> BundleScore:
    """Reuse one exact raw run or launch a fresh isolated subprocess."""
    path = raw_result_path(settings, stage, candidate, bundle_id)
    expected = expected_benchmark_config(settings, candidate, bundle_id, stage.updates)
    partial = partial_result_path(settings, stage, candidate, bundle_id)
    partial.unlink(missing_ok=True)
    reused = path.is_file()
    try:
        relative_path = str(path.relative_to(settings.output_directory))
    except ValueError:
        relative_path = str(path)
    if reused:
        try:
            return score_raw_document(
                load_raw_document(path),
                expected,
                relative_path,
                settings.container_image_digest,
                settings.source_commit,
                reused=True,
            )
        except StabilityEvidenceError:
            raise
        except RunEvidenceError as error:
            raise ResumeConfigurationError(
                f"existing final raw result is not safely reusable: {error}"
            ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    command = benchmark_command(settings, candidate, bundle_id, stage.updates, partial)
    try:
        runner(command, settings.source_root)
    except RuntimeError as error:
        _write_failure_artifact(settings, stage, candidate, bundle_id, error)
        raise RunEvidenceError(f"child run failed: {error}") from error
    if not partial.is_file():
        error = RuntimeError(
            f"Example 17 did not write requested staged result: {partial}"
        )
        _write_failure_artifact(settings, stage, candidate, bundle_id, error)
        raise RunEvidenceError(str(error)) from error
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
    except RunEvidenceError as error:
        _write_failure_artifact(settings, stage, candidate, bundle_id, error)
        raise
    partial.replace(path)
    return score


def evaluate_candidate(
    settings: SearchSettings,
    stage: SearchStage,
    candidate: LearningRateCandidate,
    runner: CommandRunner,
    progress: ProgressReporter,
) -> CandidateScore:
    """Evaluate all fixed development bundles and reject on any invalid run."""
    scores: list[BundleScore] = []
    rejections: list[str] = []
    for bundle_id in DEVELOPMENT_BUNDLES:
        mode = (
            "reuse"
            if raw_result_path(settings, stage, candidate, bundle_id).is_file()
            else "run"
        )
        started = time.perf_counter()
        progress(
            f"stage={stage.number} candidate={candidate.grid_index:02d} "
            f"bundle={bundle_id} mode={mode} start"
        )
        try:
            score = obtain_bundle_score(settings, stage, candidate, bundle_id, runner)
            scores.append(score)
            progress(
                f"stage={stage.number} candidate={candidate.grid_index:02d} "
                f"bundle={bundle_id} mode={mode} result=accepted "
                f"wall_seconds={time.perf_counter() - started:.3f} "
                f"nll={score.validation_nll:.8f} "
                f"accuracy={score.validation_accuracy:.8f} "
                f"recurrent_p99={score.recurrent_update_ratio_p99:.8g}"
            )
        except RunEvidenceError as error:
            rejections.append(f"{bundle_id}: {error}")
            progress(
                f"stage={stage.number} candidate={candidate.grid_index:02d} "
                f"bundle={bundle_id} mode={mode} result=rejected "
                f"wall_seconds={time.perf_counter() - started:.3f} reason={error}"
            )
    if rejections:
        return CandidateScore(
            candidate, tuple(scores), tuple(rejections), None, None, None
        )
    return CandidateScore(
        candidate=candidate,
        bundle_scores=tuple(scores),
        rejection_reasons=(),
        mean_validation_nll=statistics.fmean(score.validation_nll for score in scores),
        mean_validation_accuracy=statistics.fmean(
            score.validation_accuracy for score in scores
        ),
        mean_recurrent_update_ratio_p99=statistics.fmean(
            score.recurrent_update_ratio_p99 for score in scores
        ),
    )


def _candidate_document(
    score: CandidateScore,
    rank: int | None,
    promoted: bool,
) -> dict[str, object]:
    return {
        "grid_index": score.candidate.grid_index,
        "learning_rates": asdict(score.candidate.learning_rates),
        "status": "accepted" if score.accepted else "rejected",
        "rejection_reasons": list(score.rejection_reasons),
        "bundle_scores": [asdict(bundle) for bundle in score.bundle_scores],
        "mean_validation_nll": score.mean_validation_nll,
        "mean_validation_accuracy": score.mean_validation_accuracy,
        "mean_recurrent_update_ratio_p99": (score.mean_recurrent_update_ratio_p99),
        "rank": rank,
        "promoted": promoted,
    }


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_stage_summary(
    settings: SearchSettings,
    stage: SearchStage,
    scores: Sequence[CandidateScore],
    ranking: Sequence[CandidateScore],
    promoted: Sequence[CandidateScore],
) -> pathlib.Path:
    ranks = {
        score.candidate.grid_index: index + 1 for index, score in enumerate(ranking)
    }
    promoted_indices = {score.candidate.grid_index for score in promoted}
    document: dict[str, object] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_optimizer_search_stage",
        "development_only": True,
        "sealed_test": False,
        "stage": stage.number,
        "updates": stage.updates,
        "promotion_count": stage.promotion_count,
        "candidate_count": len(scores),
        "valid_candidate_count": len(ranking),
        "settings": settings_document(settings),
        "ranking": [score.candidate.grid_index for score in ranking],
        "promoted_grid_indices": [score.candidate.grid_index for score in promoted],
        "candidates": [
            _candidate_document(
                score,
                ranks.get(score.candidate.grid_index),
                score.candidate.grid_index in promoted_indices,
            )
            for score in scores
        ],
    }
    path = settings.output_directory / f"stage-{stage.number:02d}-summary.json"
    _write_json(path, document)
    return path


def run_development_optimizer_search(
    settings: SearchSettings,
    *,
    runner: CommandRunner = run_subprocess,
    progress: ProgressReporter = _print_progress,
) -> dict[str, object]:
    """Run or resume the three-stage learning-rate successive-halving search."""
    candidates = ordered_candidates()
    stage_paths: list[pathlib.Path] = []
    winner_score: CandidateScore | None = None
    for stage in SEARCH_STAGES:
        scores = tuple(
            evaluate_candidate(settings, stage, candidate, runner, progress)
            for candidate in candidates
        )
        ranking = rank_candidate_scores(scores)
        promoted = ranking[: stage.promotion_count]
        stage_paths.append(
            _write_stage_summary(settings, stage, scores, ranking, promoted)
        )
        if not promoted:
            raise RuntimeError(
                f"optimizer search failed closed: stage {stage.number} has no valid "
                "candidates"
            )
        candidates = tuple(score.candidate for score in promoted)
        winner_score = promoted[0]
    assert winner_score is not None
    winner: dict[str, object] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_optimizer_search_winner",
        "development_only": True,
        "sealed_test": False,
        "settings": settings_document(settings),
        "stage_summaries": [str(path.name) for path in stage_paths],
        "winner": _candidate_document(winner_score, 1, True),
    }
    _write_json(settings.output_directory / "winner.json", winner)
    return winner
