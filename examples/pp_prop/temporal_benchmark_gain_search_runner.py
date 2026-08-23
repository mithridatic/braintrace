"""Resumable subprocess execution and selection for development gain search."""

from __future__ import annotations

import msgspec_json
import pathlib
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from temporal_benchmark_gain_search_config import (
    GAIN_SEARCH_SCHEMA_VERSION,
    GAIN_SEARCH_STAGE,
    GainCandidate,
    GainSearchSettings,
    candidate_search_settings,
    gain_run_candidate,
    gain_search_settings_document,
    ordered_gain_candidates,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_runner import (
    obtain_bundle_score,
    raw_result_path,
    run_subprocess,
)
from temporal_benchmark_search_selection import BundleScore, RunEvidenceError

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class GainCandidateScore:
    """Store bundle evidence and aggregate metrics for one gain."""

    candidate: GainCandidate
    bundle_scores: tuple[BundleScore, ...]
    rejection_reasons: tuple[str, ...]
    mean_validation_accuracy: float | None
    mean_validation_nll: float | None

    @property
    def accepted(self) -> bool:
        """Return whether every development bundle passed strict validation."""
        return not self.rejection_reasons


def _print_progress(message: str) -> None:
    print(message, flush=True)


def evaluate_gain_candidate(
    settings: GainSearchSettings,
    candidate: GainCandidate,
    runner: CommandRunner,
    progress: ProgressReporter,
) -> GainCandidateScore:
    """Evaluate all three bundles and reject a gain if any bundle is invalid."""
    runner_settings = candidate_search_settings(settings, candidate)
    run_candidate = gain_run_candidate(candidate)
    scores: list[BundleScore] = []
    rejections: list[str] = []
    for bundle_id in DEVELOPMENT_BUNDLES:
        path = raw_result_path(
            runner_settings, GAIN_SEARCH_STAGE, run_candidate, bundle_id
        )
        mode = "reuse" if path.is_file() else "run"
        started = time.perf_counter()
        progress(f"gain={candidate.gain} bundle={bundle_id} mode={mode} start")
        try:
            score = obtain_bundle_score(
                runner_settings,
                GAIN_SEARCH_STAGE,
                run_candidate,
                bundle_id,
                runner,
            )
            scores.append(score)
            progress(
                f"gain={candidate.gain} bundle={bundle_id} mode={mode} "
                f"result=accepted wall_seconds={time.perf_counter() - started:.3f}"
            )
        except RunEvidenceError as error:
            rejections.append(f"{bundle_id}: {error}")
            progress(
                f"gain={candidate.gain} bundle={bundle_id} mode={mode} "
                f"result=rejected reason={error}"
            )
    if rejections:
        return GainCandidateScore(
            candidate, tuple(scores), tuple(rejections), None, None
        )
    return GainCandidateScore(
        candidate=candidate,
        bundle_scores=tuple(scores),
        rejection_reasons=(),
        mean_validation_accuracy=statistics.fmean(
            score.validation_accuracy for score in scores
        ),
        mean_validation_nll=statistics.fmean(
            score.validation_nll for score in scores
        ),
    )


def rank_gain_scores(
    scores: Sequence[GainCandidateScore],
) -> tuple[GainCandidateScore, ...]:
    """Rank valid gains by accuracy descending, NLL ascending, then gain."""
    return tuple(
        sorted(
            (score for score in scores if score.accepted),
            key=lambda score: (
                -float(score.mean_validation_accuracy),
                float(score.mean_validation_nll),
                score.candidate.gain,
            ),
        )
    )


def _candidate_document(
    score: GainCandidateScore, rank: int | None
) -> dict[str, object]:
    return {
        "index": score.candidate.index,
        "gain": score.candidate.gain,
        "status": "accepted" if score.accepted else "rejected",
        "rejection_reasons": list(score.rejection_reasons),
        "bundle_scores": [asdict(bundle) for bundle in score.bundle_scores],
        "mean_validation_accuracy": score.mean_validation_accuracy,
        "mean_validation_nll": score.mean_validation_nll,
        "rank": rank,
    }


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def run_development_gain_search(
    settings: GainSearchSettings,
    *,
    runner: CommandRunner = run_subprocess,
    progress: ProgressReporter = _print_progress,
) -> dict[str, object]:
    """Run or resume all fixed gain candidates and write summary and winner."""
    scores = tuple(
        evaluate_gain_candidate(settings, candidate, runner, progress)
        for candidate in ordered_gain_candidates()
    )
    ranking = rank_gain_scores(scores)
    ranks = {
        score.candidate.index: index + 1 for index, score in enumerate(ranking)
    }
    summary: dict[str, object] = {
        "schema_version": GAIN_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_gain_search_summary",
        "development_only": True,
        "sealed_test": False,
        "candidate_gains": [item.gain for item in ordered_gain_candidates()],
        "valid_candidate_count": len(ranking),
        "settings": gain_search_settings_document(settings),
        "ranking": [score.candidate.gain for score in ranking],
        "candidates": [
            _candidate_document(score, ranks.get(score.candidate.index))
            for score in scores
        ],
    }
    _write_json(settings.output_directory / "summary.json", summary)
    if not ranking:
        raise RuntimeError("Gain search failed closed: no valid candidates. Correct the reported inputs, then retry the operation.")
    winner: dict[str, object] = {
        "schema_version": GAIN_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_gain_search_winner",
        "development_only": True,
        "sealed_test": False,
        "summary": "summary.json",
        "settings": gain_search_settings_document(settings),
        "winner": _candidate_document(ranking[0], 1),
    }
    _write_json(settings.output_directory / "winner.json", winner)
    return winner
