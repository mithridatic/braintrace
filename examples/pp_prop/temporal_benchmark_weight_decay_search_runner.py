"""Resumable execution and selection for recurrent-weight-decay search."""

from __future__ import annotations

import json
import pathlib
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_runner import (
    obtain_bundle_score,
    raw_result_path,
    run_subprocess,
)
from temporal_benchmark_search_selection import BundleScore, RunEvidenceError
from temporal_benchmark_weight_decay_search_config import (
    WEIGHT_DECAY_SEARCH_SCHEMA_VERSION,
    WEIGHT_DECAY_SEARCH_STAGE,
    RecurrentWeightDecayCandidate,
    WeightDecaySearchSettings,
    candidate_search_settings,
    ordered_weight_decay_candidates,
    weight_decay_run_candidate,
    weight_decay_search_settings_document,
)

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class WeightDecayCandidateScore:
    """Store bundle evidence and aggregate metrics for one decay."""

    candidate: RecurrentWeightDecayCandidate
    bundle_scores: tuple[BundleScore, ...]
    rejection_reasons: tuple[str, ...]
    mean_validation_nll: float | None
    mean_validation_accuracy: float | None
    mean_recurrent_update_ratio_p99: float | None

    @property
    def accepted(self) -> bool:
        """Return whether every development bundle passed strict validation."""
        return not self.rejection_reasons


def _print_progress(message: str) -> None:
    print(message, flush=True)


def evaluate_weight_decay_candidate(
    settings: WeightDecaySearchSettings,
    candidate: RecurrentWeightDecayCandidate,
    runner: CommandRunner,
    progress: ProgressReporter,
) -> WeightDecayCandidateScore:
    """Evaluate every bundle and reject a decay when any bundle is invalid."""
    runner_settings = candidate_search_settings(settings, candidate)
    run_candidate = weight_decay_run_candidate(candidate)
    scores: list[BundleScore] = []
    rejections: list[str] = []
    for bundle_id in DEVELOPMENT_BUNDLES:
        path = raw_result_path(
            runner_settings, WEIGHT_DECAY_SEARCH_STAGE, run_candidate, bundle_id
        )
        mode = "reuse" if path.is_file() else "run"
        started = time.perf_counter()
        progress(
            f"weight_decay={candidate.weight_decay} bundle={bundle_id} "
            f"mode={mode} start"
        )
        try:
            score = obtain_bundle_score(
                runner_settings,
                WEIGHT_DECAY_SEARCH_STAGE,
                run_candidate,
                bundle_id,
                runner,
            )
            scores.append(score)
            progress(
                f"weight_decay={candidate.weight_decay} bundle={bundle_id} "
                f"mode={mode} result=accepted "
                f"wall_seconds={time.perf_counter() - started:.3f}"
            )
        except RunEvidenceError as error:
            rejections.append(f"{bundle_id}: {error}")
            progress(
                f"weight_decay={candidate.weight_decay} bundle={bundle_id} "
                f"mode={mode} result=rejected reason={error}"
            )
    if rejections:
        return WeightDecayCandidateScore(
            candidate, tuple(scores), tuple(rejections), None, None, None
        )
    return WeightDecayCandidateScore(
        candidate=candidate,
        bundle_scores=tuple(scores),
        rejection_reasons=(),
        mean_validation_nll=statistics.fmean(
            score.validation_nll for score in scores
        ),
        mean_validation_accuracy=statistics.fmean(
            score.validation_accuracy for score in scores
        ),
        mean_recurrent_update_ratio_p99=statistics.fmean(
            score.recurrent_update_ratio_p99 for score in scores
        ),
    )


def rank_weight_decay_scores(
    scores: Sequence[WeightDecayCandidateScore],
) -> tuple[WeightDecayCandidateScore, ...]:
    """Rank valid decays by NLL, accuracy, recurrent ratio, then decay."""
    return tuple(
        sorted(
            (score for score in scores if score.accepted),
            key=lambda score: (
                float(score.mean_validation_nll),
                -float(score.mean_validation_accuracy),
                float(score.mean_recurrent_update_ratio_p99),
                score.candidate.weight_decay,
            ),
        )
    )


def _candidate_document(
    score: WeightDecayCandidateScore, rank: int | None
) -> dict[str, object]:
    return {
        "index": score.candidate.index,
        "recurrent_weight_decay": score.candidate.weight_decay,
        "status": "accepted" if score.accepted else "rejected",
        "rejection_reasons": list(score.rejection_reasons),
        "bundle_scores": [asdict(bundle) for bundle in score.bundle_scores],
        "mean_validation_nll": score.mean_validation_nll,
        "mean_validation_accuracy": score.mean_validation_accuracy,
        "mean_recurrent_update_ratio_p99": (
            score.mean_recurrent_update_ratio_p99
        ),
        "rank": rank,
    }


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def run_development_weight_decay_search(
    settings: WeightDecaySearchSettings,
    *,
    runner: CommandRunner = run_subprocess,
    progress: ProgressReporter = _print_progress,
) -> dict[str, object]:
    """Run or resume all decay candidates and write summary and winner."""
    scores = tuple(
        evaluate_weight_decay_candidate(settings, candidate, runner, progress)
        for candidate in ordered_weight_decay_candidates()
    )
    ranking = rank_weight_decay_scores(scores)
    ranks = {
        score.candidate.index: index + 1 for index, score in enumerate(ranking)
    }
    summary: dict[str, object] = {
        "schema_version": WEIGHT_DECAY_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_weight_decay_search_summary",
        "development_only": True,
        "sealed_test": False,
        "candidate_weight_decays": [
            item.weight_decay for item in ordered_weight_decay_candidates()
        ],
        "valid_candidate_count": len(ranking),
        "settings": weight_decay_search_settings_document(settings),
        "ranking": [score.candidate.weight_decay for score in ranking],
        "candidates": [
            _candidate_document(score, ranks.get(score.candidate.index))
            for score in scores
        ],
    }
    _write_json(settings.output_directory / "summary.json", summary)
    if not ranking:
        raise RuntimeError("weight-decay search failed closed: no valid candidates")
    winner: dict[str, object] = {
        "schema_version": WEIGHT_DECAY_SEARCH_SCHEMA_VERSION,
        "kind": "temporal_credit_weight_decay_search_winner",
        "development_only": True,
        "sealed_test": False,
        "summary": "summary.json",
        "settings": weight_decay_search_settings_document(settings),
        "winner": _candidate_document(ranking[0], 1),
    }
    _write_json(settings.output_directory / "winner.json", winner)
    return winner
