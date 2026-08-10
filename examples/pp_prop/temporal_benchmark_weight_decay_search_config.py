"""Fixed configuration for recurrent-weight-decay coordinate search."""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass

from temporal_benchmark_config import GradientClipNorms, TemporalBenchmarkConfig
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    LearningRateCandidate,
    SearchSettings,
    SearchStage,
    expected_benchmark_config,
)

WEIGHT_DECAY_SEARCH_SCHEMA_VERSION = 1
DEVELOPMENT_WEIGHT_DECAYS = (0.0, 1e-5, 1e-4)
WEIGHT_DECAY_SEARCH_STAGE = SearchStage(number=1, updates=800, promotion_count=1)
FIXED_WEIGHT_DECAY_SEARCH_RATES = (0.01, 0.001, 0.001)


@dataclass(frozen=True)
class RecurrentWeightDecayCandidate:
    """Identify one recurrent-only weight-decay candidate."""

    index: int
    weight_decay: float


@dataclass(frozen=True)
class WeightDecaySearchSettings:
    """Configure execution and provenance without exposing scientific knobs."""

    source_root: pathlib.Path
    output_directory: pathlib.Path
    benchmark_script: pathlib.Path
    manifest_path: pathlib.Path
    python_executable: str
    container_image_digest: str
    source_commit: str
    device: str = "gpu"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32

    def __post_init__(self) -> None:
        candidate_search_settings(self, ordered_weight_decay_candidates()[0])


def ordered_weight_decay_candidates() -> tuple[RecurrentWeightDecayCandidate, ...]:
    """Return decay candidates in deterministic final tie-break order."""
    return tuple(
        RecurrentWeightDecayCandidate(index, weight_decay)
        for index, weight_decay in enumerate(DEVELOPMENT_WEIGHT_DECAYS)
    )


def weight_decay_run_candidate(
    candidate: RecurrentWeightDecayCandidate,
) -> LearningRateCandidate:
    """Adapt a decay identity to the generic fixed-learning-rate run helper."""
    return LearningRateCandidate(candidate.index, *FIXED_WEIGHT_DECAY_SEARCH_RATES)


def candidate_search_settings(
    settings: WeightDecaySearchSettings,
    candidate: RecurrentWeightDecayCandidate,
) -> SearchSettings:
    """Build generic subprocess settings with every non-decay knob fixed."""
    return SearchSettings(
        source_root=settings.source_root,
        output_directory=settings.output_directory,
        benchmark_script=settings.benchmark_script,
        manifest_path=settings.manifest_path,
        python_executable=settings.python_executable,
        container_image_digest=settings.container_image_digest,
        source_commit=settings.source_commit,
        search_kind="weight_decay",
        device=settings.device,
        neurons=settings.neurons,
        degree=settings.degree,
        batch_size=settings.batch_size,
        gain=0.8,
        trace_half_life_x_steps=60.0,
        trace_half_life_f_steps=60.0,
        recurrent_weight_decay=candidate.weight_decay,
        gradient_clip_norms=GradientClipNorms(1.0, 1.0, 1.0),
    )


def expected_weight_decay_benchmark_config(
    settings: WeightDecaySearchSettings,
    candidate: RecurrentWeightDecayCandidate,
    bundle_id: str,
) -> TemporalBenchmarkConfig:
    """Return the complete direct-long configuration for one decay raw run."""
    return expected_benchmark_config(
        candidate_search_settings(settings, candidate),
        weight_decay_run_candidate(candidate),
        bundle_id,
        WEIGHT_DECAY_SEARCH_STAGE.updates,
    )


def weight_decay_search_settings_document(
    settings: WeightDecaySearchSettings,
) -> dict[str, object]:
    """Return fixed settings and provenance recorded in decay summaries."""
    return {
        "benchmark_script": str(settings.benchmark_script),
        "manifest_path": str(settings.manifest_path),
        "container_image_digest": settings.container_image_digest,
        "source_commit": settings.source_commit,
        "device": settings.device,
        "neurons": settings.neurons,
        "degree": settings.degree,
        "batch_size": settings.batch_size,
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        "fixed_configuration": {
            "arm": "all_pp_prop",
            "horizon": "long",
            "updates": WEIGHT_DECAY_SEARCH_STAGE.updates,
            "gain": 0.8,
            "learning_rates": {
                "readout": FIXED_WEIGHT_DECAY_SEARCH_RATES[0],
                "feedforward": FIXED_WEIGHT_DECAY_SEARCH_RATES[1],
                "recurrent": FIXED_WEIGHT_DECAY_SEARCH_RATES[2],
            },
            "trace_half_life_x_steps": 60.0,
            "trace_half_life_f_steps": 60.0,
            "gradient_clip_norms": asdict(GradientClipNorms()),
            "curriculum": False,
            "gradient_evidence": False,
            "sealed_test": False,
        },
    }
