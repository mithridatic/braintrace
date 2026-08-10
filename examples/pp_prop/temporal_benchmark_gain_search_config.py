"""Fixed coordinate-search configuration for recurrent gain selection."""

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

GAIN_SEARCH_SCHEMA_VERSION = 1
DEVELOPMENT_GAIN_VALUES = (0.5, 0.8, 1.0, 1.2)
GAIN_SEARCH_STAGE = SearchStage(number=1, updates=800, promotion_count=1)
FIXED_GAIN_SEARCH_RATES = (0.003, 0.001, 0.0003)


@dataclass(frozen=True)
class GainCandidate:
    """Identify one recurrent-gain coordinate candidate."""

    index: int
    gain: float


@dataclass(frozen=True)
class GainSearchSettings:
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
        candidate_search_settings(self, ordered_gain_candidates()[0])


def ordered_gain_candidates() -> tuple[GainCandidate, ...]:
    """Return gain candidates in their deterministic final tie-break order."""
    return tuple(
        GainCandidate(index, gain)
        for index, gain in enumerate(DEVELOPMENT_GAIN_VALUES)
    )


def gain_run_candidate(candidate: GainCandidate) -> LearningRateCandidate:
    """Adapt a gain identity to the generic fixed-learning-rate run helper."""
    return LearningRateCandidate(candidate.index, *FIXED_GAIN_SEARCH_RATES)


def candidate_search_settings(
    settings: GainSearchSettings, candidate: GainCandidate
) -> SearchSettings:
    """Build generic subprocess settings with every non-gain knob fixed."""
    return SearchSettings(
        source_root=settings.source_root,
        output_directory=settings.output_directory,
        benchmark_script=settings.benchmark_script,
        manifest_path=settings.manifest_path,
        python_executable=settings.python_executable,
        container_image_digest=settings.container_image_digest,
        source_commit=settings.source_commit,
        search_kind="gain",
        device=settings.device,
        neurons=settings.neurons,
        degree=settings.degree,
        batch_size=settings.batch_size,
        gain=candidate.gain,
        trace_half_life_x_steps=60.0,
        trace_half_life_f_steps=60.0,
        recurrent_weight_decay=0.0,
        gradient_clip_norms=GradientClipNorms(1.0, 1.0, 1.0),
    )


def expected_gain_benchmark_config(
    settings: GainSearchSettings,
    candidate: GainCandidate,
    bundle_id: str,
) -> TemporalBenchmarkConfig:
    """Return the complete direct-long configuration for one gain raw run."""
    return expected_benchmark_config(
        candidate_search_settings(settings, candidate),
        gain_run_candidate(candidate),
        bundle_id,
        GAIN_SEARCH_STAGE.updates,
    )


def gain_search_settings_document(
    settings: GainSearchSettings,
) -> dict[str, object]:
    """Return the fixed settings and provenance recorded in gain summaries."""
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
            "updates": GAIN_SEARCH_STAGE.updates,
            "learning_rates": {
                "readout": FIXED_GAIN_SEARCH_RATES[0],
                "feedforward": FIXED_GAIN_SEARCH_RATES[1],
                "recurrent": FIXED_GAIN_SEARCH_RATES[2],
            },
            "trace_half_life_x_steps": 60.0,
            "trace_half_life_f_steps": 60.0,
            "gradient_clip_norms": asdict(GradientClipNorms()),
            "recurrent_weight_decay": 0.0,
            "curriculum": False,
            "gradient_evidence": False,
            "sealed_test": False,
        },
    }
