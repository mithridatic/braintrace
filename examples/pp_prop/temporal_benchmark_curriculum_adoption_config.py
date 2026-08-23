"""Fixed configuration for development-only curriculum adoption evidence."""

from __future__ import annotations

import math
import pathlib
from dataclasses import asdict, dataclass

from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    GradientClipNorms,
    LearningRates,
    TemporalBenchmarkConfig,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES

CURRICULUM_ADOPTION_SCHEMA_VERSION = 1
CURRICULUM_BUNDLE_KIND = "temporal_credit_curriculum_comparison_bundle"
CURRICULUM_ADOPTION_KIND = "temporal_credit_curriculum_adoption"


@dataclass(frozen=True)
class CurriculumAdoptionSettings:
    """Configure paired development execution and exact resume provenance."""

    source_root: pathlib.Path
    output_directory: pathlib.Path
    experiment_script: pathlib.Path
    manifest_path: pathlib.Path
    python_executable: str
    container_image_digest: str
    source_commit: str
    source_dirty: bool
    example15_accuracy_change: float
    device: str = "gpu"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32
    evaluation_interval: int = 50
    gain: float = 0.8
    learning_rates: LearningRates = LearningRates(0.01, 0.001, 0.001)
    recurrent_weight_decay: float = 0.0
    gradient_clip_norms: GradientClipNorms = GradientClipNorms()
    trace_half_lives: CurriculumTraceHalfLives = CurriculumTraceHalfLives()

    def __post_init__(self) -> None:
        """Reject incomplete provenance and invalid static-control evidence."""
        for name in ("python_executable", "container_image_digest", "source_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required for curriculum adoption. Fix the input condition named in the error, then rerun the operation.")
        if not math.isfinite(self.example15_accuracy_change):
            raise ValueError("example15_accuracy_change must be finite")
        expected_curriculum_config(self, DEVELOPMENT_BUNDLES[0])


def expected_curriculum_config(
    settings: CurriculumAdoptionSettings, bundle_id: str
) -> TemporalBenchmarkConfig:
    """Return the exact unsealed curriculum configuration for one bundle."""
    long_trace = settings.trace_half_lives.long
    return TemporalBenchmarkConfig(
        bundle_id=bundle_id,
        arm="all_pp_prop",
        horizon="long",
        neurons=settings.neurons,
        degree=settings.degree,
        batch_size=settings.batch_size,
        updates=800,
        evaluation_interval=settings.evaluation_interval,
        gain=settings.gain,
        trace_half_life_x_steps=long_trace.x,
        trace_half_life_f_steps=long_trace.f,
        curriculum_trace_half_lives=settings.trace_half_lives,
        gradient_clip_norms=settings.gradient_clip_norms,
        recurrent_weight_decay=settings.recurrent_weight_decay,
        learning_rates=settings.learning_rates,
        curriculum=True,
        gradient_evidence=False,
        sealed_test=False,
        allow_dirty=True,
        device=settings.device,
    )


def selected_config_from_benchmark(
    config: TemporalBenchmarkConfig,
) -> dict[str, object]:
    """Extract the selected scientific configuration from one paired run."""
    return {
        "gain": config.gain,
        "learning_rates": asdict(config.learning_rates),
        "recurrent_weight_decay": config.recurrent_weight_decay,
        "gradient_clip_norms": asdict(config.gradient_clip_norms),
        "trace_half_lives": asdict(config.curriculum_trace_half_lives),
    }


def selected_config_document(settings: CurriculumAdoptionSettings) -> dict[str, object]:
    """Return the exact selected scientific configuration consumed downstream."""
    return selected_config_from_benchmark(
        expected_curriculum_config(settings, DEVELOPMENT_BUNDLES[0])
    )


def _clip_argument(value: float | None) -> str:
    return "disabled" if value is None else repr(value)


def curriculum_bundle_command(
    settings: CurriculumAdoptionSettings,
    bundle_id: str,
    raw_path: pathlib.Path,
) -> tuple[str, ...]:
    """Return one isolated paired-development subprocess command."""
    config = expected_curriculum_config(settings, bundle_id)
    traces = settings.trace_half_lives
    command = [
        settings.python_executable,
        str(settings.experiment_script),
        "--manifest",
        str(settings.manifest_path),
        "--bundle-id",
        bundle_id,
        "--neurons",
        str(config.neurons),
        "--degree",
        str(config.degree),
        "--batch-size",
        str(config.batch_size),
        "--evaluation-interval",
        str(config.evaluation_interval),
        "--gain",
        repr(config.gain),
        "--readout-learning-rate",
        repr(config.learning_rates.readout),
        "--feedforward-learning-rate",
        repr(config.learning_rates.feedforward),
        "--recurrent-learning-rate",
        repr(config.learning_rates.recurrent),
        "--recurrent-weight-decay",
        repr(config.recurrent_weight_decay),
    ]
    for horizon in ("short", "medium", "long"):
        pair = traces.for_horizon(horizon)
        command.extend(
            (
                f"--{horizon}-trace-half-life-x-steps",
                repr(pair.x),
                f"--{horizon}-trace-half-life-f-steps",
                repr(pair.f),
            )
        )
    command.extend(
        (
            "--readout-clip-norm",
            _clip_argument(config.gradient_clip_norms.readout),
            "--feedforward-clip-norm",
            _clip_argument(config.gradient_clip_norms.feedforward),
            "--recurrent-clip-norm",
            _clip_argument(config.gradient_clip_norms.recurrent),
            "--device",
            config.device,
            "--json-output",
            str(raw_path),
        )
    )
    return tuple(command)


def curriculum_adoption_settings_document(
    settings: CurriculumAdoptionSettings,
) -> dict[str, object]:
    """Return stable construction and provenance settings for the summary."""
    return {
        "experiment_script": str(settings.experiment_script),
        "manifest_path": str(settings.manifest_path),
        "provenance": {
            "source_commit": settings.source_commit,
            "source_dirty": settings.source_dirty,
            "container_image_digest": settings.container_image_digest,
        },
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        "device": settings.device,
        "neurons": settings.neurons,
        "degree": settings.degree,
        "batch_size": settings.batch_size,
        "evaluation_interval": settings.evaluation_interval,
        "selected_config": selected_config_document(settings),
    }
