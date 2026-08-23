"""Pure configuration and command planning for the optimizer search."""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass

from temporal_benchmark_config import (
    GradientClipNorms,
    LearningRates,
    TemporalBenchmarkConfig,
)

SEARCH_SCHEMA_VERSION = 1
DEVELOPMENT_BUNDLES = (
    "split0-topology0-weight0",
    "split1-topology0-weight0",
    "split2-topology0-weight0",
)
ORDERED_LEARNING_RATE_GRID = tuple(
    (readout, feedforward, recurrent)
    for readout in (1e-3, 3e-3, 1e-2)
    for feedforward in (3e-4, 1e-3, 3e-3)
    for recurrent in (1e-4, 3e-4, 1e-3)
)


@dataclass(frozen=True)
class SearchStage:
    """Define one successive-halving stage."""

    number: int
    updates: int
    promotion_count: int


SEARCH_STAGES = (
    SearchStage(1, 100, 9),
    SearchStage(2, 300, 3),
    SearchStage(3, 800, 1),
)


@dataclass(frozen=True)
class LearningRateCandidate:
    """Associate one ordered-grid identity with three peak learning rates."""

    grid_index: int
    readout: float
    feedforward: float
    recurrent: float

    @property
    def learning_rates(self) -> LearningRates:
        """Return the candidate as the benchmark configuration value."""
        return LearningRates(self.readout, self.feedforward, self.recurrent)


@dataclass(frozen=True)
class SearchSettings:
    """Configure fixed, development-only Example 17 subprocess runs."""

    source_root: pathlib.Path
    output_directory: pathlib.Path
    benchmark_script: pathlib.Path
    manifest_path: pathlib.Path
    python_executable: str
    container_image_digest: str
    source_commit: str
    search_kind: str = "optimizer"
    device: str = "gpu"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32
    gain: float = 0.8
    trace_half_life_x_steps: float = 60.0
    trace_half_life_f_steps: float = 60.0
    recurrent_weight_decay: float = 0.0
    gradient_clip_norms: GradientClipNorms = GradientClipNorms()

    def __post_init__(self) -> None:
        for name in ("container_image_digest", "source_commit", "search_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required for development search. Fix the input condition named in the error, then rerun the operation.")


def ordered_candidates() -> tuple[LearningRateCandidate, ...]:
    """Return the fixed 27-candidate grid in deterministic specification order."""
    return tuple(
        LearningRateCandidate(index, *rates)
        for index, rates in enumerate(ORDERED_LEARNING_RATE_GRID)
    )


def expected_benchmark_config(
    settings: SearchSettings,
    candidate: LearningRateCandidate,
    bundle_id: str,
    updates: int,
) -> TemporalBenchmarkConfig:
    """Build the complete Example 17 configuration expected in one raw file."""
    return TemporalBenchmarkConfig(
        bundle_id=bundle_id,
        arm="all_pp_prop",
        horizon="long",
        neurons=settings.neurons,
        degree=settings.degree,
        batch_size=settings.batch_size,
        updates=updates,
        gain=settings.gain,
        trace_half_life_x_steps=settings.trace_half_life_x_steps,
        trace_half_life_f_steps=settings.trace_half_life_f_steps,
        gradient_clip_norms=settings.gradient_clip_norms,
        recurrent_weight_decay=settings.recurrent_weight_decay,
        learning_rates=candidate.learning_rates,
        curriculum=False,
        gradient_evidence=False,
        sealed_test=False,
        allow_dirty=True,
        device=settings.device,
    )


def _clip_argument(value: float | None) -> str:
    return "disabled" if value is None else repr(value)


def benchmark_command(
    settings: SearchSettings,
    candidate: LearningRateCandidate,
    bundle_id: str,
    updates: int,
    raw_path: pathlib.Path,
) -> tuple[str, ...]:
    """Return one isolated validation-only Example 17 subprocess command."""
    config = expected_benchmark_config(settings, candidate, bundle_id, updates)
    return (
        settings.python_executable,
        str(settings.benchmark_script),
        "--manifest",
        str(settings.manifest_path),
        "--bundle-id",
        config.bundle_id,
        "--arm",
        config.arm,
        "--horizon",
        config.horizon,
        "--neurons",
        str(config.neurons),
        "--degree",
        str(config.degree),
        "--batch-size",
        str(config.batch_size),
        "--updates",
        str(config.updates),
        "--gain",
        repr(config.gain),
        "--trace-half-life-x-steps",
        repr(config.trace_half_life_x_steps),
        "--trace-half-life-f-steps",
        repr(config.trace_half_life_f_steps),
        "--readout-learning-rate",
        repr(config.learning_rates.readout),
        "--feedforward-learning-rate",
        repr(config.learning_rates.feedforward),
        "--recurrent-learning-rate",
        repr(config.learning_rates.recurrent),
        "--readout-clip-norm",
        _clip_argument(config.gradient_clip_norms.readout),
        "--feedforward-clip-norm",
        _clip_argument(config.gradient_clip_norms.feedforward),
        "--recurrent-clip-norm",
        _clip_argument(config.gradient_clip_norms.recurrent),
        "--recurrent-weight-decay",
        repr(config.recurrent_weight_decay),
        "--device",
        config.device,
        "--no-curriculum",
        "--no-gradient-evidence",
        "--no-sealed-test",
        "--allow-dirty",
        "--json-output",
        str(raw_path),
    )


def settings_document(settings: SearchSettings) -> dict[str, object]:
    """Return stable search settings for stage and winner summaries."""
    return {
        "benchmark_script": str(settings.benchmark_script),
        "manifest_path": str(settings.manifest_path),
        "container_image_digest": settings.container_image_digest,
        "source_commit": settings.source_commit,
        "search_kind": settings.search_kind,
        "device": settings.device,
        "neurons": settings.neurons,
        "degree": settings.degree,
        "batch_size": settings.batch_size,
        "gain": settings.gain,
        "trace_half_life_x_steps": settings.trace_half_life_x_steps,
        "trace_half_life_f_steps": settings.trace_half_life_f_steps,
        "recurrent_weight_decay": settings.recurrent_weight_decay,
        "gradient_clip_norms": asdict(settings.gradient_clip_norms),
        "development_bundles": list(DEVELOPMENT_BUNDLES),
    }
