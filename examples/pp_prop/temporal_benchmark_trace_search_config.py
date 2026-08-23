"""Fixed configuration and commands for trace half-life coordinate search."""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass
from typing import Literal

from temporal_benchmark_config import (
    GradientClipNorms,
    HorizonName,
    LearningRates,
    TemporalBenchmarkConfig,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES

TRACE_SEARCH_SCHEMA_VERSION = 1
FIXED_TRACE_SEARCH_RATES = LearningRates(0.01, 0.001, 0.001)
FIXED_TRACE_SEARCH_GAIN = 0.8
FIXED_TRACE_SEARCH_CLIPS = GradientClipNorms(1.0, 1.0, 1.0)
CoordinateName = Literal["x", "f"]


@dataclass(frozen=True)
class HorizonTraceGrid:
    """Define one horizon's direct-training budget and half-life grid."""

    horizon: HorizonName
    updates: int
    half_lives: tuple[float, ...]
    provisional_f: float


HORIZON_TRACE_GRIDS = (
    HorizonTraceGrid("short", 200, (5.0, 10.0), 10.0),
    HorizonTraceGrid("medium", 400, (10.0, 20.0, 30.0), 20.0),
    HorizonTraceGrid("long", 800, (30.0, 60.0, 100.0), 60.0),
)


@dataclass(frozen=True)
class TraceCandidate:
    """Identify one variable coordinate and its fixed companion value."""

    coordinate: CoordinateName
    index: int
    half_life: float
    fixed_half_life: float

    @property
    def x_half_life(self) -> float:
        """Return the X half-life represented by this coordinate candidate."""
        return self.half_life if self.coordinate == "x" else self.fixed_half_life

    @property
    def f_half_life(self) -> float:
        """Return the F half-life represented by this coordinate candidate."""
        return self.half_life if self.coordinate == "f" else self.fixed_half_life


@dataclass(frozen=True)
class TraceSearchSettings:
    """Configure development execution and exact resume provenance."""

    source_root: pathlib.Path
    output_directory: pathlib.Path
    benchmark_script: pathlib.Path
    manifest_path: pathlib.Path
    python_executable: str
    container_image_digest: str
    source_commit: str
    recurrent_weight_decay: float = 0.0
    device: str = "gpu"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32

    def __post_init__(self) -> None:
        for name in ("python_executable", "container_image_digest", "source_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required for trace search. Fix the input condition named in the error, then rerun the operation.")
        expected_trace_benchmark_config(
            self,
            HORIZON_TRACE_GRIDS[0],
            TraceCandidate("x", 0, 5.0, 10.0),
            DEVELOPMENT_BUNDLES[0],
        )


def coordinate_candidates(
    grid: HorizonTraceGrid,
    coordinate: CoordinateName,
    fixed_half_life: float,
) -> tuple[TraceCandidate, ...]:
    """Return candidates in deterministic lower-half-life tie-break order."""
    return tuple(
        TraceCandidate(coordinate, index, value, fixed_half_life)
        for index, value in enumerate(grid.half_lives)
    )


def expected_trace_benchmark_config(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    bundle_id: str,
) -> TemporalBenchmarkConfig:
    """Build the exact validation-only direct-training run configuration."""
    return TemporalBenchmarkConfig(
        bundle_id=bundle_id,
        arm="all_pp_prop",
        horizon=grid.horizon,
        neurons=settings.neurons,
        degree=settings.degree,
        batch_size=settings.batch_size,
        updates=grid.updates,
        gain=FIXED_TRACE_SEARCH_GAIN,
        trace_half_life_x_steps=candidate.x_half_life,
        trace_half_life_f_steps=candidate.f_half_life,
        gradient_clip_norms=FIXED_TRACE_SEARCH_CLIPS,
        recurrent_weight_decay=settings.recurrent_weight_decay,
        learning_rates=FIXED_TRACE_SEARCH_RATES,
        curriculum=False,
        gradient_evidence=False,
        sealed_test=False,
        allow_dirty=True,
        device=settings.device,
    )


def _clip_argument(value: float | None) -> str:
    return "disabled" if value is None else repr(value)


def trace_benchmark_command(
    settings: TraceSearchSettings,
    grid: HorizonTraceGrid,
    candidate: TraceCandidate,
    bundle_id: str,
    raw_path: pathlib.Path,
) -> tuple[str, ...]:
    """Return one isolated Example 17 development subprocess command."""
    config = expected_trace_benchmark_config(settings, grid, candidate, bundle_id)
    command = (
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
    )
    return command + (
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


def trace_search_settings_document(
    settings: TraceSearchSettings,
) -> dict[str, object]:
    """Return all fixed settings and provenance for summary documents."""
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
        "fixed_gain": FIXED_TRACE_SEARCH_GAIN,
        "fixed_learning_rates": asdict(FIXED_TRACE_SEARCH_RATES),
        "fixed_gradient_clip_norms": asdict(FIXED_TRACE_SEARCH_CLIPS),
        "recurrent_weight_decay": settings.recurrent_weight_decay,
        "coordinate_order": ["x", "f"],
    }
