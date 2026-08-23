"""Validated configuration for the delayed-cue temporal-credit benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

HorizonName = Literal["short", "medium", "long"]
ArmName = Literal[
    "readout_only",
    "feedforward_readout_recurrence_zero",
    "recurrent_readout",
    "all_pp_prop",
    "no_recurrent_module",
    "frozen_random_recurrence",
    "all_bptt",
]

ARMS: tuple[ArmName, ...] = (
    "readout_only",
    "feedforward_readout_recurrence_zero",
    "recurrent_readout",
    "all_pp_prop",
    "no_recurrent_module",
    "frozen_random_recurrence",
    "all_bptt",
)


@dataclass(frozen=True)
class HorizonSpec:
    """Define cue, silent, and supervised response durations."""

    cue_steps: int
    silent_steps: int
    response_steps: int = 4

    @property
    def total_steps(self) -> int:
        """Return the full trial duration."""
        return self.cue_steps + self.silent_steps + self.response_steps


HORIZONS: dict[HorizonName, HorizonSpec] = {
    "short": HorizonSpec(4, 2),
    "medium": HorizonSpec(4, 22),
    "long": HorizonSpec(4, 92),
}

MEMBRANE_TAU_MS = 0.5
FEEDFORWARD_SYNAPSE_TAU_MS = 0.5
RECURRENT_SYNAPSE_TAU_MS = 3.0
READOUT_TAU_MS = 0.5


@dataclass(frozen=True)
class SplitSizes:
    """Define balanced trial counts for each data split."""

    train: int = 1024
    validation: int = 256
    test: int = 512

    def __post_init__(self) -> None:
        for name, count in asdict(self).items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} size must be a positive integer. Set {name} size to a positive integer.")
            if count % 2:
                raise ValueError(f"{name} size must be balanced and therefore even. Set {name} size to balanced and therefore even.")


@dataclass(frozen=True)
class LearningRates:
    """Peak learning rates for disjoint Adam parameter groups."""

    readout: float = 3e-3
    feedforward: float = 1e-3
    recurrent: float = 3e-4

    def __post_init__(self) -> None:
        for name, rate in asdict(self).items():
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"{name} learning rate must be finite and positive. Set {name} learning rate to a finite positive value.")


@dataclass(frozen=True)
class GradientClipNorms:
    """Define independent gradient clipping for each parameter group."""

    readout: float | None = 1.0
    feedforward: float | None = 1.0
    recurrent: float | None = 1.0

    def __post_init__(self) -> None:
        for name, clip_norm in asdict(self).items():
            if clip_norm is None:
                continue
            if isinstance(clip_norm, bool) or not math.isfinite(clip_norm):
                raise ValueError(
                    f"{name} clip norm must be finite and positive or None. Set {name} clip norm to a finite positive value or None."
                )
            if clip_norm <= 0.0:
                raise ValueError(
                    f"{name} clip norm must be finite and positive or None. Set {name} clip norm to a finite positive value or None."
                )


@dataclass(frozen=True)
class TraceHalfLives:
    """Define independent X and F trace half-lives for one horizon."""

    x: float
    f: float

    def __post_init__(self) -> None:
        for name, half_life in asdict(self).items():
            if not math.isfinite(half_life) or half_life <= 0.0:
                raise ValueError(f"{name} trace half-life must be finite and positive. Set {name} trace half-life to a finite positive value.")


@dataclass(frozen=True)
class CurriculumTraceHalfLives:
    """Define independently selected X/F trace pairs for every horizon."""

    short: TraceHalfLives = TraceHalfLives(10.0, 10.0)
    medium: TraceHalfLives = TraceHalfLives(20.0, 20.0)
    long: TraceHalfLives = TraceHalfLives(60.0, 60.0)

    def for_horizon(self, horizon: HorizonName) -> TraceHalfLives:
        """Return the selected trace pair for ``horizon``."""
        return {"short": self.short, "medium": self.medium, "long": self.long}[horizon]


@dataclass(frozen=True)
class TemporalBenchmarkConfig:
    """Define one reproducible Example 17 arm run."""

    bundle_id: str = "split0-topology0-weight0"
    arm: ArmName = "all_pp_prop"
    horizon: HorizonName = "long"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32
    updates: int = 800
    evaluation_interval: int = 50
    dt_seconds: float = 0.001
    cue_rate_hz: float = 200.0
    go_rate_hz: float = 200.0
    gain: float = 0.8
    trace_half_life_x_steps: float = 60.0
    trace_half_life_f_steps: float = 60.0
    curriculum_trace_half_lives: CurriculumTraceHalfLives = CurriculumTraceHalfLives()
    gradient_clip_norms: GradientClipNorms = GradientClipNorms()
    recurrent_weight_decay: float = 0.0
    learning_rates: LearningRates = LearningRates()
    split_sizes: SplitSizes = SplitSizes()
    curriculum: bool = False
    gradient_evidence: bool = False
    sealed_test: bool = False
    allow_dirty: bool = False
    device: Literal["auto", "cpu", "gpu"] = "gpu"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"Arm must be one of {ARMS}. Set Arm to one of {ARMS}.")
        if self.horizon not in HORIZONS:
            raise ValueError(f"Unknown horizon: {self.horizon}. Set the named field to one of the supported values, then rerun the operation.")
        for name in ("neurons", "degree", "batch_size", "updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer. Set {name} to a positive integer.")
        if self.degree >= self.neurons:
            raise ValueError("Degree must be smaller than neurons to forbid self-loops. Set Degree to smaller than neurons to forbid self-loops.")
        if self.split_sizes.train % self.batch_size:
            raise ValueError("batch_size must divide the training split. Set batch_size to divide the training split.")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive. Set evaluation_interval to a positive value.")
        for name in ("dt_seconds", "cue_rate_hz", "go_rate_hz", "gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive. Set {name} to a finite positive value.")
        for name in ("trace_half_life_x_steps", "trace_half_life_f_steps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive. Set {name} to a finite positive value.")
        if self.recurrent_weight_decay not in {0.0, 1e-5, 1e-4}:
            raise ValueError("recurrent_weight_decay is outside the sealed grid. Set the named field to a value in the stated range, then rerun the operation.")
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("Device must be auto, cpu, or gpu. Set Device to auto, cpu, or gpu.")


def config_to_dict(config: TemporalBenchmarkConfig) -> dict[str, object]:
    """Return a JSON-compatible benchmark configuration."""
    return cast(dict[str, object], asdict(config))


def half_life_decay(half_life_steps: float) -> float:
    """Convert a trace half-life into its per-step decay."""
    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ValueError("half_life_steps must be finite and positive. Set half_life_steps to a finite positive value.")
    return 2.0 ** (-1.0 / half_life_steps)
