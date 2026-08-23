"""Tests for sealed temporal benchmark configuration."""

import math

import pytest
from hypothesis import given, strategies as st

from temporal_benchmark_config import (
    HORIZONS,
    CurriculumTraceHalfLives,
    GradientClipNorms,
    TemporalBenchmarkConfig,
    TraceHalfLives,
    config_to_dict,
    half_life_decay,
)


def test_horizons_hold_information_and_supervision_constant() -> None:
    assert {name: spec.total_steps for name, spec in HORIZONS.items()} == {
        "short": 10,
        "medium": 30,
        "long": 100,
    }
    assert {spec.cue_steps for spec in HORIZONS.values()} == {4}
    assert {spec.response_steps for spec in HORIZONS.values()} == {4}


@given(st.floats(min_value=0.01, max_value=1000, allow_nan=False))
def test_half_life_conversion_halves_after_requested_steps(half_life: float) -> None:
    decay = half_life_decay(half_life)

    assert math.isclose(decay**half_life, 0.5, rel_tol=1e-12)


def test_no_self_loop_topology_requires_degree_below_neurons() -> None:
    with pytest.raises(ValueError, match="(?i)degree must be smaller"):
        TemporalBenchmarkConfig(neurons=8, degree=8)


def test_integer_rank_is_not_a_benchmark_configuration_knob() -> None:
    assert "rank" not in TemporalBenchmarkConfig.__dataclass_fields__


def test_curriculum_trace_half_lives_preserve_independent_horizon_pairs() -> None:
    selected = CurriculumTraceHalfLives(
        short=TraceHalfLives(5.0, 10.0),
        medium=TraceHalfLives(10.0, 30.0),
        long=TraceHalfLives(100.0, 30.0),
    )
    config = TemporalBenchmarkConfig(curriculum_trace_half_lives=selected)

    assert selected.for_horizon("medium") == TraceHalfLives(10.0, 30.0)
    assert config_to_dict(config)["curriculum_trace_half_lives"] == {
        "short": {"x": 5.0, "f": 10.0},
        "medium": {"x": 10.0, "f": 30.0},
        "long": {"x": 100.0, "f": 30.0},
    }


@given(
    invalid=st.one_of(
        st.floats(max_value=0.0, allow_nan=False),
        st.sampled_from((float("nan"), float("inf"), float("-inf"))),
    )
)
def test_trace_half_life_pairs_reject_invalid_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="trace half-life"):
        TraceHalfLives(invalid, 10.0)


def test_gradient_clip_norms_default_each_parameter_group_to_one() -> None:
    config = TemporalBenchmarkConfig()

    assert config.gradient_clip_norms == GradientClipNorms(
        readout=1.0,
        feedforward=1.0,
        recurrent=1.0,
    )
    assert "clip_norm" not in config_to_dict(config)
    assert config_to_dict(config)["gradient_clip_norms"] == {
        "readout": 1.0,
        "feedforward": 1.0,
        "recurrent": 1.0,
    }


@given(
    group=st.sampled_from(("readout", "feedforward", "recurrent")),
    invalid=st.one_of(
        st.floats(max_value=0.0, allow_nan=False),
        st.sampled_from((float("nan"), float("inf"), float("-inf"))),
    ),
)
def test_gradient_clip_norms_reject_invalid_group_values(
    group: str, invalid: float
) -> None:
    values = {"readout": 1.0, "feedforward": 1.0, "recurrent": 1.0}
    values[group] = invalid

    with pytest.raises(ValueError, match=f"{group} clip norm"):
        GradientClipNorms(**values)
