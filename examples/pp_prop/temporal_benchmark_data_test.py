"""Property tests for delayed-cue trial generation and encoding."""

import math

import numpy as np
from hypothesis import given, strategies as st

from temporal_benchmark_config import HORIZONS
from temporal_benchmark_data import (
    GO_CHANNEL,
    balanced_trial_specs,
    encode_trials,
    rate_template,
    response_is_label_independent,
    response_mask,
    spike_probability,
)


@given(
    st.floats(min_value=0.0, max_value=5000.0, allow_nan=False),
    st.floats(min_value=1e-6, max_value=0.1, allow_nan=False),
)
def test_physical_rate_conversion(rate: float, dt_seconds: float) -> None:
    probability = spike_probability(rate, dt_seconds)

    assert 0.0 <= probability <= 1.0
    assert math.isclose(probability, -math.expm1(-rate * dt_seconds))


def test_balanced_trials_are_reproducible() -> None:
    first = balanced_trial_specs(1024, 42)
    repeated = balanced_trial_specs(1024, 42)

    assert first == repeated
    assert sum(spec.label == 0 for spec in first) == 512
    assert sum(spec.label == 1 for spec in first) == 512


def test_response_rates_and_supervision_are_label_independent() -> None:
    for horizon in HORIZONS.values():
        assert response_is_label_independent(horizon, 200.0, 200.0)
        mask = response_mask(horizon)
        assert mask.sum() == 4
        assert np.all(mask[:-4] == 0)
        for label in (0, 1):
            rates = rate_template(label, horizon, 200.0, 200.0)
            assert np.all(rates[-4:, GO_CHANNEL] == 200.0)
            assert np.count_nonzero(rates) == 4 * 8 + 4


def test_encoding_is_shared_and_horizon_shaped() -> None:
    specs = balanced_trial_specs(4, 7)
    short = encode_trials(
        specs,
        HORIZONS["short"],
        99,
        cue_rate_hz=200.0,
        go_rate_hz=200.0,
        dt_seconds=0.001,
    )
    repeated = encode_trials(
        specs,
        HORIZONS["short"],
        99,
        cue_rate_hz=200.0,
        go_rate_hz=200.0,
        dt_seconds=0.001,
    )

    assert short.shape == (10, 4, 17)
    assert np.array_equal(short, repeated)
