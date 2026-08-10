"""Property tests for scientific fixed-degree topology construction."""

import math

import numpy as np
from hypothesis import given, strategies as st

from temporal_benchmark_topology import fixed_degree_topology, topology_metrics


@given(
    neurons=st.integers(min_value=2, max_value=50),
    data=st.data(),
    topology_seed=st.integers(min_value=0, max_value=2**32 - 1),
    weight_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_rows_are_unique_fixed_degree_and_self_loop_free(
    neurons: int, data, topology_seed: int, weight_seed: int
) -> None:
    degree = data.draw(st.integers(min_value=1, max_value=neurons - 1))
    topology = fixed_degree_topology(neurons, degree, topology_seed, weight_seed, 0.8)
    rows = topology.indices.reshape(neurons, degree)

    assert topology.indices.size == neurons * degree
    for row, neighbors in enumerate(rows):
        assert np.unique(neighbors).size == degree
        assert row not in neighbors


def test_weight_seed_is_separate_from_topology_seed() -> None:
    first = fixed_degree_topology(24, 4, 7, 11, 0.8)
    changed_weights = fixed_degree_topology(24, 4, 7, 12, 0.8)

    assert np.array_equal(first.indices, changed_weights.indices)
    assert not np.array_equal(first.values, changed_weights.values)
    assert math.isclose(float(first.values.std()), 0.8 / math.sqrt(4), rel_tol=0.35)


def test_topology_metrics_name_power_gain_as_an_estimate() -> None:
    metrics = topology_metrics(fixed_degree_topology(24, 4, 1, 2, 0.8))

    assert metrics["duplicate_count"] == 0
    assert metrics["self_loop_count"] == 0
    assert metrics["out_degree"]["minimum"] == 4
    assert metrics["out_degree"]["maximum"] == 4
    assert metrics["sparse_power_gain_is_exact_spectral_radius"] is False
