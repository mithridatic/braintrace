"""O(nnz) scientific recurrent topology construction and diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedDegreeTopology:
    """Store a square fixed-out-degree weighted CSR topology."""

    neurons: int
    degree: int
    indices: np.ndarray
    indptr: np.ndarray
    values: np.ndarray


def _sample_without_replacement(
    random: np.random.Generator, population: int, count: int
) -> np.ndarray:
    """Use Floyd sampling in O(count) space and expected time."""
    chosen: set[int] = set()
    for upper in range(population - count, population):
        candidate = int(random.integers(0, upper + 1))
        chosen.add(upper if candidate in chosen else candidate)
    return np.fromiter(chosen, dtype=np.int32, count=count)


def fixed_degree_topology(
    neurons: int,
    degree: int,
    topology_seed: int,
    weight_seed: int,
    gain: float,
) -> FixedDegreeTopology:
    """Build unique per-row neighbors without self-loops or a dense mask."""
    if neurons <= 1 or degree <= 0 or degree >= neurons:
        raise ValueError("Require neurons > 1 and 0 < degree < neurons. Fix the input condition named in the error, then rerun the operation.")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Gain must be finite and positive. Set Gain to a finite positive value.")
    topology_random = np.random.default_rng(topology_seed)
    rows = np.empty((neurons, degree), dtype=np.int32)
    for row in range(neurons):
        sampled = _sample_without_replacement(topology_random, neurons - 1, degree)
        rows[row] = np.sort(sampled + (sampled >= row))
    weight_random = np.random.default_rng(weight_seed)
    values = weight_random.normal(
        0.0, gain / math.sqrt(degree), neurons * degree
    ).astype(np.float32)
    indptr = np.arange(neurons + 1, dtype=np.int64) * degree
    return FixedDegreeTopology(neurons, degree, rows.reshape(-1), indptr, values)


def _degree_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def sparse_power_gain_estimate(
    topology: FixedDegreeTopology, *, iterations: int = 30, seed: int = 0
) -> float:
    """Estimate recurrent operator gain by normalized sparse power iteration."""
    if iterations <= 0:
        raise ValueError("Iterations must be positive. Set Iterations to a positive value.")
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(
        (topology.values, topology.indices, topology.indptr),
        shape=(topology.neurons, topology.neurons),
    )
    vector = np.random.default_rng(seed).normal(size=topology.neurons)
    vector /= np.linalg.norm(vector)
    gain = 0.0
    for _ in range(iterations):
        projected = matrix @ vector
        gain = float(np.linalg.norm(projected))
        if gain == 0.0:
            return 0.0
        vector = projected / gain
    return gain


def topology_metrics(topology: FixedDegreeTopology) -> dict[str, object]:
    """Report structural diagnostics and a named non-exact gain estimate."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.repeat(np.arange(topology.neurons), np.diff(topology.indptr))
    out_degree = np.diff(topology.indptr).astype(np.int64)
    in_degree = np.bincount(topology.indices, minlength=topology.neurons)
    duplicate_count = sum(
        topology.degree
        - np.unique(topology.indices[start : start + topology.degree]).size
        for start in range(0, topology.indices.size, topology.degree)
    )
    graph = csr_matrix(
        (np.ones(topology.indices.size), topology.indices, topology.indptr),
        shape=(topology.neurons, topology.neurons),
    )
    weak_count, weak_labels = connected_components(
        graph, directed=True, connection="weak"
    )
    strong_count, strong_labels = connected_components(
        graph, directed=True, connection="strong"
    )
    weak_sizes = np.bincount(weak_labels, minlength=weak_count)
    strong_sizes = np.bincount(strong_labels, minlength=strong_count)
    return {
        "topology_family": "independent_fixed_degree_no_self_loops",
        "row_count": topology.neurons,
        "edge_count": int(topology.indices.size),
        "row_degree": _degree_stats(out_degree),
        "out_degree": _degree_stats(out_degree),
        "in_degree": _degree_stats(in_degree),
        "duplicate_count": int(duplicate_count),
        "self_loop_count": int(np.count_nonzero(rows == topology.indices)),
        "weak_component_count": int(weak_count),
        "strong_component_count": int(strong_count),
        "largest_weak_component_fraction": float(weak_sizes.max() / topology.neurons),
        "largest_strong_component_fraction": float(
            strong_sizes.max() / topology.neurons
        ),
        "sparse_power_gain_estimate": sparse_power_gain_estimate(topology),
        "sparse_power_gain_is_exact_spectral_radius": False,
    }
