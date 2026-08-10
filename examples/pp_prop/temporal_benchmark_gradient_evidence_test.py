"""Tests for finite-window recurrent-gradient evidence."""

import numpy as np
import pytest

from temporal_benchmark_config import SplitSizes, TemporalBenchmarkConfig
from temporal_benchmark_gradient_evidence import (
    _reference_config,
    run_gradient_evidence,
)
from temporal_benchmark_manifest import find_bundle, generate_manifest


def _config(updates: int) -> TemporalBenchmarkConfig:
    return TemporalBenchmarkConfig(
        horizon="short",
        batch_size=4,
        updates=updates,
        split_sizes=SplitSizes(train=8, validation=4, test=4),
        device="cpu",
        allow_dirty=True,
    )


def test_reference_requires_quarter_training_checkpoints() -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        _reference_config(_config(5))


def test_reference_reports_all_finite_window_checkpoints() -> None:
    bundle = find_bundle(generate_manifest(), "split0-topology0-weight0")

    evidence = run_gradient_evidence(_config(4), bundle)

    assert evidence["reference_neurons"] == 24
    assert evidence["reference_degree"] == 4
    assert [probe["update"] for probe in evidence["probes"]] == [0, 1, 3]
    for probe in evidence["probes"]:
        assert all(
            np.isfinite(value) for key, value in probe.items() if key != "update"
        )
