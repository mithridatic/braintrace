"""Integration tests for paired BrainTrace and BPTT benchmark construction."""

from dataclasses import replace

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temporal_benchmark_config import (
    GradientClipNorms,
    SplitSizes,
    TemporalBenchmarkConfig,
)
from temporal_benchmark_manifest import find_bundle, generate_manifest
from temporal_benchmark_training import _build_model, run_training


def _config(arm: str) -> TemporalBenchmarkConfig:
    return TemporalBenchmarkConfig(
        arm=arm,
        horizon="short",
        neurons=8,
        degree=2,
        batch_size=4,
        updates=1,
        split_sizes=SplitSizes(train=8, validation=4, test=4),
        device="cpu",
        allow_dirty=True,
    )


def _bundle():
    return find_bundle(generate_manifest(), "split0-topology0-weight0")


def _parameter_values(model) -> dict:
    return model.states(brainstate.ParamState).to_dict_values()


def test_pp_prop_and_bptt_start_from_identical_parameter_state() -> None:
    with brainstate.environ.context(dt=1.0 * u.ms):
        pp_prop, _ = _build_model(_config("all_pp_prop"), _bundle())
        bptt, _ = _build_model(_config("all_bptt"), _bundle())
    pp_leaves = jax.tree.leaves(_parameter_values(pp_prop))
    bptt_leaves = jax.tree.leaves(_parameter_values(bptt))

    assert len(pp_leaves) == len(bptt_leaves)
    for left, right in zip(pp_leaves, bptt_leaves):
        np.testing.assert_array_equal(u.get_mantissa(left), u.get_mantissa(right))


def test_recurrence_zero_and_absent_module_are_forward_equivalent() -> None:
    inputs = jnp.zeros((10, 4, 17), dtype=jnp.float32)

    def evolve(arm: str):
        model, _ = _build_model(_config(arm), _bundle())
        brainstate.nn.init_all_states(model, batch_size=4)
        return brainstate.transform.for_loop(model, inputs)

    with brainstate.environ.context(dt=1.0 * u.ms):
        recurrence_zero = evolve("feedforward_readout_recurrence_zero")
        no_recurrence = evolve("no_recurrent_module")

    np.testing.assert_array_equal(recurrence_zero, no_recurrence)


def test_recurrence_zero_and_absent_module_train_equivalently() -> None:
    recurrence_zero = run_training(
        _config("feedforward_readout_recurrence_zero"), _bundle()
    )
    no_recurrence = run_training(_config("no_recurrent_module"), _bundle())

    np.testing.assert_array_equal(recurrence_zero["losses"], no_recurrence["losses"])
    assert recurrence_zero["final_validation"] == no_recurrence["final_validation"]


def test_long_no_recurrence_path_cannot_carry_cue_into_response() -> None:
    inputs = jnp.zeros((100, 2, 17), dtype=jnp.float32)
    inputs = inputs.at[:4, 0, :8].set(1.0)
    inputs = inputs.at[:4, 1, 8:16].set(1.0)
    inputs = inputs.at[-4:, :, 16].set(1.0)
    config = _config("no_recurrent_module")

    with brainstate.environ.context(dt=1.0 * u.ms):
        model, _ = _build_model(config, _bundle())
        brainstate.nn.init_all_states(model, batch_size=2)
        outputs = brainstate.transform.for_loop(model, inputs)

    np.testing.assert_allclose(outputs[-4:, 0], outputs[-4:, 1], atol=1e-7)


@pytest.mark.parametrize(
    ("arm", "algorithm"),
    [
        ("readout_only", "single_step_pp_prop"),
        ("feedforward_readout_recurrence_zero", "single_step_pp_prop"),
        ("recurrent_readout", "single_step_pp_prop"),
        ("all_pp_prop", "single_step_pp_prop"),
        ("no_recurrent_module", "single_step_pp_prop"),
        ("frozen_random_recurrence", "single_step_pp_prop"),
        ("all_bptt", "full_window_reverse_mode_bptt_oracle"),
    ],
)
def test_tiny_training_smoke_uses_declared_algorithm(arm: str, algorithm: str) -> None:
    result = run_training(_config(arm), _bundle())

    assert result["status"] == "completed"
    assert result["algorithm"] == algorithm
    assert len(result["losses"]) == 1
    assert np.isfinite(result["losses"][0])
    assert result["response_label_independent"] is True
    assert result["fixed_dynamics_time_constants_ms"]["recurrent_synapse"] == 3.0


def test_training_applies_each_parameter_group_clip_norm() -> None:
    config = _config("all_bptt")
    config = replace(
        config,
        gradient_clip_norms=GradientClipNorms(
            readout=None,
            feedforward=1e-6,
            recurrent=1e-5,
        ),
    )

    telemetry = run_training(config, _bundle())["optimizer_telemetry"]

    readout = telemetry["readout"]
    feedforward = telemetry["feedforward"]
    recurrent = telemetry["recurrent"]
    np.testing.assert_allclose(
        readout["clipped_gradient_norm"], readout["raw_gradient_norm"]
    )
    assert max(feedforward["clipped_gradient_norm"]) <= 1.01e-6
    assert max(recurrent["clipped_gradient_norm"]) <= 1.01e-5
