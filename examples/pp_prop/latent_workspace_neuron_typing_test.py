"""Tests for Example 21 neuron typing (Stage A: binary E/I split with Dale's law).

Covers the spec acceptance list in
``docs/specs/2026-08-21-example21-neuron-types.md`` §4:

1. ``neuron_typing="none"`` stays bit-exact with the pre-typing model.
2. Under ``ei_dale`` every recurrent edge sign follows its *presynaptic*
   neuron's type (the executed CSR row axis).
3. Type assignment is deterministic in the seed.
4. Post-step projection removes violations and leaves legal weights intact.
5. Configuration validation fails closed.
6. The model report carries type counts and a zero violation count.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        NEURON_TYPINGS,
        apply_dale_signs,
        assign_neuron_type_signs,
        build_sparse_topology,
        parameter_snapshot,
        project_dale_weights,
    )
except ImportError:
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        NEURON_TYPINGS,
        apply_dale_signs,
        assign_neuron_type_signs,
        build_sparse_topology,
        parameter_snapshot,
        project_dale_weights,
    )

EXAMPLE = pathlib.Path(__file__).with_name("21-latent-reasoning-in-context.py")

# Golden digests captured from the pre-typing implementation (commit cac015e)
# with the exact `_small_config()` below, inside the pinned Docker image
# (braintrace-gpu:0.11.0-py314, CPU backend).
GOLDEN_TOPOLOGY_VALUES_SHA256 = (
    "ee2a0389612239a38e959b35a47a129a187476f9ba9705ea15a4e064947b160d"
)
GOLDEN_PARAMETER_DIGEST = (
    "4ef4c3411e99197351a0aaeb2f6b7757c4361235a42fa4ec250b1f5aded1b7e2"
)
GOLDEN_SHORT_RUN_SHA256 = (
    "4867e0c0c7cdb92f009a4dd282f4371273afb76b78eb4f90cf32529319305dff"
)


def _small_config(**overrides: object) -> ModelConfig:
    arguments: dict[str, object] = {
        "input_width": 8,
        "neuron_count": 128,
        "recurrent_edges": 1024,
        "max_latent_steps": 32,
        "readout_width": 32,
        "color_rank": 4,
        "seed": 2108,
    }
    arguments.update(overrides)
    return ModelConfig(**arguments)


def _tree_digest(values: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        for leaf in jax.tree.leaves(values[key]):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _recurrent_weight_mantissa(model: LatentWorkspaceModel) -> np.ndarray:
    weight = model.rec_syn.comm.weight.value["weight"]
    return np.asarray(u.get_mantissa(weight))


class TestAssignNeuronTypeSigns:
    def test_deterministic_in_seed(self) -> None:
        first = assign_neuron_type_signs(128, 0.8, seed=2108)
        second = assign_neuron_type_signs(128, 0.8, seed=2108)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_differs(self) -> None:
        first = assign_neuron_type_signs(128, 0.8, seed=2108)
        second = assign_neuron_type_signs(128, 0.8, seed=2109)
        assert not np.array_equal(first, second)

    def test_counts_match_rounded_fraction(self) -> None:
        signs = assign_neuron_type_signs(128, 0.8, seed=2108)
        assert signs.dtype == np.int8
        assert int(np.sum(signs == 1)) == round(0.8 * 128)
        assert int(np.sum(signs == -1)) == 128 - round(0.8 * 128)

    def test_values_are_unit_signs(self) -> None:
        signs = assign_neuron_type_signs(64, 0.5, seed=0)
        assert set(np.unique(signs).tolist()) == {-1, 1}

    @pytest.mark.parametrize("fraction", [0.001, 0.999])
    def test_rejects_degenerate_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            assign_neuron_type_signs(128, fraction, seed=2108)


class TestApplyDaleSigns:
    def test_signs_follow_presynaptic_rows(self) -> None:
        topology = build_sparse_topology(128, 1024, seed=2108)
        signs = assign_neuron_type_signs(128, 0.8, seed=2108)
        typed = apply_dale_signs(topology, signs)
        edge_signs = signs[typed.rows]
        assert np.all(np.sign(typed.values) == edge_signs)

    def test_magnitudes_preserved(self) -> None:
        topology = build_sparse_topology(128, 1024, seed=2108)
        signs = assign_neuron_type_signs(128, 0.8, seed=2108)
        typed = apply_dale_signs(topology, signs)
        np.testing.assert_array_equal(np.abs(typed.values), np.abs(topology.values))

    def test_endpoints_unchanged(self) -> None:
        topology = build_sparse_topology(64, 256, seed=7)
        signs = assign_neuron_type_signs(64, 0.5, seed=7)
        typed = apply_dale_signs(topology, signs)
        np.testing.assert_array_equal(typed.rows, topology.rows)
        np.testing.assert_array_equal(typed.columns, topology.columns)
        assert typed.neuron_count == topology.neuron_count

    def test_rejects_wrong_shape(self) -> None:
        topology = build_sparse_topology(64, 256, seed=7)
        with pytest.raises(ValueError):
            apply_dale_signs(topology, np.ones(63, dtype=np.int8))

    def test_rejects_non_unit_signs(self) -> None:
        topology = build_sparse_topology(64, 256, seed=7)
        signs = np.ones(64, dtype=np.int8)
        signs[0] = 0
        with pytest.raises(ValueError):
            apply_dale_signs(topology, signs)


class TestProjectDaleWeights:
    def test_clamps_violations_to_zero(self) -> None:
        weights = jnp.asarray([1.0, -2.0, 3.0, -4.0]) * u.mA
        edge_signs = jnp.asarray([1, 1, -1, -1], dtype=jnp.int8)
        projected = project_dale_weights(weights, edge_signs)
        np.testing.assert_array_equal(
            np.asarray(u.get_mantissa(projected)), [1.0, 0.0, 0.0, -4.0]
        )

    def test_legal_weights_untouched(self) -> None:
        weights = jnp.asarray([0.5, -0.25, 0.0]) * u.mA
        edge_signs = jnp.asarray([1, -1, 1], dtype=jnp.int8)
        projected = project_dale_weights(weights, edge_signs)
        np.testing.assert_array_equal(
            np.asarray(u.get_mantissa(projected)), [0.5, -0.25, 0.0]
        )

    def test_plain_arrays_supported(self) -> None:
        weights = jnp.asarray([-1.0, 1.0])
        edge_signs = jnp.asarray([1, -1], dtype=jnp.int8)
        projected = project_dale_weights(weights, edge_signs)
        np.testing.assert_array_equal(np.asarray(projected), [0.0, 0.0])


class TestModelConfigValidation:
    def test_default_mode_is_none(self) -> None:
        assert _small_config().neuron_typing == "none"
        assert NEURON_TYPINGS == ("none", "ei_dale")

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            _small_config(neuron_typing="dale")

    def test_rejects_non_string_mode(self) -> None:
        with pytest.raises(TypeError):
            _small_config(neuron_typing=1)

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_rejects_fraction_outside_unit_interval(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            _small_config(neuron_typing="ei_dale", excitatory_fraction=fraction)

    def test_rejects_fraction_without_ei_dale(self) -> None:
        with pytest.raises(ValueError):
            _small_config(excitatory_fraction=0.5)

    def test_rejects_degenerate_split(self) -> None:
        with pytest.raises(ValueError):
            _small_config(neuron_typing="ei_dale", excitatory_fraction=0.999)

    def test_accepts_ei_dale_defaults(self) -> None:
        config = _small_config(neuron_typing="ei_dale")
        assert config.excitatory_fraction == 0.8


class TestUntypedDefaultPin:
    def test_topology_values_bit_exact(self) -> None:
        model = LatentWorkspaceModel(_small_config())
        digest = hashlib.sha256(model.topology.values.tobytes()).hexdigest()
        assert digest == GOLDEN_TOPOLOGY_VALUES_SHA256

    def test_parameter_digest_bit_exact(self) -> None:
        model = LatentWorkspaceModel(_small_config())
        assert _tree_digest(parameter_snapshot(model)) == GOLDEN_PARAMETER_DIGEST

    def test_short_run_bit_exact(self) -> None:
        model = LatentWorkspaceModel(_small_config())
        event = jnp.zeros((1, 8), dtype=jnp.float32).at[:, 0].set(1.0)
        advance = jnp.ones((1,), dtype=jnp.bool_)
        outputs = [np.asarray(model(event, advance)) for _ in range(5)]
        stacked = np.ascontiguousarray(np.stack(outputs))
        assert hashlib.sha256(stacked.tobytes()).hexdigest() == GOLDEN_SHORT_RUN_SHA256

    def test_untyped_model_has_no_type_state(self) -> None:
        model = LatentWorkspaceModel(_small_config())
        assert model.neuron_type_signs is None
        report = model.neuron_typing_report()
        assert report["mode"] == "none"


class TestEiDaleModel:
    def _model(self) -> LatentWorkspaceModel:
        return LatentWorkspaceModel(_small_config(neuron_typing="ei_dale"))

    def test_initial_weights_obey_dale(self) -> None:
        model = self._model()
        signs = np.asarray(model.neuron_type_signs)
        edge_signs = signs[model.topology.rows]
        mantissa = _recurrent_weight_mantissa(model)
        assert np.all(np.sign(mantissa) == edge_signs)

    def test_type_assignment_deterministic(self) -> None:
        first = self._model()
        second = self._model()
        np.testing.assert_array_equal(
            np.asarray(first.neuron_type_signs), np.asarray(second.neuron_type_signs)
        )

    def test_report_counts_and_zero_violations(self) -> None:
        model = self._model()
        report = model.neuron_typing_report()
        assert report["mode"] == "ei_dale"
        assert report["excitatory_count"] == round(0.8 * 128)
        assert report["inhibitory_count"] == 128 - round(0.8 * 128)
        assert report["excitatory_count"] + report["inhibitory_count"] == 128
        assert report["configured_excitatory_fraction"] == 0.8
        assert report["initial_sign_flip_count"] > 0
        assert report["recurrent_sign_violation_count"] == 0

    def test_projection_removes_violations_only(self) -> None:
        model = self._model()
        parameters = dict(model.rec_syn.comm.weight.value)
        edge_signs = np.asarray(model.neuron_type_signs)[model.topology.rows]
        flipped = parameters["weight"] * jnp.asarray(
            np.where(edge_signs == 1, -1.0, 1.0), dtype=jnp.float32
        )
        parameters["weight"] = flipped
        model.rec_syn.comm.weight.value = parameters
        before = model.neuron_typing_report()["recurrent_sign_violation_count"]
        assert before > 0
        model.project_recurrent_dale_weights()
        after = model.neuron_typing_report()
        assert after["recurrent_sign_violation_count"] == 0
        mantissa = _recurrent_weight_mantissa(model)
        legal = np.asarray(u.get_mantissa(flipped)) * edge_signs >= 0
        np.testing.assert_array_equal(
            mantissa[legal], np.asarray(u.get_mantissa(flipped))[legal]
        )
        assert np.all(mantissa[~legal] == 0.0)

    def test_forward_pass_finite(self) -> None:
        model = self._model()
        event = jnp.zeros((1, 8), dtype=jnp.float32).at[:, 0].set(1.0)
        advance = jnp.ones((1,), dtype=jnp.bool_)
        for _ in range(3):
            output = model(event, advance)
        assert bool(jnp.all(jnp.isfinite(output)))


def _load_example():
    name = "example21_neuron_typing_under_test"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


class TestExperimentSurface:
    def test_parser_accepts_typing_flags(self, example) -> None:
        args = example._parser().parse_args(
            ["--neuron-typing", "ei_dale", "--excitatory-fraction", "0.75"]
        )
        config = example._config_from_args(args)
        assert config.neuron_typing == "ei_dale"
        assert config.excitatory_fraction == 0.75

    def test_parser_defaults_to_none(self, example) -> None:
        config = example._config_from_args(example._parser().parse_args([]))
        assert config.neuron_typing == "none"
        assert config.excitatory_fraction == 0.8

    def test_smoke_config_accepts_typing(self, example) -> None:
        config = example.ExperimentConfig.smoke_config(neuron_typing="ei_dale")
        assert config.neuron_typing == "ei_dale"
        assert config.smoke

    def test_rejects_ei_dale_with_task_local_adaptation(self, example) -> None:
        with pytest.raises(ValueError):
            example.ExperimentConfig(
                neuron_typing="ei_dale", task_local_adaptation=True
            )

    def test_model_config_carries_typing(self, example) -> None:
        config = example.ExperimentConfig.smoke_config(neuron_typing="ei_dale")
        rows = example._row_config(config)
        model_config = example._model_config(config, rows, batch_size=1)
        assert model_config.neuron_typing == "ei_dale"
        assert model_config.excitatory_fraction == 0.8

    def test_training_applies_projection_after_update(self, example) -> None:
        source = EXAMPLE.read_text(encoding="utf-8")
        update_index = source.index("optimizer.update(")
        projection_index = source.index("project_recurrent_dale_weights", update_index)
        assert projection_index > update_index
