"""Tests for Example 21's latent-workspace model."""

from __future__ import annotations

import ast
import inspect
import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import braintrace

try:
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        _surrogate_spike,
        factored_memory_read,
        occupied_slot_derangement,
        parameter_snapshot,
        phase_masks,
        run_sequence,
        shuffled_memory_factors,
    )
    from examples.pp_prop.latent_workspace_task import (
        TaskConfig,
        build_codebook,
        generate_episode,
    )
except ModuleNotFoundError:
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        _surrogate_spike,
        factored_memory_read,
        occupied_slot_derangement,
        parameter_snapshot,
        phase_masks,
        run_sequence,
        shuffled_memory_factors,
    )
    from latent_workspace_task import TaskConfig, build_codebook, generate_episode


def _task(*, bindings: int = 2, slots: int = 3, latent: int = 1) -> TaskConfig:
    return TaskConfig(
        symbol_count=10,
        binding_count=bindings,
        slot_capacity=slots,
        latent_steps=latent,
        code_width=12,
        spike_rate=0.25,
        symbol_ticks=2,
    )


def _model(
    *, bindings: int = 2, slots: int = 3, latent: int = 1, width: int = 8
) -> LatentWorkspaceModel:
    return LatentWorkspaceModel(
        ModelConfig(
            task=_task(bindings=bindings, slots=slots, latent=latent),
            latent_width=width,
        )
    )


def _run_prefix(model: LatentWorkspaceModel, model_inputs: jax.Array) -> jax.Array:
    sequence = jnp.asarray(model_inputs, dtype=jnp.float32)
    if sequence.ndim == 2:
        sequence = sequence[:, None, :]
    return brainstate.transform.for_loop(model, sequence)


@settings(max_examples=40, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=3),
    slots=st.integers(min_value=1, max_value=4),
    width=st.integers(min_value=1, max_value=6),
)
def test_factored_read_equals_dense_outer_product(
    batch: int, slots: int, width: int
) -> None:
    size = batch * slots * width
    values = jnp.arange(size, dtype=jnp.float32).reshape(batch, slots, width)
    values = values / jnp.asarray(max(size, 1), dtype=jnp.float32)
    keys = jnp.flip(values + 0.125, axis=-1)
    query = jnp.arange(batch * width, dtype=jnp.float32).reshape(batch, width)
    query = query / jnp.asarray(max(batch * width, 1), dtype=jnp.float32)

    factored = factored_memory_read(values, keys, query)
    dense = jnp.einsum("bmi,bmj->bij", values, keys)
    expected = jnp.einsum("bij,bj->bi", dense, query)

    np.testing.assert_allclose(factored, expected, rtol=2e-5, atol=2e-5)


def test_factored_read_lowers_to_exactly_two_contractions() -> None:
    values = jnp.ones((1, 2, 3), dtype=jnp.float32)
    keys = jnp.ones_like(values)
    query = jnp.ones((1, 3), dtype=jnp.float32)

    jaxpr = jax.make_jaxpr(factored_memory_read)(values, keys, query).jaxpr

    assert [eqn.primitive.name for eqn in jaxpr.eqns].count("dot_general") == 2


def test_phase_masks_activate_exactly_one_submap_per_tick() -> None:
    episode = generate_episode(
        _task(bindings=3, slots=3, latent=2),
        brainstate.random.RandomState(21),
    )

    demo, query, latent, seed = phase_masks(
        jnp.asarray(episode.model_inputs), episode.config
    )
    combined = jnp.concatenate((demo, query, latent), axis=-1)

    np.testing.assert_array_equal(np.asarray(combined.sum(axis=-1)), 1.0)
    np.testing.assert_array_equal(
        np.argmax(np.asarray(combined), axis=-1),
        np.concatenate((np.zeros(6), np.ones(2), np.full(2, 2))).astype(np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(seed)[:, 0],
        np.concatenate((np.zeros(8), np.ones(1), np.zeros(1))),
    )


def test_demo_step_changes_memory_without_changing_parameters() -> None:
    model = _model()
    episode = generate_episode(model.config.task, brainstate.random.RandomState(22))
    before = parameter_snapshot(model)
    memory_before = np.asarray(model.memory_factors()[0]).copy()

    model(jnp.asarray(episode.model_inputs[0:1]))

    after = parameter_snapshot(model)
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[name], after[name]) for name in before)
    assert not np.array_equal(memory_before, np.asarray(model.memory_factors()[0]))


def test_different_demonstrations_produce_different_memory() -> None:
    task = _task(bindings=3, slots=3)
    first = generate_episode(task, brainstate.random.RandomState(23))
    second = generate_episode(task, brainstate.random.RandomState(24))
    model = _model(bindings=3, slots=3)

    _run_prefix(model, jnp.asarray(first.model_inputs[: task.demonstration_steps]))
    first_values, first_keys = model.memory_factors()
    first_memory = np.concatenate(
        (np.asarray(first_values).ravel(), np.asarray(first_keys).ravel())
    )
    model.reset_state()
    _run_prefix(model, jnp.asarray(second.model_inputs[: task.demonstration_steps]))
    second_values, second_keys = model.memory_factors()
    second_memory = np.concatenate(
        (np.asarray(second_values).ravel(), np.asarray(second_keys).ravel())
    )

    assert not np.array_equal(first_memory, second_memory)


def test_memory_storage_is_linear_in_slots_and_width() -> None:
    small = _model(slots=3, width=8)
    more_slots = _model(slots=6, width=8)
    wider = _model(slots=3, width=16)

    assert small.memory_storage_elements == 2 * 3 * 8
    assert more_slots.memory_storage_elements == 2 * small.memory_storage_elements
    assert wider.memory_storage_elements == 2 * small.memory_storage_elements
    assert small.memory_storage_elements != small.config.latent_width**2


def test_shuffle_preserves_shape_magnitude_and_keys_but_changes_pairing() -> None:
    task = _task(bindings=3, slots=3)
    episode = generate_episode(task, brainstate.random.RandomState(25))
    model = _model(bindings=3, slots=3)
    _run_prefix(model, jnp.asarray(episode.model_inputs[: task.demonstration_steps]))
    values, keys = model.memory_factors()
    permutation = occupied_slot_derangement(task.slot_capacity, task.binding_count)

    shuffled_values, shuffled_keys = shuffled_memory_factors(values, keys, permutation)

    assert shuffled_values.shape == values.shape
    assert shuffled_keys.shape == keys.shape
    np.testing.assert_array_equal(shuffled_keys, keys)
    np.testing.assert_allclose(
        jnp.linalg.norm(shuffled_values), jnp.linalg.norm(values)
    )
    assert not np.array_equal(shuffled_values, values)
    query = jnp.arange(model.config.latent_width, dtype=jnp.float32)[None, :]
    assert not np.allclose(
        np.asarray(factored_memory_read(values, keys, query)),
        np.asarray(factored_memory_read(shuffled_values, shuffled_keys, query)),
    )

    before_shape = model.grouped_state.value.shape
    model.shuffle_memory(permutation)
    assert model.grouped_state.value.shape == before_shape
    np.testing.assert_allclose(model.memory_factors()[0], shuffled_values)


def test_occupied_slot_derangement_moves_only_occupied_slots() -> None:
    permutation = occupied_slot_derangement(slot_count=5, occupied_count=3)

    np.testing.assert_array_equal(permutation, jnp.asarray([1, 2, 0, 3, 4]))


@pytest.mark.parametrize(
    "permutation",
    (
        jnp.asarray([0, 0, 2], dtype=jnp.int32),
        jnp.asarray([0, 1, 3], dtype=jnp.int32),
        jnp.asarray([0.0, 2.0, 1.0], dtype=jnp.float32),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
    ),
)
def test_shuffle_rejects_non_mismatching_or_malformed_permutation(
    permutation: jax.Array,
) -> None:
    values = jnp.ones((1, 3, 4), dtype=jnp.float32)
    keys = jnp.ones_like(values)

    with pytest.raises(ValueError, match="permutation"):
        shuffled_memory_factors(values, keys, permutation)


def test_zero_latent_steps_runs_and_reads_query_terminal() -> None:
    task = _task(bindings=2, slots=3, latent=0)
    episode = generate_episode(task, brainstate.random.RandomState(26))
    model = _model(bindings=2, slots=3, latent=0)

    result = run_sequence(model, jnp.asarray(episode.model_inputs))

    assert result.logits.shape == (task.total_steps, 1, task.symbol_count)
    assert result.workspace.shape == (task.total_steps, 1, model.config.latent_width)
    assert result.memory_read.shape == (1, model.config.latent_width)
    assert result.memory_values.shape == (
        1,
        task.slot_capacity,
        model.config.latent_width,
    )
    assert result.memory_keys.shape == (
        1,
        task.slot_capacity,
        model.config.latent_width,
    )
    values, keys = result.memory_factors
    expected_query_encoding = (
        jnp.sum(
            jnp.asarray(episode.model_inputs[task.query_slice, task.key_slice])
            @ model.Wk.value,
            axis=0,
            keepdims=True,
        )
        / task.symbol_ticks
    )
    np.testing.assert_allclose(result.query_encoding, expected_query_encoding)
    np.testing.assert_array_equal(values, result.memory_values)
    np.testing.assert_array_equal(keys, result.memory_keys)
    expected_query_read = factored_memory_read(values, keys, result.query_encoding)
    np.testing.assert_allclose(result.memory_read, expected_query_read)
    assert not np.array_equal(result.query_encoding, result.workspace[-1])
    np.testing.assert_array_equal(result.terminal_logits, result.logits[-1])
    assert np.all(np.isfinite(np.asarray(result.logits)))


def test_memory_read_includes_final_query_tick_and_controls_r0_logits() -> None:
    task = _task(bindings=2, slots=3, latent=0)
    episode = generate_episode(task, brainstate.random.RandomState(2600))
    model = LatentWorkspaceModel(ModelConfig(task=task, latent_width=16))

    result = run_sequence(model, jnp.asarray(episode.model_inputs))
    h0_index = task.query_slice.stop - 1
    values, keys = result.memory_factors
    incomplete_query = (
        jnp.sum(
            jnp.asarray(
                episode.model_inputs[task.query_slice.start : h0_index, task.key_slice]
            )
            @ model.Wk.value,
            axis=0,
            keepdims=True,
        )
        / task.symbol_ticks
    )
    incomplete_read = factored_memory_read(values, keys, incomplete_query)

    h0 = np.asarray(result.workspace[h0_index])
    assert np.all((h0 == 0.0) | (h0 == 1.0))
    assert np.any(
        (np.asarray(result.memory_read) != 0.0)
        & (np.asarray(result.memory_read) != 1.0)
    )
    np.testing.assert_allclose(
        result.terminal_logits,
        result.memory_read @ model.Wo.value,
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(np.asarray(result.memory_read), np.asarray(incomplete_read))


def test_memory_factors_and_query_equal_direct_aggregate_projections() -> None:
    task = TaskConfig(
        symbol_count=10,
        binding_count=4,
        slot_capacity=6,
        latent_steps=0,
        code_width=24,
        spike_rate=0.25,
        symbol_ticks=4,
    )
    episode = generate_episode(task, brainstate.random.RandomState(2601))
    model = LatentWorkspaceModel(ModelConfig(task=task, latent_width=32))

    result = run_sequence(model, jnp.asarray(episode.model_inputs))
    codebook = jnp.asarray(episode.codebook)
    aggregate_codes = jnp.mean(codebook, axis=1)
    direct_keys = aggregate_codes @ model.Wk.value
    direct_values = aggregate_codes @ model.Wv.value
    occupied = task.binding_count

    np.testing.assert_allclose(
        result.memory_keys[0, :occupied],
        direct_keys[jnp.asarray(episode.demonstration_keys)],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result.memory_values[0, :occupied],
        direct_values[jnp.asarray(episode.demonstration_values)],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result.query_encoding,
        direct_keys[jnp.asarray([episode.query_symbol])],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(result.memory_keys[0, occupied:], 0.0)
    np.testing.assert_array_equal(result.memory_values[0, occupied:], 0.0)


def test_reset_clears_all_hidden_state() -> None:
    model = _model()
    episode = generate_episode(model.config.task, brainstate.random.RandomState(27))
    run_sequence(model, jnp.asarray(episode.model_inputs))

    model.reset_state()

    np.testing.assert_array_equal(model.grouped_state.value, 0.0)
    np.testing.assert_array_equal(model.ingestion_state.value, 0.0)
    np.testing.assert_array_equal(model.latent_voltage.value, 0.0)
    np.testing.assert_array_equal(model.query_encoding.value, 0.0)


def test_fixed_random_write_mode_is_explicit_but_uses_param_states() -> None:
    model = _model()

    assert model.config.write_mode == "fixed_random"
    assert model.write_projections_trainable is False
    assert isinstance(model.Wk, brainstate.ParamState)
    assert isinstance(model.Wv, brainstate.ParamState)
    assert {"Wk", "Wv"}.issubset(parameter_snapshot(model))


def test_projection_seed_is_separate_from_recurrent_experiment_seed() -> None:
    task = _task()
    baseline = LatentWorkspaceModel(
        ModelConfig(task=task, latent_width=8, seed=2108, projection_seed=210848)
    )
    new_recurrence = LatentWorkspaceModel(
        ModelConfig(task=task, latent_width=8, seed=2109, projection_seed=210848)
    )
    new_projection = LatentWorkspaceModel(
        ModelConfig(task=task, latent_width=8, seed=2108, projection_seed=210849)
    )

    for name in ("Wk", "Wv", "Wo"):
        np.testing.assert_array_equal(
            getattr(baseline, name).value,
            getattr(new_recurrence, name).value,
        )
        assert not np.array_equal(
            np.asarray(getattr(baseline, name).value),
            np.asarray(getattr(new_projection, name).value),
        )
    np.testing.assert_array_equal(baseline.Wf.value, new_projection.Wf.value)
    assert not np.array_equal(
        np.asarray(baseline.Wf.value), np.asarray(new_recurrence.Wf.value)
    )


def _pure_memory_supported_accuracy(binding_count: int, episodes: int = 4096) -> float:
    task = TaskConfig(binding_count=binding_count, latent_steps=0)
    model = LatentWorkspaceModel(ModelConfig(task=task, latent_width=32))
    random = brainstate.random.RandomState(210_848 + binding_count)
    rule = jnp.argsort(
        random.uniform(size=(episodes, task.symbol_count)), axis=1
    ).astype(jnp.int32)
    query = random.randint(task.symbol_count, size=(episodes,)).astype(jnp.int32)
    candidate_scores = random.uniform(size=(episodes, task.symbol_count))
    candidate_scores = candidate_scores.at[jnp.arange(episodes), query].set(2.0)
    other_keys = jnp.argsort(candidate_scores, axis=1)[:, : binding_count - 1]
    keys = jnp.concatenate((query[:, None], other_keys), axis=1)
    values = jnp.take_along_axis(rule, keys, axis=1)
    target = jnp.take_along_axis(rule, query[:, None], axis=1)[:, 0]

    aggregate_codes = jnp.mean(jnp.asarray(build_codebook(task)), axis=1)
    key_embeddings = aggregate_codes @ model.Wk.value
    value_embeddings = aggregate_codes @ model.Wv.value
    read = factored_memory_read(
        value_embeddings[values], key_embeddings[keys], key_embeddings[query]
    )
    prediction = jnp.argmax(read @ value_embeddings.T, axis=1)
    return float(jnp.mean(prediction == target))


def test_fixed_projection_pure_memory_has_expected_interference_curve() -> None:
    k2_accuracy = _pure_memory_supported_accuracy(2)
    k8_accuracy = _pure_memory_supported_accuracy(8)

    assert k2_accuracy >= 0.90
    assert k8_accuracy <= 0.60
    assert k2_accuracy > k8_accuracy


def test_fixed_random_trainable_mapping_excludes_write_parameters() -> None:
    model = _model()
    before = parameter_snapshot(model)

    trainable = model.trainable_parameters()
    for state in trainable.values():
        state.value = state.value + 0.01

    after = parameter_snapshot(model)
    assert {"Wf", "Wo"} == {".".join(map(str, path)) for path in trainable}
    np.testing.assert_array_equal(after["Wk"], before["Wk"])
    np.testing.assert_array_equal(after["Wv"], before["Wv"])
    assert not np.array_equal(after["Wf"], before["Wf"])
    assert not np.array_equal(after["Wo"], before["Wo"])


def test_learned_write_mode_includes_write_parameters() -> None:
    config = ModelConfig(task=_task(), latent_width=8, write_mode="learned")
    model = LatentWorkspaceModel(config)

    assert {"Wk", "Wv", "Wf", "Wo"} == {
        ".".join(map(str, path)) for path in model.trainable_parameters()
    }


def test_complete_frozen_episode_leaves_all_parameters_bitwise_identical() -> None:
    model = _model()
    episode = generate_episode(model.config.task, brainstate.random.RandomState(270))
    before = parameter_snapshot(model)

    run_sequence(model, jnp.asarray(episode.model_inputs))

    after = parameter_snapshot(model)
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[name], after[name]) for name in before)


def test_native_batch_execution_keeps_episode_state_separate() -> None:
    task = _task(bindings=2, slots=3, latent=2)
    first = generate_episode(task, brainstate.random.RandomState(271))
    second = generate_episode(task, brainstate.random.RandomState(272))
    inputs = jnp.stack(
        (jnp.asarray(first.model_inputs), jnp.asarray(second.model_inputs)), axis=1
    )
    model = LatentWorkspaceModel(
        ModelConfig(task=task, batch_size=2, latent_width=8, seed=273)
    )

    result = run_sequence(model, inputs)

    assert result.workspace.shape == (task.total_steps, 2, 8)
    assert result.memory_read.shape == (2, 8)
    assert not np.array_equal(result.memory_values[0], result.memory_values[1])


def test_reported_h0_through_hr_and_internal_workspace_are_binary() -> None:
    model = _model(latent=2)
    episode = generate_episode(model.config.task, brainstate.random.RandomState(274))

    result = run_sequence(model, jnp.asarray(episode.model_inputs))

    trajectory = np.asarray(result.workspace[model.config.task.query_slice.stop - 1 :])
    internal = np.asarray(model.workspace)
    assert np.all((trajectory == 0.0) | (trajectory == 1.0))
    assert np.all((internal == 0.0) | (internal == 1.0))
    np.testing.assert_allclose(
        result.terminal_logits,
        model.latent_voltage_view @ model.Wo.value,
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.all(np.isfinite(np.asarray(model.latent_voltage.value)))


def test_terminal_logits_decode_one_analog_carrier_at_every_depth() -> None:
    for latent in (0, 1, 2, 4):
        model = _model(latent=latent)
        episode = generate_episode(
            model.config.task, brainstate.random.RandomState(2740 + latent)
        )

        result = run_sequence(model, jnp.asarray(episode.model_inputs))

        carrier = result.memory_read if latent == 0 else model.latent_voltage_view
        np.testing.assert_allclose(
            result.terminal_logits,
            carrier @ model.Wo.value,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"readout diverged from its analog carrier at R={latent}",
        )


def test_h0_is_seeded_with_the_analog_contextual_read() -> None:
    model = _model(latent=3)
    task = model.config.task
    episode = generate_episode(task, brainstate.random.RandomState(2745))
    sequence = jnp.asarray(episode.model_inputs)[:, None, :]

    brainstate.transform.for_loop(model, sequence[: task.query_slice.stop])
    values, keys = model.memory_factors()
    expected_h0 = factored_memory_read(values, keys, model.query_encoding_view)
    model(sequence[task.latent_slice.start])

    np.testing.assert_allclose(
        model.latent_voltage_view, expected_h0, rtol=1e-6, atol=1e-6
    )
    assert np.any(
        (np.asarray(model.latent_voltage_view) != 0.0)
        & (np.asarray(model.latent_voltage_view) != 1.0)
    )


def test_query_terminal_read_and_seeded_h0_decode_identically() -> None:
    model = _model(latent=2)
    task = model.config.task
    episode = generate_episode(task, brainstate.random.RandomState(2748))
    sequence = jnp.asarray(episode.model_inputs)[:, None, :]

    query_logits = brainstate.transform.for_loop(
        model, sequence[: task.query_slice.stop]
    )[-1]
    model(sequence[task.latent_slice.start])

    np.testing.assert_allclose(
        query_logits,
        model.latent_voltage_view @ model.Wo.value,
        rtol=1e-5,
        atol=1e-5,
    )


def test_seeding_the_workspace_leaves_stored_bindings_untouched() -> None:
    model = _model(latent=2)
    task = model.config.task
    episode = generate_episode(task, brainstate.random.RandomState(2750))
    sequence = jnp.asarray(episode.model_inputs)[:, None, :]

    brainstate.transform.for_loop(model, sequence[: task.demonstration_steps])
    values_before, keys_before = model.memory_factors()
    values_before = np.asarray(values_before).copy()
    keys_before = np.asarray(keys_before).copy()
    brainstate.transform.for_loop(model, sequence[task.demonstration_steps :])
    values_after, keys_after = model.memory_factors()

    np.testing.assert_array_equal(values_after, values_before)
    np.testing.assert_array_equal(keys_after, keys_before)


def test_binary_spike_surrogate_has_finite_nonzero_training_gradient() -> None:
    voltage_minus_threshold = jnp.asarray([-0.5, -0.1, 0.1, 0.5])

    spikes = _surrogate_spike(voltage_minus_threshold)
    gradient = jax.grad(lambda value: jnp.sum(_surrogate_spike(value)))(
        voltage_minus_threshold
    )

    np.testing.assert_array_equal(spikes, jnp.asarray([0.0, 0.0, 1.0, 1.0]))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) > 0.0)


def test_production_latent_dynamics_retain_activity_at_depth_eight() -> None:
    task = _task(bindings=2, slots=3, latent=8)
    model = LatentWorkspaceModel(
        ModelConfig(task=task, batch_size=32, latent_width=64, seed=275)
    )
    random = brainstate.random.RandomState(276)
    initial_spikes = random.bernoulli(0.35, size=(32, 64)).astype(jnp.float32)
    initial_voltage = random.uniform(0.65, 0.95, size=(32, 64))
    grouped = model.grouped_state.value.reshape(32, model.state_rows, 64)
    model.grouped_state.value = (
        grouped.at[:, -1].set(initial_spikes).reshape(32 * model.state_rows, 64)
    )
    voltage = model.latent_voltage.value.reshape(32, model.state_rows, 64)
    model.latent_voltage.value = (
        voltage.at[:, -1].set(initial_voltage).reshape(32 * model.state_rows, 64)
    )
    latent_tick = jnp.zeros((32, task.input_width), dtype=jnp.float32)
    latent_tick = latent_tick.at[:, task.phase_slice.start + 3].set(1.0)
    latent_ticks = jnp.broadcast_to(latent_tick, (8, *latent_tick.shape))

    def step(one_tick: jax.Array) -> jax.Array:
        model(one_tick)
        return model.workspace

    trajectory = brainstate.transform.for_loop(step, latent_ticks)
    retention = float(jnp.mean(trajectory[-1]) / jnp.mean(initial_spikes))

    assert retention >= 0.25


def test_run_sequence_uses_brainstate_loop_without_python_driver() -> None:
    source = inspect.getsource(run_sequence)
    tree = ast.parse(source)

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "for_loop"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "scan"
        for call in calls
    )


def test_invalid_model_configuration_names_the_quantity() -> None:
    with pytest.raises(ValueError, match="latent_width"):
        ModelConfig(task=_task(), latent_width=0)
    with pytest.raises(ValueError, match="latent_tau_ms"):
        ModelConfig(task=_task(), latent_tau_ms=0.0)
    with pytest.raises(ValueError, match="latent_spectral_radius"):
        ModelConfig(task=_task(), latent_spectral_radius=-0.1)
    with pytest.raises(ValueError, match="write_mode"):
        ModelConfig(task=_task(), write_mode="mystery")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        ModelConfig(task=_task(), batch_size=True)
    with pytest.raises(ValueError, match="latent_width"):
        ModelConfig(task=_task(), latent_width=8.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_jacobian_elements"):
        ModelConfig(task=_task(), max_jacobian_elements=0)
    with pytest.raises(ValueError, match="projection_seed"):
        ModelConfig(task=_task(), projection_seed=True)
    with pytest.raises(ValueError, match="projection_seed"):
        ModelConfig(task=_task(), projection_seed=1.5)  # type: ignore[arg-type]


def test_run_sequence_rejects_invalid_phase_or_missing_query() -> None:
    model = _model()
    episode = generate_episode(model.config.task, brainstate.random.RandomState(277))
    invalid = np.asarray(episode.model_inputs).copy()
    invalid[0, model.config.task.phase_slice.start + 1] = 1.0

    with pytest.raises(ValueError, match="phase"):
        run_sequence(model, invalid)
    with pytest.raises(ValueError, match="query"):
        run_sequence(
            model,
            jnp.asarray(episode.model_inputs[: model.config.task.demonstration_steps]),
        )


def test_tiny_coupled_compile_is_warning_free_and_relates_write_weights() -> None:
    task = _task(bindings=1, slots=2, latent=0)
    model = LatentWorkspaceModel(
        ModelConfig(task=task, latent_width=4, batch_size=1, seed=28)
    )
    sample = jnp.zeros((1, task.input_width), dtype=jnp.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner = braintrace.compile(
            model,
            model.etrace_config(),
            sample,
            batch_size=1,
            vmap=False,
            verbose=0,
        )
        jax.block_until_ready(learner(sample))

    assert not caught
    assert not [
        record
        for record in learner.graph.diagnostics
        if record.level
        in (braintrace.DiagnosticLevel.WARNING, braintrace.DiagnosticLevel.ERROR)
    ]
    relations = {".".join(map(str, path)) for path, _ in learner.report.etrace_weights}
    assert {"Wk", "Wv", "Wf"}.issubset(relations)
    memory_groups = [
        group
        for group in learner.graph.hidden_groups
        if ("grouped_state",) in set(group.hidden_paths)
    ]
    assert len(memory_groups) == 1
    primitives = [
        equation.primitive.name for equation in memory_groups[0].transition_jaxpr.eqns
    ]
    assert primitives.count("dot_general") == 2


def test_default_native_batch_four_coupled_budget_compiles() -> None:
    task = TaskConfig()
    model = LatentWorkspaceModel(
        ModelConfig(task=task, batch_size=4, latent_width=32, seed=2108)
    )
    config = model.etrace_config()
    sample = jnp.zeros((4, task.input_width), dtype=jnp.float32)

    learner = braintrace.compile(
        model,
        config,
        sample,
        batch_size=4,
        vmap=False,
        verbose=0,
        **model.compile_options(),
    )

    groups = [
        group
        for group in learner.graph.hidden_groups
        if ("grouped_state",) in set(group.hidden_paths)
    ]
    assert len(groups) == 1
    group = groups[0]
    assert set(group.hidden_paths) == {
        ("grouped_state",),
        ("query_encoding",),
        ("latent_voltage",),
    }
    assert group.varshape == (68, 32)
    jacobian_elements = (int(np.prod(group.varshape)) * len(group.hidden_paths)) ** 2
    assert jacobian_elements == 42_614_784
    assert model.compile_options()["snap_max_jacobian_elements"] == 1 << 26
