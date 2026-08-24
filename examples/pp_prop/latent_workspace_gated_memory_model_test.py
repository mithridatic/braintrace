"""Tests for the V44 phase-separated gated relation memory."""

from __future__ import annotations

import importlib

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_gated_memory_model"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _config(subject, **overrides):
    values = {
        "input_width": subject.MODEL_INPUT_WIDTH,
        "memory_width": 16,
        "expert_count": 12,
        "seed": 23,
    }
    values.update(overrides)
    return subject.GatedMemoryConfig(**values)


def _event(model, *, demonstration: bool = False, query: bool = False):
    event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    event = event.at[:, 0].set(1.0)
    if demonstration:
        event = event.at[:, model.config.demonstration_phase_index].set(1.0)
    if query:
        event = event.at[:, model.config.query_phase_index].set(1.0)
    return event.at[:, 20].set(1.0)


def test_v44_config_fails_closed_and_binds_relation_width() -> None:
    subject = _subject()
    config = _config(subject)

    assert config.hidden_width == 48
    assert config.architecture_version == "phase_separated_gated_memory_v44"
    with pytest.raises(ValueError, match="input_width"):
        subject.GatedMemoryConfig(input_width=8)
    with pytest.raises(ValueError, match="memory_width"):
        _config(subject, memory_width=0)
    with pytest.raises(ValueError, match="expert_count"):
        _config(subject, expert_count=1)
    with pytest.raises(ValueError, match="hidden_width"):
        _config(subject, memory_width=2, expert_count=12)
    with pytest.raises(ValueError, match="architecture_version"):
        _config(subject, architecture_version="old")
    with pytest.raises(TypeError, match="seed"):
        _config(subject, seed=True)
    with pytest.raises(ValueError, match="seed"):
        _config(subject, seed=-1)
    with pytest.raises(TypeError, match="config"):
        subject.PhaseSeparatedGatedMemoryRNN(object())


def test_v44_phase_gates_preserve_demo_and_query_memories() -> None:
    subject = _subject()
    model = subject.PhaseSeparatedGatedMemoryRNN(_config(subject, seed=29))
    brainstate.nn.init_all_states(model, batch_size=1)
    zero_demo = np.asarray(model.recurrent[0].h.value).copy()
    zero_query = np.asarray(model.recurrent[1].h.value).copy()

    model(_event(model, demonstration=True))
    demo_after_demo = np.asarray(model.recurrent[0].h.value).copy()
    query_after_demo = np.asarray(model.recurrent[1].h.value).copy()
    model(_event(model, query=True))
    demo_after_query = np.asarray(model.recurrent[0].h.value).copy()
    query_after_query = np.asarray(model.recurrent[1].h.value).copy()
    model(jnp.zeros((1, model.config.input_width), dtype=jnp.float32))

    assert demo_after_demo.tobytes() != zero_demo.tobytes()
    assert query_after_demo.tobytes() == zero_query.tobytes()
    assert demo_after_query.tobytes() == demo_after_demo.tobytes()
    assert query_after_query.tobytes() != query_after_demo.tobytes()
    assert np.asarray(model.recurrent[0].h.value).tobytes() == demo_after_query.tobytes()
    assert np.asarray(model.recurrent[1].h.value).tobytes() == query_after_query.tobytes()


def test_v44_relation_activation_depends_on_both_memories() -> None:
    subject = _subject()
    model = subject.PhaseSeparatedGatedMemoryRNN(_config(subject))
    demo = jnp.zeros((1, model.config.memory_width), dtype=jnp.float32)
    query = jnp.zeros_like(demo)
    changed_demo = demo.at[0, 3].set(2.0)
    changed_query = query.at[0, 3].set(4.0)

    baseline = np.asarray(model._relation_hidden(demo, query))
    demo_relation = np.asarray(model._relation_hidden(changed_demo, query))
    query_relation = np.asarray(model._relation_hidden(demo, changed_query))
    joint = np.asarray(model._relation_hidden(changed_demo, changed_query))

    assert baseline.shape == (1, model.config.hidden_width)
    assert baseline.tobytes() != demo_relation.tobytes()
    assert baseline.tobytes() != query_relation.tobytes()
    assert joint[0, -model.config.memory_width + 3] == pytest.approx(8.0)


def test_v44_expert_gate_is_normalized_and_relation_dependent() -> None:
    subject = _subject()
    model = subject.PhaseSeparatedGatedMemoryRNN(_config(subject))
    relation = jnp.zeros((2, model.config.hidden_width), dtype=jnp.float32)
    changed = relation.at[0, 5].set(5.0)

    baseline = np.asarray(model._expert_weights(relation))
    observed = np.asarray(model._expert_weights(changed))

    assert np.allclose(np.sum(baseline, axis=-1), 1.0)
    assert np.allclose(np.sum(observed, axis=-1), 1.0)
    assert baseline[0].tobytes() != observed[0].tobytes()
    assert baseline[1].tobytes() == observed[1].tobytes()


def test_v44_compiler_includes_every_recurrent_synapse() -> None:
    subject = _subject()
    model = subject.PhaseSeparatedGatedMemoryRNN(_config(subject, seed=31))
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, model.config.input_width), dtype=jnp.float32),
        batch_size=1,
        vmap=False,
        decay_or_rank=0.9,
        vjp_method="single-step",
    )

    recurrent_included = {
        tuple(path)
        for path, _ in learner.report.etrace_weights
        if path[0] == "recurrent"
    }
    recurrent_excluded = [
        path for path, _ in learner.report.excluded_weights if path[0] == "recurrent"
    ]

    assert len(recurrent_included) == 6
    assert recurrent_excluded == []


def test_v44_model_emits_finite_fixed_layout_logits() -> None:
    subject = _subject()
    model = subject.PhaseSeparatedGatedMemoryRNN(_config(subject, seed=37))
    brainstate.nn.init_all_states(model, batch_size=2)

    output = np.asarray(
        model(jnp.zeros((2, model.config.input_width), dtype=jnp.float32))
    )

    assert output.shape == (2, subject.OUTPUT_WIDTH)
    assert np.all(np.isfinite(output))
    paths = {str(path[0]) for path in model.states(brainstate.ParamState)}
    assert paths == {"recurrent", "cell_color_head", "height_head", "width_head"}
    with pytest.raises(ValueError, match="last dimension"):
        model(jnp.zeros((2, model.config.input_width - 1), dtype=jnp.float32))
