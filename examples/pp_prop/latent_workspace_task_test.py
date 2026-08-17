"""Tests for Example 21's latent-workspace episode task."""

import hashlib
from dataclasses import replace

import brainstate
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from examples.pp_prop import latent_workspace_task as task_module
from examples.pp_prop.latent_workspace_task import (
    Episode,
    MatchedEpisodes,
    TaskConfig,
    build_codebook,
    build_episode,
    draw_rule,
    generate_episode,
    generate_matched_episodes,
    oracle_answer,
)


def _rng(seed: int = 0) -> brainstate.random.RandomState:
    return brainstate.random.RandomState(seed)


def _valid_metadata(config: TaskConfig):
    rule = np.roll(np.arange(config.symbol_count, dtype=np.int32), -1)
    query = 0
    keys = np.arange(config.binding_count, dtype=np.int32)
    return {
        "rule": rule,
        "demonstration_keys": keys,
        "demonstration_values": rule[keys],
        "query_symbol": query,
        "terminal_target": int(rule[query]),
        "condition": "supported",
    }


def test_default_codebook_matches_measured_feasibility_fingerprint():
    config = TaskConfig()
    first = build_codebook(config)
    repeated = build_codebook(config)

    assert first.shape == (10, 4, 24)
    assert np.array_equal(first, repeated)
    assert hashlib.sha256(first.tobytes()).hexdigest() == (
        "46ac5df9d15f0bf2bdf6fb72230acbddd22c423d4b1fb57947a120d90dc243c1"
    )
    assert float(first.mean()) == pytest.approx(0.24895833333333334)
    flattened = first.reshape(config.symbol_count, -1)
    augmented = np.column_stack((flattened, np.ones(config.symbol_count)))
    pairwise_hamming = np.sum(flattened[:, None, :] != flattened[None, :, :], axis=-1)
    pairwise_hamming += np.eye(config.symbol_count, dtype=np.int64) * flattened.shape[1]
    assert np.unique(flattened, axis=0).shape[0] == config.symbol_count
    assert np.linalg.matrix_rank(augmented) == config.symbol_count
    assert int(pairwise_hamming.min()) == 29


def test_default_codebook_is_attempt_zero_of_a_local_brainstate_stream():
    config = TaskConfig()
    expected = np.asarray(
        brainstate.random.RandomState(config.codebook_seed).bernoulli(
            config.spike_rate,
            size=(config.symbol_count, config.symbol_ticks, config.code_width),
        ),
        dtype=np.float32,
    )

    assert np.array_equal(build_codebook(config), expected)


def test_build_codebook_does_not_consume_the_global_brainstate_stream():
    with brainstate.random.seed_context(8712):
        expected_before = np.asarray(brainstate.random.uniform(size=(8,)))
        expected_after = np.asarray(brainstate.random.uniform(size=(8,)))
    with brainstate.random.seed_context(8712):
        actual_before = np.asarray(brainstate.random.uniform(size=(8,)))
        build_codebook(TaskConfig())
        actual_after = np.asarray(brainstate.random.uniform(size=(8,)))

    assert np.array_equal(actual_before, expected_before)
    assert np.array_equal(actual_after, expected_after)


def test_codebook_seed_reproduces_bytes_and_changes_the_fingerprint():
    base = TaskConfig(codebook_seed=41)

    first = build_codebook(base)
    repeated = build_codebook(base)
    different = build_codebook(replace(base, codebook_seed=42))

    assert first.tobytes() == repeated.tobytes()
    assert first.tobytes() != different.tobytes()


def test_codebook_retries_seed_plus_attempt_until_design_is_usable(monkeypatch):
    seeds = []

    class _RetryStream:
        def __init__(self, seed):
            seeds.append(seed)

        def bernoulli(self, probability, *, size):
            del probability
            codebook = np.zeros(size, dtype=np.bool_)
            if len(seeds) > 1:
                flattened = codebook.reshape(size[0], -1)
                flattened[np.arange(size[0]), np.arange(size[0])] = True
            return codebook

    monkeypatch.setattr(brainstate.random, "RandomState", _RetryStream)

    codebook = build_codebook(TaskConfig(codebook_seed=700))

    assert seeds == [700, 701]
    assert np.linalg.matrix_rank(codebook.reshape(10, -1)) == 10


def test_codebook_retry_exhaustion_names_seed_and_bound(monkeypatch):
    seeds = []

    class _DuplicateStream:
        def __init__(self, seed):
            seeds.append(seed)

        def bernoulli(self, probability, *, size):
            del probability
            return np.zeros(size, dtype=np.bool_)

    monkeypatch.setattr(brainstate.random, "RandomState", _DuplicateStream)
    monkeypatch.setattr(task_module, "_MAX_CODEBOOK_ATTEMPTS", 3)

    with pytest.raises(
        ValueError,
        match=r"codebook_seed 700.*3 attempts.*unique.*full-rank",
    ):
        build_codebook(TaskConfig(codebook_seed=700))

    assert seeds == [700, 701, 702]


def test_flat_axis_has_ordered_phases_silent_latency_and_terminal_target():
    config = TaskConfig(binding_count=3, latent_steps=5)
    episode = generate_episode(config, _rng(1))
    phases = episode.model_inputs[:, config.phase_slice]

    assert episode.model_inputs.shape == (21, config.input_width)
    assert np.all(phases.sum(axis=1) == 1)
    assert np.all(phases[:12] == (1.0, 0.0, 0.0, 0.0))
    assert np.all(phases[12:16] == (0.0, 1.0, 0.0, 0.0))
    assert np.all(phases[16] == (0.0, 0.0, 1.0, 0.0))
    assert np.all(phases[17:] == (0.0, 0.0, 0.0, 1.0))
    external = episode.model_inputs[
        config.latent_slice, slice(0, config.slot_slice.stop)
    ]
    assert not np.any(external)
    assert episode.terminal_index == 20
    assert episode.target == episode.terminal_target


def test_zero_latent_steps_reads_the_last_query_tick():
    config = TaskConfig(latent_steps=0)
    episode = generate_episode(config, _rng(2))

    assert episode.latent_inputs.shape == (0, config.input_width)
    assert episode.terminal_index == config.query_slice.stop - 1
    assert episode.model_inputs[
        episode.terminal_index, config.phase_slice
    ].tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]


def test_oracle_agreement_is_checked_independently_of_generator():
    episode = generate_episode(TaskConfig(), _rng(3))
    independent_mapping = dict(enumerate(episode.rule.tolist()))

    for key, value in episode.demonstration_pairs:
        assert independent_mapping[key] == value
    assert independent_mapping[episode.query_symbol] == episode.terminal_target
    assert sorted(independent_mapping.values()) == list(range(10))


def test_rules_vary_across_episodes_from_one_seed_sequence():
    rng = _rng(4)
    rules = {
        tuple(generate_episode(TaskConfig(), rng).rule.tolist()) for _ in range(12)
    }

    assert len(rules) > 1


def test_model_tensor_has_no_rule_episode_or_terminal_target_field():
    config = TaskConfig()
    pair = generate_matched_episodes(config, _rng(5))
    for episode in (pair.supported, pair.short):
        assert episode.model_inputs.shape[1] == 2 * 24 + 8 + 4 + config.clock_width
        binary_columns = episode.model_inputs[:, : config.clock_slice.start]
        assert set(np.unique(binary_columns)).issubset({0.0, 1.0})
        assert not np.any(episode.query_inputs[:, config.value_slice])
        assert not np.any(episode.query_inputs[:, config.slot_slice])
        assert not np.any(episode.latent_inputs[:, : config.slot_slice.stop])
        assert np.array_equal(
            episode.query_inputs[:, config.key_slice],
            episode.codebook[episode.query_symbol],
        )


def test_slot_address_is_phase_local_and_identifies_only_demonstration_position():
    config = TaskConfig(binding_count=4)
    episode = generate_episode(config, _rng(51))
    slots = episode.model_inputs[:, config.slot_slice]

    for slot in range(config.binding_count):
        span = slice(slot * config.symbol_ticks, (slot + 1) * config.symbol_ticks)
        assert np.all(slots[span, slot] == 1.0)
        assert np.all(slots[span].sum(axis=1) == 1.0)
    assert not np.any(slots[config.query_slice])
    assert not np.any(slots[config.latent_slice])


def test_matched_views_share_query_target_and_differ_in_one_binding():
    pair = generate_matched_episodes(TaskConfig(), _rng(6))
    supported, short = pair.supported, pair.short

    assert supported.query_symbol == short.query_symbol
    assert supported.terminal_target == short.terminal_target
    assert supported.rule.tobytes() == short.rule.tobytes()
    assert supported.query_inputs.tobytes() == short.query_inputs.tobytes()
    assert supported.latent_inputs.tobytes() == short.latent_inputs.tobytes()
    assert (
        sum(
            left != right
            for left, right in zip(
                supported.demonstration_pairs, short.demonstration_pairs, strict=True
            )
        )
        == 1
    )
    assert (
        sum(key == supported.query_symbol for key, _ in supported.demonstration_pairs)
        == 1
    )
    assert (
        supported.query_symbol,
        supported.terminal_target,
    ) in supported.demonstration_pairs
    assert all(
        supported.query_symbol not in binding for binding in short.demonstration_pairs
    )


def test_same_seed_reproduces_encoded_pair_exactly():
    first = generate_matched_episodes(TaskConfig(), _rng(7))
    second = generate_matched_episodes(TaskConfig(), _rng(7))

    for condition in ("supported", "short"):
        left = getattr(first, condition)
        right = getattr(second, condition)
        assert left.model_inputs.tobytes() == right.model_inputs.tobytes()
        assert left.rule.tobytes() == right.rule.tobytes()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symbol_count": 9}, "symbol_count"),
        ({"binding_count": 0}, "binding_count"),
        ({"slot_capacity": 0}, "slot_capacity"),
        ({"latent_steps": -1}, "latent_steps"),
        ({"code_width": 9}, "code_width"),
        ({"symbol_ticks": 0}, "symbol_ticks"),
        ({"codebook_seed": -1}, "codebook_seed"),
        ({"spike_rate": 0.0}, "spike_rate"),
    ],
)
def test_malformed_config_names_the_quantity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TaskConfig(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "symbol_count",
        "binding_count",
        "slot_capacity",
        "latent_steps",
        "code_width",
        "symbol_ticks",
        "codebook_seed",
    ],
)
@pytest.mark.parametrize("value", [True, 4.0, float("nan"), float("inf")])
def test_integer_config_fields_require_finite_non_boolean_integral_scalars(
    field, value
):
    with pytest.raises(ValueError, match=field):
        TaskConfig(**{field: value})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), [0.25]])
def test_spike_rate_requires_a_finite_non_boolean_real_scalar(value):
    with pytest.raises(ValueError, match="spike_rate"):
        TaskConfig(spike_rate=value)


def test_binding_overflow_names_requested_count_and_capacity():
    with pytest.raises(ValueError, match=r"binding_count 9 exceeds slot_capacity 8"):
        TaskConfig(binding_count=9, slot_capacity=8, symbol_count=12)


def test_short_context_capacity_constraint_names_both_quantities():
    with pytest.raises(ValueError, match=r"symbol_count.*binding_count \+ 2"):
        TaskConfig(symbol_count=10, binding_count=9, slot_capacity=9)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rule": np.arange(9)}, "rule shape"),
        ({"rule": np.zeros(10, dtype=int)}, "rule must be a bijection"),
        ({"demonstration_keys": [0, 0, 1, 2]}, "demonstration_keys.*distinct"),
        ({"demonstration_keys": [0, 1, 2, 20]}, "demonstration_keys.*symbol_count"),
        ({"demonstration_values": [3, 2, 3, 4]}, "demonstration_values disagree"),
        ({"terminal_target": 9}, "terminal_target disagrees"),
        ({"query_symbol": 10}, "query_symbol.*symbol_count"),
        ({"condition": "unknown"}, "condition"),
    ],
)
def test_malformed_episode_data_names_the_quantity(change, message):
    config = TaskConfig()
    metadata = _valid_metadata(config)
    metadata.update(change)

    with pytest.raises(ValueError, match=message):
        build_episode(config, **metadata)


def test_condition_specific_malformed_data_is_rejected():
    config = TaskConfig()
    metadata = _valid_metadata(config)
    metadata["condition"] = "short"
    with pytest.raises(ValueError, match="short condition.*query_symbol"):
        build_episode(config, **metadata)

    metadata = _valid_metadata(config)
    metadata["demonstration_keys"] = np.arange(1, 5)
    metadata["demonstration_values"] = metadata["rule"][1:5]
    with pytest.raises(ValueError, match="supported condition.*once"):
        build_episode(config, **metadata)


def test_public_oracle_and_rule_draw_validate_their_inputs():
    with pytest.raises(ValueError, match="symbol_count"):
        draw_rule(0, _rng())
    with pytest.raises(ValueError, match="rule shape"):
        oracle_answer(np.zeros((2, 2), dtype=int), 0)
    with pytest.raises(ValueError, match="bijection"):
        oracle_answer([0, 0], 0)
    with pytest.raises(ValueError, match="query_symbol"):
        oracle_answer([0, 1], 2)
    with pytest.raises(ValueError, match="condition"):
        generate_episode(TaskConfig(), _rng(), condition="other")


@pytest.mark.parametrize("value", [True, 10.0, float("nan"), float("inf")])
def test_draw_rule_symbol_count_requires_a_finite_integral_scalar(value):
    with pytest.raises(ValueError, match="symbol_count"):
        draw_rule(value, _rng())


@pytest.mark.parametrize("value", [True, 0.0, float("nan"), float("inf")])
def test_oracle_query_symbol_requires_a_finite_integral_scalar(value):
    with pytest.raises(ValueError, match="query_symbol"):
        oracle_answer(np.arange(10, dtype=np.int32), value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_symbol", True),
        ("query_symbol", 0.0),
        ("query_symbol", float("nan")),
        ("terminal_target", True),
        ("terminal_target", 1.0),
        ("terminal_target", float("inf")),
    ],
)
def test_episode_scalar_symbols_require_finite_non_boolean_integral_values(
    field, value
):
    config = TaskConfig()
    metadata = _valid_metadata(config)
    metadata[field] = value

    with pytest.raises(ValueError, match=field):
        build_episode(config, **metadata)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        (
            "rule",
            np.array([2**32, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint64),
        ),
        (
            "demonstration_keys",
            np.array([2**40, 1, 2, 3], dtype=np.int64),
        ),
        (
            "demonstration_values",
            np.array([2**40, 2, 3, 4], dtype=np.int64),
        ),
    ],
)
def test_episode_symbol_arrays_are_range_checked_before_int32_cast(field, values):
    config = TaskConfig()
    metadata = _valid_metadata(config)
    metadata[field] = values

    with pytest.raises(ValueError, match=field):
        build_episode(config, **metadata)


@pytest.mark.parametrize(
    "rule",
    [
        np.arange(10, dtype=np.float64),
        np.arange(10, dtype=np.int32).astype(bool),
        np.array([2**32, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint64),
    ],
)
def test_oracle_validates_raw_rule_values_before_indexing(rule):
    with pytest.raises(ValueError, match="rule"):
        oracle_answer(rule, 0)


def test_invalid_condition_does_not_consume_the_random_stream():
    config = TaskConfig()
    rejected_rng = _rng(91)
    control_rng = _rng(91)

    with pytest.raises(ValueError, match="condition"):
        generate_episode(config, rejected_rng, condition="other")

    after_rejection = generate_matched_episodes(config, rejected_rng)
    control = generate_matched_episodes(config, control_rng)
    assert after_rejection.supported.rule.tobytes() == control.supported.rule.tobytes()
    assert (
        after_rejection.supported.model_inputs.tobytes()
        == control.supported.model_inputs.tobytes()
    )


@pytest.mark.parametrize("public_class", [TaskConfig, Episode, MatchedEpisodes])
def test_public_dataclass_docstrings_describe_attributes(public_class):
    assert "Attributes\n----------" in public_class.__doc__


@st.composite
def _valid_sizes(draw):
    symbol_count = draw(st.integers(min_value=10, max_value=20))
    binding_count = draw(st.integers(min_value=1, max_value=min(8, symbol_count - 2)))
    capacity = draw(st.integers(min_value=binding_count, max_value=10))
    latent_steps = draw(st.integers(min_value=0, max_value=8))
    return symbol_count, binding_count, capacity, latent_steps


@given(_valid_sizes(), st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_symbol_and_binding_property(size, seed):
    symbol_count, binding_count, capacity, latent_steps = size
    config = TaskConfig(
        symbol_count=symbol_count,
        binding_count=binding_count,
        slot_capacity=capacity,
        latent_steps=latent_steps,
    )
    pair = generate_matched_episodes(config, _rng(seed))

    assert pair.supported.model_inputs.shape == (config.total_steps, config.input_width)
    assert pair.supported.query_inputs.tobytes() == pair.short.query_inputs.tobytes()
    assert pair.supported.terminal_target == pair.short.terminal_target
    assert (
        sum(
            key == pair.supported.query_symbol
            for key, _ in pair.supported.demonstration_pairs
        )
        == 1
    )
    assert all(
        pair.short.query_symbol not in binding
        for binding in pair.short.demonstration_pairs
    )
    assert pair.supported.terminal_index == config.total_steps - 1


@st.composite
def _valid_codebook_configs(draw):
    symbol_count = draw(st.integers(min_value=10, max_value=20))
    code_width = draw(st.integers(min_value=symbol_count, max_value=32))
    return TaskConfig(
        symbol_count=symbol_count,
        code_width=code_width,
        symbol_ticks=draw(st.integers(min_value=2, max_value=6)),
        spike_rate=draw(st.sampled_from((0.15, 0.25, 0.5, 0.75, 0.85))),
        codebook_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )


@given(_valid_codebook_configs())
@settings(max_examples=40, deadline=None)
def test_codebook_property_is_deterministic_unique_and_augmented_full_rank(config):
    first = build_codebook(config)
    second = build_codebook(config)
    flattened = first.reshape(config.symbol_count, -1)
    augmented = np.column_stack((flattened, np.ones(config.symbol_count)))

    assert first.tobytes() == second.tobytes()
    assert np.unique(flattened, axis=0).shape[0] == config.symbol_count
    assert np.linalg.matrix_rank(augmented) == config.symbol_count
    assert float(first.mean()) == pytest.approx(float(first.sum()) / float(first.size))


def test_slice_helpers_are_stable_under_nondefault_dimensions():
    config = replace(TaskConfig(), code_width=20, symbol_count=10, slot_capacity=9)

    assert config.key_slice == slice(0, 20)
    assert config.value_slice == slice(20, 40)
    assert config.slot_slice == slice(40, 49)
    assert config.phase_slice == slice(49, 53)
    assert config.clock_slice == slice(53, 53 + config.clock_width)
    assert config.input_width == 53 + config.clock_width


def test_clock_code_repeats_only_at_the_slowest_period() -> None:
    code = task_module.latent_clock_code(8, 4)

    assert code.shape == (8, 4)
    assert np.allclose(code[0], code[4], atol=1e-5)
    for step in range(1, 4):
        assert not np.allclose(code[0], code[step])


def test_clock_code_width_is_independent_of_latent_depth() -> None:
    shallow = task_module.latent_clock_code(2, 6)
    deep = task_module.latent_clock_code(64, 6)

    assert shallow.shape[1] == deep.shape[1] == 6
    assert np.allclose(deep[:2], shallow)


def test_zero_latent_steps_yields_an_empty_clock_code() -> None:
    assert task_module.latent_clock_code(0, 4).shape == (0, 4)


@pytest.mark.parametrize("clock_width", [0, -2, 3])
def test_invalid_clock_width_is_rejected(clock_width: int) -> None:
    with pytest.raises(ValueError):
        TaskConfig(clock_width=clock_width)


def test_clock_bank_is_confined_to_the_latent_span() -> None:
    config = TaskConfig(latent_steps=6)
    episode = generate_episode(config, _rng(77))
    clock = episode.model_inputs[:, config.clock_slice]

    assert not np.any(clock[: config.latent_slice.start])
    assert np.any(clock[config.latent_slice])
    assert np.array_equal(
        clock[config.latent_slice],
        task_module.latent_clock_code(config.latent_steps, config.clock_width),
    )


def test_latent_span_drive_differs_between_successive_ticks() -> None:
    config = TaskConfig(latent_steps=4)
    episode = generate_episode(config, _rng(78))
    latent_rows = episode.model_inputs[config.latent_slice]

    assert not np.allclose(latent_rows[1], latent_rows[2])
