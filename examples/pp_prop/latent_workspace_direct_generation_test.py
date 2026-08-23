"""Tests for direct checkpoint-owned ARC prediction and qualification."""

from __future__ import annotations

import copy
import importlib

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_generation"
    )


def _model_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_direct_model")


def _logits(*, first_color: int = 7):
    subject = _subject()
    height = np.full((30,), -5.0, dtype=np.float32)
    width = np.full((30,), -5.0, dtype=np.float32)
    colors = np.full((30, 30, 10), -5.0, dtype=np.float32)
    height[1] = 4.0
    width[2] = 4.0
    colors[:2, :3, 0] = 1.0
    colors[0, 0, first_color] = 3.0
    return subject.DirectPredictionLogits(
        height=height,
        width=width,
        colors=colors,
        parameter_dependencies=("encoder.weight", "decoder.weight"),
    )


def test_direct_candidate_rejects_forest_reranking_loophole() -> None:
    old_conditioned_candidate = {
        "rank": 1,
        "height": 2,
        "width": 3,
        "grid": [[1, 2, 3], [4, 5, 6]],
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "proposal_source": "demonstration_fitted_forest",
        "ranking_source": "trained_network_factorized_candidate_log_probability",
        "answer_head_version": "checkpoint_conditioned_v1",
        "selection_role": "checkpoint_conditioned_rank",
        "parameter_dependencies": ["encoder.weight", "decoder.weight"],
        "forest_log_probability": -1.0,
        "network_log_probability": -2.0,
        "combined_log_probability": -3.0,
    }

    with pytest.raises(ValueError, match="direct model logits|forest|proposal"):
        _subject().validate_direct_candidate(old_conditioned_candidate)


def test_direct_decode_uses_shape_and_cell_argmax_only() -> None:
    candidate = _subject().decode_first_candidate(_logits())

    assert candidate["height"] == 2
    assert candidate["width"] == 3
    assert candidate["grid"] == [[7, 0, 0], [0, 0, 0]]
    assert candidate["proposal_source"] == "direct_model_logits"
    assert candidate["selection_role"] == "greedy_argmax"
    assert candidate["parameter_dependencies"] == [
        "encoder.weight",
        "decoder.weight",
    ]
    _subject().validate_direct_candidate(candidate)


def test_direct_decode_has_no_target_input() -> None:
    subject = _subject()
    baseline = subject.decode_first_candidate(_logits())
    target_a = np.zeros((2, 3), dtype=np.int8)
    target_b = np.full((9, 4), 9, dtype=np.int8)

    # Targets are scorer-only values and cannot be passed to the decoder.
    assert subject.first_prediction_bytes([baseline]) == subject.first_prediction_bytes(
        [subject.decode_first_candidate(_logits())]
    )
    assert not np.array_equal(target_a, target_b)
    with pytest.raises(TypeError):
        subject.decode_first_candidate(_logits(), target=target_a)


def test_first_prediction_bytes_exclude_metadata_and_scores() -> None:
    subject = _subject()
    candidate = subject.decode_first_candidate(_logits())
    altered = copy.deepcopy(candidate)
    altered["diagnostic_score"] = 123.5
    altered["timestamp"] = "not-bound"

    assert subject.first_prediction_bytes([candidate]) == subject.first_prediction_bytes(
        [altered]
    )
    altered["grid"][0][0] = 6
    assert subject.first_prediction_bytes([candidate]) != subject.first_prediction_bytes(
        [altered]
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_direct_decode_rejects_nonfinite_logits(value: float, message: str) -> None:
    logits = _logits()
    logits.height[0] = value

    with pytest.raises(ValueError, match=message):
        _subject().decode_first_candidate(logits)


def test_direct_decode_rejects_missing_or_duplicate_dependencies() -> None:
    subject = _subject()
    for dependencies in ((), ("decoder.weight", "decoder.weight")):
        logits = _logits()
        invalid = subject.DirectPredictionLogits(
            height=logits.height,
            width=logits.width,
            colors=logits.colors,
            parameter_dependencies=dependencies,
        )
        with pytest.raises((TypeError, ValueError), match="parameter_dependencies"):
            subject.decode_first_candidate(invalid)


def test_strict_task_threshold_is_the_only_acceptance_score() -> None:
    subject = _subject()

    assert subject.qualifies_strict_task_pass_at_1(15) is False
    assert subject.qualifies_strict_task_pass_at_1(16) is True
    assert subject.qualifies_strict_task_pass_at_1(400) is True
    for invalid in (-1, 401, 16.0, True):
        with pytest.raises((TypeError, ValueError)):
            subject.qualifies_strict_task_pass_at_1(invalid)


def test_strict_task_score_requires_every_query_exact() -> None:
    subject = _subject()
    predictions = [
        [[1]],
        [[2]],
        [[3]],
        [[4]],
    ]
    targets = [
        [[1]],
        [[2]],
        [[9]],
        [[4]],
    ]
    task_ids = ["a", "a", "b", "c"]

    score = subject.strict_task_pass_at_1(predictions, targets, task_ids)

    assert score["strict_task_pass_at_1_count"] == 2
    assert score["strict_task_ids"] == ["a", "c"]
    assert score["task_membership"] == {"a": True, "b": False, "c": True}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("height", np.zeros((29,), dtype=np.float32), "shape"),
        ("width", np.zeros((30,), dtype=np.int32), "floating"),
        ("colors", np.zeros((30, 30, 9), dtype=np.float32), "shape"),
    ],
)
def test_direct_decode_rejects_invalid_logit_schema(
    field: str, value: np.ndarray, message: str
) -> None:
    logits = _logits()
    invalid = _subject().DirectPredictionLogits(
        height=value if field == "height" else logits.height,
        width=value if field == "width" else logits.width,
        colors=value if field == "colors" else logits.colors,
        parameter_dependencies=logits.parameter_dependencies,
    )

    with pytest.raises(ValueError, match=message):
        _subject().decode_first_candidate(invalid)


@pytest.mark.parametrize(
    "corrupt",
    ["rank", "grid_shape", "grid_dtype", "grid_color", "dependencies", "forbidden"],
)
def test_direct_candidate_validation_fails_closed(corrupt: str) -> None:
    subject = _subject()
    candidate = subject.decode_first_candidate(_logits())
    if corrupt == "rank":
        candidate["rank"] = 2
    elif corrupt == "grid_shape":
        candidate["height"] = 3
    elif corrupt == "grid_dtype":
        candidate["grid"][0][0] = 1.5
    elif corrupt == "grid_color":
        candidate["grid"][0][0] = 10
    elif corrupt == "dependencies":
        candidate["parameter_dependencies"] = "decoder.weight"
    else:
        candidate["rule_name"] = "identity"

    with pytest.raises((TypeError, ValueError)):
        subject.validate_direct_candidate(candidate)


def test_strict_score_rejects_mismatched_or_invalid_inputs() -> None:
    subject = _subject()
    with pytest.raises(ValueError, match="equal lengths"):
        subject.strict_task_pass_at_1([[[0]]], [], ["a"])
    with pytest.raises(ValueError, match="task_id"):
        subject.strict_task_pass_at_1([[[0]]], [[[0]]], [""])
    with pytest.raises(ValueError, match="integer"):
        subject.strict_task_pass_at_1([[[0.5]]], [[[0]]], ["a"])


def test_direct_gru_emits_complete_shape_and_cell_logits() -> None:
    subject = _model_subject()
    config = subject.DirectModelConfig(
        input_width=12,
        encoder_width=8,
        hidden_width=16,
        decoder_width=12,
        recurrent_layers=2,
        seed=7,
    )
    model = subject.DirectARCGRU(config)
    events = jnp.zeros((5, 2, 12), dtype=jnp.float32)
    events = events.at[:, :, 0].set(1.0)
    query = jnp.zeros((2, 30, 30, 11), dtype=jnp.float32)
    query = query.at[:, 0, 0, 3].set(1.0)
    query = query.at[:, 0, 0, 10].set(1.0)
    brainstate.nn.init_all_states(model, batch_size=2)

    height, width, colors = model.run(events, query)

    assert height.shape == (2, 30)
    assert width.shape == (2, 30)
    assert colors.shape == (2, 30, 30, 10)
    assert np.all(np.isfinite(np.asarray(height)))
    assert np.all(np.isfinite(np.asarray(width)))
    assert np.all(np.isfinite(np.asarray(colors)))


def test_direct_gru_is_seed_deterministic() -> None:
    subject = _model_subject()
    config = subject.DirectModelConfig(
        input_width=6,
        encoder_width=4,
        hidden_width=8,
        decoder_width=6,
        seed=13,
    )
    events = jnp.ones((3, 1, 6), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)

    first_model = subject.DirectARCGRU(config)
    brainstate.nn.init_all_states(first_model, batch_size=1)
    first = tuple(np.asarray(value) for value in first_model.run(events, query))
    second_model = subject.DirectARCGRU(config)
    brainstate.nn.init_all_states(second_model, batch_size=1)
    second = tuple(np.asarray(value) for value in second_model.run(events, query))

    for left, right in zip(first, second, strict=True):
        assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize(
    "changes",
    [
        {"input_width": 0},
        {"hidden_width": 0},
        {"recurrent_layers": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_direct_model_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    subject = _model_subject()
    values = {
        "input_width": 12,
        "encoder_width": 8,
        "hidden_width": 16,
        "decoder_width": 12,
        "recurrent_layers": 1,
        "seed": 7,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        subject.DirectModelConfig(**values)
