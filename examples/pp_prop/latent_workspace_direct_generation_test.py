"""Tests for direct checkpoint-owned ARC prediction and qualification."""

from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_generation"
    )


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
    ("proposal_source", "answer_head_version"),
    [
        ("online_model_logits", "online_row_decoder_v20"),
        ("online_model_logits", "hierarchical_row_decoder_v29"),
        ("online_model_logits", "task_conditioned_shared_cell_decoder_v36"),
        ("online_model_logits", "task_conditioned_query_patch_decoder_v39"),
        ("spatial_model_logits", "spatial_conv_lif_row_decoder_v22"),
        (
            "continuous_spatial_model_logits",
            "continuous_spatial_row_decoder_v41",
        ),
        ("task_gated_model_logits", "task_gated_operator_bank_v42"),
        ("gated_memory_model_logits", "phase_separated_gated_memory_v44"),
    ],
)
def test_candidate_validator_accepts_only_bound_neural_source_head_pair(
    proposal_source: str, answer_head_version: str
) -> None:
    subject = _subject()
    candidate = subject.decode_first_candidate(_logits())
    candidate["proposal_source"] = proposal_source
    candidate["answer_head_version"] = answer_head_version

    subject.validate_direct_candidate(candidate)
    assert subject.first_prediction_bytes([candidate])

    candidate["answer_head_version"] = "direct_model_generation_v1"
    with pytest.raises(ValueError, match="source/head"):
        subject.validate_direct_candidate(candidate)


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
