"""Tests for the training-only direct ARC synthetic curriculum."""

from __future__ import annotations

import importlib
import inspect

import brainstate
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_curriculum"
    )


def test_curriculum_is_seed_deterministic_balanced_and_bounded() -> None:
    subject = _subject()
    config = subject.SyntheticCurriculumConfig(task_count=24, demonstrations=3)

    first = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(17)
    )
    second = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(17)
    )

    assert first.task_sha256 == second.task_sha256
    expected_counts = {family: 2 for family in subject.FAMILIES}
    expected_counts["pattern_label"] = 4
    assert first.schema_version == "direct_synthetic_curriculum_v3"
    assert first.family_counts == expected_counts
    assert first.family_counts == second.family_counts
    assert len(first.tasks) == 24
    for left, right in zip(first.tasks, second.tasks, strict=True):
        assert left == right
        assert len(left.train) == 3
        assert len(left.test) == 1
        for pair in (*left.train, *left.test):
            assert pair.output is not None
            assert 1 <= pair.input.height <= 30
            assert 1 <= pair.input.width <= 30
            assert 1 <= pair.output.height <= 30
            assert 1 <= pair.output.width <= 30


def test_curriculum_changes_with_brainstate_seed() -> None:
    subject = _subject()
    config = subject.SyntheticCurriculumConfig(task_count=12, demonstrations=2)

    first = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(3)
    )
    second = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(5)
    )

    assert first.task_sha256 != second.task_sha256


@pytest.mark.parametrize("seed", [23, 29, 31, 37])
def test_every_curriculum_family_has_its_declared_relation(seed: int) -> None:
    subject = _subject()
    result = subject.generate_synthetic_curriculum(
        subject.SyntheticCurriculumConfig(task_count=12, demonstrations=3),
        brainstate.random.RandomState(seed),
    )
    by_family = {
        task.task_id.split(":")[1]: task
        for task in result.tasks
        if task.task_id is not None
    }

    copy_task = by_family["copy"]
    assert all(
        pair.input == pair.output for pair in (*copy_task.train, *copy_task.test)
    )

    recolor = by_family["recolor"]
    assert all(
        pair.input.as_array().shape == pair.output.as_array().shape
        for pair in (*recolor.train, *recolor.test)
    )
    assert any(
        pair.input.as_array().tobytes() != pair.output.as_array().tobytes()
        for pair in (*recolor.train, *recolor.test)
    )

    dihedral = by_family["dihedral"]
    for pair in (*dihedral.train, *dihedral.test):
        assert sorted(pair.input.as_array().ravel()) == sorted(
            pair.output.as_array().ravel()
        )

    crop = by_family["crop"]
    assert all(
        pair.output.height <= pair.input.height
        and pair.output.width <= pair.input.width
        for pair in (*crop.train, *crop.test)
    )

    upscale = by_family["upscale"]
    assert all(
        pair.output.height > pair.input.height and pair.output.width > pair.input.width
        for pair in (*upscale.train, *upscale.test)
    )

    for family in ("count", "pattern_label"):
        task = by_family[family]
        assert all(
            pair.output.height == pair.output.width == 1
            for pair in (*task.train, *task.test)
        )

    selected = by_family["select_marked_region"]
    for pair in (*selected.train, *selected.test):
        input_array = pair.input.as_array()
        output_array = pair.output.as_array()
        assert output_array.shape == (1, 1)
        row_midpoint = input_array.shape[0] // 2
        column_midpoint = input_array.shape[1] // 2
        regions = (
            input_array[:row_midpoint, :column_midpoint],
            input_array[:row_midpoint, column_midpoint:],
            input_array[row_midpoint:, :column_midpoint],
            input_array[row_midpoint:, column_midpoint:],
        )
        base_and_marker_counts = []
        for region in regions:
            colors, counts = np.unique(region, return_counts=True)
            order = np.argsort(counts)
            base_and_marker_counts.append((int(colors[order[-1]]), int(counts[order[0]])))
        winning_base = max(base_and_marker_counts, key=lambda item: item[1])[0]
        assert int(output_array[0, 0]) == winning_base

    projected = by_family["project_marker"]
    for pair in (*projected.train, *projected.test):
        input_array = pair.input.as_array()
        output_array = pair.output.as_array()
        changed = np.argwhere(input_array != output_array)
        assert input_array.shape == output_array.shape
        assert len(changed) >= 2 and len(changed) % 2 == 0
        removed_markers = np.unique(
            input_array[(input_array != output_array) & (output_array == 0)]
        )
        assert len(removed_markers) == 1
        marker = int(removed_markers[0])
        assert np.count_nonzero((input_array == marker) & (output_array == 0)) > 0
        assert np.count_nonzero((input_array != marker) & (output_array == marker)) > 0

    completed = by_family["complete_corner"]
    for pair in (*completed.train, *completed.test):
        input_array = pair.input.as_array()
        output_array = pair.output.as_array()
        changed = np.argwhere(input_array != output_array)
        assert len(changed) >= 1
        assert np.all(input_array[tuple(changed.T)] == 0)
        assert len(np.unique(output_array[tuple(changed.T)])) == 1
        for row, column in changed:
            neighborhoods = []
            for row_start in (row - 1, row):
                for column_start in (column - 1, column):
                    if (
                        0 <= row_start < input_array.shape[0] - 1
                        and 0 <= column_start < input_array.shape[1] - 1
                    ):
                        neighborhoods.append(
                            input_array[
                                row_start : row_start + 2,
                                column_start : column_start + 2,
                            ]
                        )
            assert any(np.count_nonzero(block) == 3 for block in neighborhoods)

    mirrored = by_family["mirror_concat"]
    candidate_relations = (
        lambda array: np.concatenate((np.flipud(array), array), axis=0),
        lambda array: np.concatenate((array, np.flipud(array)), axis=0),
        lambda array: np.concatenate((np.fliplr(array), array), axis=1),
        lambda array: np.concatenate((array, np.fliplr(array)), axis=1),
    )
    pairs = (*mirrored.train, *mirrored.test)
    assert any(
        all(
            np.array_equal(relation(pair.input.as_array()), pair.output.as_array())
            for pair in pairs
        )
        for relation in candidate_relations
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"task_count": 0},
        {"task_count": True},
        {"demonstrations": 1},
        {"demonstrations": 11},
        {"max_grid_size": 31},
    ],
)
def test_curriculum_config_fails_closed(changes: dict[str, object]) -> None:
    subject = _subject()
    values = {"task_count": 7, "demonstrations": 3, "max_grid_size": 12}
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        subject.SyntheticCurriculumConfig(**values)


def test_curriculum_requires_brainstate_random_state() -> None:
    subject = _subject()

    with pytest.raises(TypeError, match="RandomState"):
        subject.generate_synthetic_curriculum(
            subject.SyntheticCurriculumConfig(task_count=7), object()
        )


def test_curriculum_source_has_no_numpy_or_jax_randomness() -> None:
    subject = _subject()
    source = inspect.getsource(subject)

    assert "np.random" not in source
    assert "jax.random" not in source
    assert "brainstate.random.RandomState" in source


def test_curriculum_uses_declared_weighted_family_cycle() -> None:
    subject = _subject()

    assert subject.FAMILY_SCHEDULE == (*subject.FAMILIES, "pattern_label")


def test_inference_modules_do_not_import_training_curriculum() -> None:
    model = importlib.import_module("examples.pp_prop.latent_workspace_direct_model")
    generation = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_generation"
    )

    assert "latent_workspace_direct_curriculum" not in inspect.getsource(model)
    assert "latent_workspace_direct_curriculum" not in inspect.getsource(generation)
