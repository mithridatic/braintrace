"""Tests for the training-only direct ARC synthetic curriculum."""

from __future__ import annotations

import importlib
import inspect

import brainstate
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_curriculum"
    )


def test_curriculum_is_seed_deterministic_balanced_and_bounded() -> None:
    subject = _subject()
    config = subject.SyntheticCurriculumConfig(task_count=14, demonstrations=3)

    first = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(17)
    )
    second = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(17)
    )

    assert first.task_sha256 == second.task_sha256
    assert first.family_counts == {family: 2 for family in subject.FAMILIES}
    assert first.family_counts == second.family_counts
    assert len(first.tasks) == 14
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
    config = subject.SyntheticCurriculumConfig(task_count=7, demonstrations=2)

    first = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(3)
    )
    second = subject.generate_synthetic_curriculum(
        config, brainstate.random.RandomState(5)
    )

    assert first.task_sha256 != second.task_sha256


def test_every_curriculum_family_has_its_declared_relation() -> None:
    subject = _subject()
    result = subject.generate_synthetic_curriculum(
        subject.SyntheticCurriculumConfig(task_count=7, demonstrations=3),
        brainstate.random.RandomState(23),
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


def test_inference_modules_do_not_import_training_curriculum() -> None:
    model = importlib.import_module("examples.pp_prop.latent_workspace_direct_model")
    generation = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_generation"
    )

    assert "latent_workspace_direct_curriculum" not in inspect.getsource(model)
    assert "latent_workspace_direct_curriculum" not in inspect.getsource(generation)
