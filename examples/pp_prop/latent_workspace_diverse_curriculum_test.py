"""Tests for the surface-diversified schema-v4 synthetic curriculum."""

from __future__ import annotations

import importlib

import brainstate
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_diverse_curriculum"
    )


def _arrays(pair):
    return np.array(pair.input.cells), np.array(pair.output.cells)


def _pairs(task):
    return [(_arrays(pair)) for pair in (*task.train, *task.test)]


def _by_family(result):
    return {
        task.task_id.split(":")[1]: task
        for task in result.tasks
        if task.task_id is not None
    }


def _generate(seed: int = 41, task_count: int = 48):
    subject = _subject()
    return subject.generate_diverse_curriculum(
        subject.DiverseCurriculumConfig(task_count=task_count),
        brainstate.random.RandomState(seed),
    )


def test_config_validation() -> None:
    subject = _subject()
    with pytest.raises(TypeError):
        subject.DiverseCurriculumConfig(task_count="12")
    with pytest.raises(ValueError):
        subject.DiverseCurriculumConfig(task_count=0)
    with pytest.raises(ValueError):
        subject.DiverseCurriculumConfig(task_count=4, max_grid_size=5)
    with pytest.raises(ValueError):
        subject.DiverseCurriculumConfig(task_count=4, max_grid_size=31)
    with pytest.raises(ValueError):
        subject.DiverseCurriculumConfig(task_count=4, min_demonstrations=1)
    with pytest.raises(ValueError):
        subject.DiverseCurriculumConfig(
            task_count=4, min_demonstrations=8, max_demonstrations=4
        )
    with pytest.raises(TypeError):
        subject.generate_diverse_curriculum(
            {"task_count": 4}, brainstate.random.RandomState(1)
        )
    with pytest.raises(TypeError):
        subject.generate_diverse_curriculum(
            subject.DiverseCurriculumConfig(task_count=4), rng=object()
        )


def test_seed_deterministic_balanced_bounded_and_structured() -> None:
    subject = _subject()
    first = _generate(seed=17)
    second = _generate(seed=17)

    assert first.task_sha256 == second.task_sha256
    assert first.schema_version == "direct_synthetic_curriculum_v4"
    expected_counts = {family: 4 for family in subject.FAMILIES}
    expected_counts["pattern_label"] = 8
    assert first.family_counts == expected_counts == second.family_counts
    assert len(first.tasks) == 48
    for left, right in zip(first.tasks, second.tasks, strict=True):
        assert left == right
        assert 2 <= len(left.train) <= 6
        assert len(left.test) == 1
        family, index = left.task_id.split(":")[1], left.task_id.split(":")[2]
        assert left.task_id == f"synthetic-v4:{family}:{index}"
        for pair in (*left.train, *left.test):
            assert pair.output is not None
            for grid in (pair.input, pair.output):
                assert 1 <= grid.height <= 30
                assert 1 <= grid.width <= 30
                for row in grid.cells:
                    assert all(0 <= int(cell) <= 9 for cell in row)


def test_curriculum_changes_with_seed() -> None:
    assert _generate(seed=3).task_sha256 != _generate(seed=5).task_sha256


def test_surface_diversity_is_real() -> None:
    result = _generate(seed=97, task_count=96)
    sides, densities, color_counts, demo_counts = [], [], [], set()
    for task in result.tasks:
        demo_counts.add(len(task.train))
        for pair in (*task.train, *task.test):
            array, _ = _arrays(pair)
            sides.extend([array.shape[0], array.shape[1]])
            densities.append(float((array != 0).mean()))
            color_counts.append(len(np.unique(array)))
    assert max(sides) > 9, "diversified curriculum must exceed v3's side cap"
    assert min(densities) < 0.10 or max(densities) > 0.35
    assert max(densities) > 0.90
    assert max(color_counts) >= 3
    assert len(demo_counts) > 1


def test_copy_family_semantics() -> None:
    task = _by_family(_generate())["copy"]
    for input_array, output_array in _pairs(task):
        assert np.array_equal(input_array, output_array)


def test_recolor_family_semantics() -> None:
    task = _by_family(_generate())["recolor"]
    mapping = {}
    changed_any = False
    for input_array, output_array in _pairs(task):
        assert input_array.shape == output_array.shape
        changed_any |= bool((input_array != output_array).any())
        for source, target in zip(input_array.ravel(), output_array.ravel()):
            source, target = int(source), int(target)
            if source in mapping:
                assert mapping[source] == target
            else:
                mapping[source] = target
    assert changed_any
    assert any(source != target for source, target in mapping.items())


def _dihedral_variants(array: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(array, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def test_dihedral_family_semantics() -> None:
    task = _by_family(_generate())["dihedral"]
    pairs = _pairs(task)
    first_in, first_out = pairs[0]
    candidates = {
        index
        for index, variant in enumerate(_dihedral_variants(first_in))
        if variant.shape == first_out.shape and np.array_equal(variant, first_out)
    }
    assert candidates
    for input_array, output_array in pairs[1:]:
        matched = {
            index
            for index in candidates
            if index < len(_dihedral_variants(input_array))
        }
        matched = {
            index
            for index in matched
            if _dihedral_variants(input_array)[index].shape == output_array.shape
            and np.array_equal(_dihedral_variants(input_array)[index], output_array)
        }
        candidates &= matched
        assert candidates


def _is_subrectangle(inner: np.ndarray, outer: np.ndarray) -> bool:
    if inner.shape[0] > outer.shape[0] or inner.shape[1] > outer.shape[1]:
        return False
    for row in range(outer.shape[0] - inner.shape[0] + 1):
        for column in range(outer.shape[1] - inner.shape[1] + 1):
            if np.array_equal(
                outer[row : row + inner.shape[0], column : column + inner.shape[1]],
                inner,
            ):
                return True
    return False


def test_crop_family_semantics() -> None:
    task = _by_family(_generate())["crop"]
    for input_array, output_array in _pairs(task):
        assert _is_subrectangle(output_array, input_array)
        assert (
            output_array.shape[0] < input_array.shape[0]
            or output_array.shape[1] < input_array.shape[1]
        )


def test_upscale_family_semantics() -> None:
    task = _by_family(_generate())["upscale"]
    for input_array, output_array in _pairs(task):
        factor_height, remainder_h = divmod(
            output_array.shape[0], input_array.shape[0]
        )
        factor_width, remainder_w = divmod(
            output_array.shape[1], input_array.shape[1]
        )
        assert remainder_h == remainder_w == 0
        assert factor_height == factor_width >= 2
        assert np.array_equal(
            np.kron(input_array, np.ones((factor_height, factor_width), int)),
            output_array,
        )


def test_count_family_semantics() -> None:
    task = _by_family(_generate())["count"]
    for input_array, output_array in _pairs(task):
        assert output_array.shape == (1, 1)
        assert int(output_array[0, 0]) == int((input_array != 0).sum())
        assert 1 <= int(output_array[0, 0]) <= 9


def test_pattern_label_family_semantics() -> None:
    task = _by_family(_generate())["pattern_label"]
    pattern_labels = {}
    for input_array, output_array in _pairs(task):
        assert input_array.shape == (3, 3)
        assert output_array.shape == (1, 1)
        key = tuple(int(cell) for cell in (input_array != 0).ravel())
        label = int(output_array[0, 0])
        assert 1 <= label <= 9
        if key in pattern_labels:
            assert pattern_labels[key] == label
        else:
            pattern_labels[key] = label
    assert 2 <= len(pattern_labels) <= 4


def test_select_marked_region_family_semantics() -> None:
    task = _by_family(_generate())["select_marked_region"]
    for input_array, output_array in _pairs(task):
        assert output_array.shape == (1, 1)
        height, width = input_array.shape
        assert height % 2 == 0 and width % 2 == 0
        bases, marker_counts = [], []
        for row_start in (0, height // 2):
            for column_start in (0, width // 2):
                region = input_array[
                    row_start : row_start + height // 2,
                    column_start : column_start + width // 2,
                ]
                values, counts = np.unique(region, return_counts=True)
                base = int(values[int(np.argmax(counts))])
                bases.append(base)
                marker_counts.append(int(counts.sum() - counts.max()))
        assert len(set(marker_counts)) == 4, "marker counts must be unambiguous"
        winner = bases[int(np.argmax(marker_counts))]
        assert int(output_array[0, 0]) == winner


def test_project_marker_family_semantics() -> None:
    task = _by_family(_generate())["project_marker"]
    for input_array, output_array in _pairs(task):
        assert input_array.shape == output_array.shape
        changed = input_array != output_array
        assert int(changed.sum()) == 2
        removed = [int(v) for v in input_array[changed]]
        added = [int(v) for v in output_array[changed]]
        # One cell loses the marker to background; one gains the marker.
        assert added.count(0) == 1
        moved = [v for v in added if v != 0]
        assert len(moved) == 1 and moved[0] in removed


def test_complete_corner_family_semantics() -> None:
    task = _by_family(_generate())["complete_corner"]
    completion_colors = set()
    for input_array, output_array in _pairs(task):
        assert input_array.shape == output_array.shape
        changed = input_array != output_array
        assert int(changed.sum()) >= 1
        assert (input_array[changed] == 0).all()
        completion_colors |= {int(v) for v in output_array[changed]}
    assert len(completion_colors) == 1


def test_mirror_concat_family_semantics() -> None:
    task = _by_family(_generate())["mirror_concat"]
    pairs = _pairs(task)
    first_in, first_out = pairs[0]

    def _matches(direction: int, array: np.ndarray, output: np.ndarray) -> bool:
        if direction == 0:
            expected = np.concatenate((np.flipud(array), array), axis=0)
        elif direction == 1:
            expected = np.concatenate((array, np.flipud(array)), axis=0)
        elif direction == 2:
            expected = np.concatenate((np.fliplr(array), array), axis=1)
        else:
            expected = np.concatenate((array, np.fliplr(array)), axis=1)
        return expected.shape == output.shape and np.array_equal(expected, output)

    candidates = {
        direction
        for direction in range(4)
        if _matches(direction, first_in, first_out)
    }
    assert candidates
    for input_array, output_array in pairs[1:]:
        candidates = {
            direction
            for direction in candidates
            if _matches(direction, input_array, output_array)
        }
        assert candidates
