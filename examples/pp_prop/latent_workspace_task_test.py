"""Tests for standard ARC data, provenance, and row-event encoding."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import brainstate
import numpy as np
import pytest

from examples.pp_prop import latent_workspace_task as task_module
from examples.pp_prop.latent_workspace_task import (
    AssociativeMemoryFeatureIndices,
    ArcGrid,
    ArcPair,
    ArcQueryEpisode,
    ArcTask,
    AugmentationConfig,
    DatasetSource,
    DecodedQueryContext,
    EncodedQueryEpisode,
    ExcludedTask,
    GridTarget,
    LoadedDataset,
    RowEventConfig,
    SourceManifest,
    SplitLeakageError,
    arc_task_from_mapping,
    arc_task_to_mapping,
    assert_no_evaluation_leakage,
    associative_memory_feature_indices,
    augment_training_task,
    canonical_task_fingerprint,
    decode_row_events,
    detect_split_leakage,
    draw_training_augmentation,
    encode_arc_query_episode,
    encode_query_episode,
    encode_target_grid,
    encode_task_queries,
    load_arc_task,
    load_dataset_source,
    leave_one_demonstration_out_episodes,
    query_episodes,
    smoke_arc_tasks,
    smoke_loaded_dataset,
)


def _task_payload(*, test_output: list[list[int]] | None = None) -> dict:
    test: dict[str, object] = {"input": [[0, 1], [2, 9]]}
    if test_output is not None:
        test["output"] = test_output
    return {
        "train": [
            {"input": [[1, 2, 3], [4, 5, 6]], "output": [[1, 4], [2, 5], [3, 6]]},
            {"input": [[7, 8]], "output": [[7], [8]]},
        ],
        "test": [test],
    }


def _task(*, task_id: str = "sample", target: bool = True) -> ArcTask:
    payload = _task_payload(test_output=[[0, 2], [1, 9]] if target else None)
    return arc_task_from_mapping(payload, task_id=task_id)


def _source(
    path: Path,
    *,
    name: str = "ARC-AGI-1 training",
    role: str = "train",
    source_format: str = "auto",
) -> DatasetSource:
    return DatasetSource(
        name=name,
        role=role,
        version="unit-test",
        path=str(path),
        license_reference="https://example.test/dataset",
        format=source_format,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(task: ArcTask, *, name: str, role: str) -> SourceManifest:
    fingerprint = canonical_task_fingerprint(task)
    source = DatasetSource(
        name=name,
        role=role,
        version="test",
        path="unused",
        license_reference="test reference",
    )
    return SourceManifest(
        source=source,
        resolved_path="unused",
        files=(),
        parsed_task_count=1,
        valid_task_count=1,
        rejected=(),
        duplicate_fingerprints=(),
        task_fingerprints=(fingerprint,),
    )


def test_arc_grid_is_immutable_rectangular_and_copying() -> None:
    source = [[0, 9], [1, 2]]
    grid = ArcGrid(source)
    source[0][0] = 8

    assert grid.cells == ((0, 9), (1, 2))
    assert (grid.height, grid.width) == (2, 2)
    array = grid.as_array()
    array[0, 0] = 7
    assert grid.cells[0][0] == 0
    assert grid.to_list() == [[0, 9], [1, 2]]
    with pytest.raises(Exception):
        grid.cells = ((1,),)


def test_arc_grid_accepts_boundary_shapes_and_numpy_integers() -> None:
    tiny = ArcGrid(((np.int64(9),),))
    large = ArcGrid(np.zeros((30, 30), dtype=np.int32))

    assert tiny.cells == ((9,),)
    assert (large.height, large.width) == (30, 30)


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ([], "at least one row"),
        ([[]], "non-empty row"),
        ([[1], [1, 2]], "ragged"),
        (np.zeros((1, 1, 1), dtype=np.int32), "two-dimensional"),
        (np.zeros((31, 1), dtype=np.int32), "height"),
        (np.zeros((1, 31), dtype=np.int32), "width"),
        ([[True]], "integer ARC color"),
        ([[1.0]], "integer ARC color"),
        ([[10]], "outside 0..9"),
        ([[-1]], "outside 0..9"),
    ],
)
def test_invalid_arc_grid_fails_closed(grid: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ArcGrid(grid)


def test_arc_task_parses_unequal_dimensions_and_optional_test_output() -> None:
    task = _task(target=False)

    assert len(task.train) == 2
    assert task.train[0].input.width == 3
    assert task.train[0].output.height == 3
    assert task.test[0].output is None
    assert arc_task_to_mapping(task) == _task_payload(test_output=None)


def test_arc_pair_coerces_grids_but_task_requires_pair_instances() -> None:
    pair = ArcPair([[1]], [[2]])

    assert isinstance(pair.input, ArcGrid)
    assert isinstance(pair.output, ArcGrid)
    with pytest.raises(ValueError, match="ArcPair"):
        ArcTask(train=(pair,), test=({"input": [[1]]},))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"train": [], "test": [{"input": [[1]]}]}, "train"),
        ({"train": [{"input": [[1]], "output": [[1]]}], "test": []}, "test"),
        ({"train": [1], "test": [{"input": [[1]]}]}, r"train\[0\]"),
        ({"train": [{"output": [[1]]}], "test": [{"input": [[1]]}]}, "input"),
        ({"train": [{"input": [[1]]}], "test": [{"input": [[1]]}]}, "output"),
        (
            {"train": [{"input": [[1]], "output": None}], "test": [{"input": [[1]]}]},
            "output",
        ),
        ({"train": [{"input": [[1]], "output": [[1]]}], "test": [{}]}, "input"),
    ],
)
def test_invalid_task_shape_names_the_offending_field(
    payload: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        arc_task_from_mapping(payload, task_id="bad")


def test_scored_loading_requires_every_test_target() -> None:
    with pytest.raises(ValueError, match=r"test\[0\]\.output"):
        arc_task_from_mapping(
            _task_payload(test_output=None),
            task_id="unscored",
            require_test_outputs=True,
        )


def test_direct_per_task_json_loader_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "a1b2c3.json"
    _write_json(path, _task_payload(test_output=[[0, 2], [1, 9]]))

    task = load_arc_task(path, require_test_outputs=True)

    assert task.task_id == "a1b2c3"
    assert task.test[0].output == ArcGrid(((0, 2), (1, 9)))


def test_direct_loader_rejects_json_collection_and_bad_json(tmp_path: Path) -> None:
    collection = tmp_path / "collection.json"
    _write_json(collection, [_task_payload()])
    with pytest.raises(ValueError, match="one JSON object"):
        load_arc_task(collection)

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        load_arc_task(broken)


def test_fingerprint_ignores_identifier_but_covers_content() -> None:
    first = _task(task_id="first")
    renamed = replace(first, task_id="renamed")
    changed = _task(task_id="first")
    changed = replace(
        changed,
        test=(ArcPair(changed.test[0].input, ArcGrid(((0, 2), (1, 8)))),),
    )

    assert canonical_task_fingerprint(first) == canonical_task_fingerprint(renamed)
    assert canonical_task_fingerprint(first) != canonical_task_fingerprint(changed)
    assert canonical_task_fingerprint(
        first, include_test_outputs=False
    ) == canonical_task_fingerprint(changed, include_test_outputs=False)


def test_multiple_query_episodes_retain_shared_task_identity() -> None:
    base = _task()
    task = replace(
        base,
        test=base.test + (ArcPair(ArcGrid(((9,),)), ArcGrid(((9,),))),),
    )

    episodes = query_episodes(task, task_index=12)

    assert [episode.query_index for episode in episodes] == [0, 1]
    assert all(episode.task_index == 12 for episode in episodes)
    assert episodes[0].demonstrations is episodes[1].demonstrations
    assert episodes[0].task_fingerprint == episodes[1].task_fingerprint
    assert episodes[1].query_input == ArcGrid(((9,),))


def test_loo_episodes_preserve_parent_identity_and_order() -> None:
    base = _task()
    third = ArcPair(ArcGrid(((3, 0),)), ArcGrid(((0,), (3,))))
    task = replace(base, train=base.train + (third,))

    episodes = leave_one_demonstration_out_episodes(task, task_index=7)

    assert [episode.query_index for episode in episodes] == [0, 1, 2]
    assert all(episode.task_index == 7 for episode in episodes)
    assert all(episode.task_id == task.task_id for episode in episodes)
    assert all(
        episode.task_fingerprint == canonical_task_fingerprint(task)
        for episode in episodes
    )
    assert [episode.query_input for episode in episodes] == [
        pair.input for pair in task.train
    ]
    assert [episode.target for episode in episodes] == [
        pair.output for pair in task.train
    ]
    assert episodes[0].demonstrations == task.train[1:]
    assert episodes[1].demonstrations == (task.train[0], task.train[2])
    assert episodes[2].demonstrations == task.train[:2]


def test_leave_one_demonstration_out_target_is_not_in_model_context() -> None:
    task = _task()

    episodes = leave_one_demonstration_out_episodes(task)

    for held_out_index, episode in enumerate(episodes):
        held_out = task.train[held_out_index]
        assert episode.query_input is held_out.input
        assert episode.target is held_out.output
        assert held_out not in episode.demonstrations
        assert len(episode.demonstrations) == len(task.train) - 1


def test_leave_one_demonstration_out_requires_context_demonstration() -> None:
    base = _task()
    task = replace(base, train=base.train[:1])

    with pytest.raises(ValueError, match="at least two demonstrations"):
        leave_one_demonstration_out_episodes(task)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_query_episode_task_index_must_be_non_negative_integer(index: object) -> None:
    with pytest.raises(ValueError, match="task_index"):
        query_episodes(_task(), task_index=index)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_leave_one_demonstration_out_task_index_must_be_non_negative_integer(
    index: object,
) -> None:
    with pytest.raises(ValueError, match="task_index"):
        leave_one_demonstration_out_episodes(_task(), task_index=index)


def test_task_json_directory_manifest_hashes_deduplicates_and_rejects(
    tmp_path: Path,
) -> None:
    payload = _task_payload(test_output=[[0, 2], [1, 9]])
    _write_json(tmp_path / "one.json", payload)
    _write_json(tmp_path / "renamed-copy.json", payload)
    _write_json(tmp_path / "invalid.json", {"train": [], "test": []})

    loaded = load_dataset_source(_source(tmp_path, source_format="task_json"))

    assert len(loaded.tasks) == 1
    manifest = loaded.manifest
    assert manifest.parsed_task_count == 2
    assert manifest.valid_task_count == 1
    assert manifest.duplicate_task_count == 1
    assert manifest.rejected_task_count == 1
    assert len(manifest.files) == 3
    assert all(
        len(file.sha256) == 64 and file.size_bytes > 0 for file in manifest.files
    )
    evidence = manifest.to_dict()
    assert evidence["source"]["role"] == "train"
    assert evidence["private_paper_data_available"] is False
    assert evidence["private_training_recipe_available"] is False


def test_collection_json_supports_task_mapping_list_and_wrapper(tmp_path: Path) -> None:
    first = _task_payload(test_output=[[0, 2], [1, 9]])
    second = _task_payload(test_output=[[9, 2], [1, 9]])
    _write_json(tmp_path / "mapping.json", {"alpha": first})
    _write_json(tmp_path / "list.json", [{"id": "beta", "task": second}])

    loaded = load_dataset_source(_source(tmp_path, source_format="collection_json"))

    assert [task.task_id for task in loaded.tasks] == ["beta", "alpha"]
    assert loaded.manifest.valid_task_count == 2


def test_collection_tasks_envelope_and_auto_format(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    _write_json(
        path,
        {"tasks": {"named": _task_payload(test_output=[[0, 2], [1, 9]])}},
    )

    loaded = load_dataset_source(_source(path, source_format="auto"))

    assert loaded.tasks[0].task_id == "named"
    assert loaded.manifest.files[0].path == "tasks.json"


def test_declared_fingerprint_exclusion_is_matched_recorded_and_removed(
    tmp_path: Path,
) -> None:
    excluded_payload = _task_payload(test_output=[[0, 2], [1, 9]])
    retained_payload = _task_payload(test_output=[[9, 2], [1, 9]])
    path = tmp_path / "tasks.json"
    _write_json(
        path, {"excluded-id": excluded_payload, "retained-id": retained_payload}
    )
    fingerprint = canonical_task_fingerprint(
        arc_task_from_mapping(excluded_payload, task_id="different-name")
    )
    source = replace(
        _source(path, source_format="collection_json"),
        exclude_fingerprints=(fingerprint.upper(),),
    )

    loaded = load_dataset_source(source)

    assert [task.task_id for task in loaded.tasks] == ["retained-id"]
    assert loaded.manifest.parsed_task_count == 2
    assert loaded.manifest.valid_task_count == 1
    assert loaded.manifest.excluded_task_count == 1
    assert loaded.manifest.exclusions == (
        ExcludedTask(
            origin="tasks.json:excluded-id",
            task_id="excluded-id",
            fingerprint=fingerprint,
        ),
    )
    evidence = loaded.manifest.to_dict()
    assert evidence["source"]["exclude_fingerprints"] == [fingerprint]
    assert evidence["excluded_task_count"] == 1
    assert evidence["exclusions"][0]["task_id"] == "excluded-id"


def test_unmatched_fingerprint_exclusion_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    _write_json(path, _task_payload(test_output=[[0, 2], [1, 9]]))
    source = replace(
        _source(path, source_format="task_json"),
        exclude_fingerprints=("0" * 64,),
    )

    with pytest.raises(ValueError, match="did not match.*0{64}"):
        load_dataset_source(source)


def test_jsonl_supports_wrappers_blank_lines_and_rejection_accounting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    valid = {
        "task_id": "line-task",
        "task": _task_payload(test_output=[[0, 2], [1, 9]]),
    }
    invalid = {"id": "bad", "task": {"train": [], "test": []}}
    path.write_text(
        "\n" + json.dumps(valid) + "\n{\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )

    loaded = load_dataset_source(_source(path, source_format="jsonl"))

    assert loaded.tasks[0].task_id == "line-task"
    assert loaded.manifest.rejected_task_count == 2
    assert all(rejected.reason for rejected in loaded.manifest.rejected)
    assert any("train" in rejected.reason for rejected in loaded.manifest.rejected)


def test_source_with_no_valid_tasks_fails_with_rejection_summary(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "bad.json", {"train": [], "test": []})

    with pytest.raises(ValueError, match="no valid unique tasks.*train"):
        load_dataset_source(_source(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("version", ""),
        ("license_reference", ""),
        ("path", ""),
        ("role", "other"),
        ("format", "csv"),
    ],
)
def test_source_declaration_fails_closed(field: str, value: object) -> None:
    arguments = {
        "name": "source",
        "role": "train",
        "version": "1",
        "path": "unused",
        "license_reference": "reference",
        "format": "auto",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        DatasetSource(**arguments)


@pytest.mark.parametrize(
    "fingerprints",
    [
        ("short",),
        ("z" * 64,),
        ("a" * 64, "A" * 64),
    ],
)
def test_source_exclusion_fingerprints_are_valid_unique_sha256(
    fingerprints: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="exclude_fingerprints"):
        replace(
            _source(Path("unused")),
            exclude_fingerprints=fingerprints,
        )


def test_evaluation_source_cannot_exclude_fingerprints() -> None:
    with pytest.raises(ValueError, match="train/tuning"):
        DatasetSource(
            "ARC-AGI-1 evaluation",
            "evaluation",
            "1",
            "unused",
            "reference",
            exclude_fingerprints=("a" * 64,),
        )


def test_eval_only_and_private_paper_sources_cannot_be_mislabelled() -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        DatasetSource("ARC-AGI-1 evaluation", "train", "1", "x", "ref")
    with pytest.raises(ValueError, match="private data"):
        DatasetSource("BDH-CQ private paper data", "train", "1", "x", "ref")


def test_missing_or_empty_source_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_dataset_source(_source(tmp_path / "missing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no supported files"):
        load_dataset_source(_source(empty))


def test_evaluation_source_requires_test_outputs_by_default(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    _write_json(path, _task_payload(test_output=None))
    source = _source(
        path,
        name="ARC-AGI-1 evaluation",
        role="evaluation",
        source_format="task_json",
    )

    with pytest.raises(ValueError, match="no valid unique tasks.*output"):
        load_dataset_source(source)
    loaded = load_dataset_source(source, require_test_outputs=False)
    assert loaded.tasks[0].test[0].output is None


def test_split_leakage_detects_train_and_tuning_against_evaluation() -> None:
    task = _task()
    train = _manifest(task, name="train-a", role="train")
    tuning = _manifest(task, name="tune-a", role="tuning")
    evaluation = _manifest(task, name="eval-a", role="evaluation")
    fixture = _manifest(task, name="fixture-a", role="fixture")

    overlaps = detect_split_leakage((fixture, evaluation, tuning, train))

    assert len(overlaps) == 1
    assert overlaps[0].fingerprint == canonical_task_fingerprint(task)
    assert overlaps[0].fitting_sources == ("train-a (train)", "tune-a (tuning)")
    assert overlaps[0].evaluation_sources == ("eval-a (evaluation)",)
    with pytest.raises(SplitLeakageError, match=r"train-a.*eval-a") as caught:
        assert_no_evaluation_leakage((train, evaluation))
    assert caught.value.overlaps == (
        overlaps[0].__class__(
            fingerprint=overlaps[0].fingerprint,
            fitting_sources=("train-a (train)",),
            evaluation_sources=("eval-a (evaluation)",),
        ),
    )


def test_nonoverlapping_or_fixture_only_content_passes_leakage_gate() -> None:
    first = _task()
    second = replace(
        first, test=(ArcPair(first.test[0].input, ArcGrid(((9, 9), (9, 9)))),)
    )
    assert_no_evaluation_leakage(
        (
            _manifest(first, name="train", role="train"),
            _manifest(second, name="eval", role="evaluation"),
            _manifest(first, name="fixture", role="fixture"),
        )
    )


@pytest.mark.parametrize(
    "claim",
    ["private_paper_data_available", "private_training_recipe_available"],
)
def test_manifest_cannot_claim_unavailable_private_paper_assets(claim: str) -> None:
    task = _task()
    manifest = _manifest(task, name="public", role="train")

    with pytest.raises(ValueError, match="private data.*unavailable"):
        replace(manifest, **{claim: True})


def test_training_augmentation_is_deterministic_relation_consistent_and_immutable() -> (
    None
):
    original = smoke_arc_tasks()[0]
    original_mapping = arc_task_to_mapping(original)

    first = draw_training_augmentation(
        original, brainstate.random.RandomState(731), role="train"
    )
    second = draw_training_augmentation(
        original, brainstate.random.RandomState(731), role="train"
    )

    assert first == second
    assert first.color_map[0] == 0
    assert sorted(first.color_map) == list(range(10))
    assert 0 <= first.dihedral_index < 8
    assert sorted(first.demonstration_order) == list(range(len(original.train)))
    assert arc_task_to_mapping(original) == original_mapping

    source_pair = original.train[first.demonstration_order[0]]
    augmented_pair = first.task.train[0]
    expected_input = task_module._transform_grid(
        source_pair.input,
        np.asarray(first.color_map, dtype=np.int32),
        first.dihedral_index,
    )
    expected_output = task_module._transform_grid(
        source_pair.output,
        np.asarray(first.color_map, dtype=np.int32),
        first.dihedral_index,
    )
    assert augmented_pair == ArcPair(expected_input, expected_output)


def test_identity_augmentation_and_convenience_api() -> None:
    original = _task()
    config = AugmentationConfig(False, False, False)

    result = draw_training_augmentation(
        original, brainstate.random.RandomState(22), config=config
    )
    convenience = augment_training_task(
        original, brainstate.random.RandomState(22), config=config
    )

    assert result.task == original
    assert result.color_map == tuple(range(10))
    assert result.dihedral_index == 0
    assert result.demonstration_order == (0, 1)
    assert convenience == original
    assert convenience is not original


@pytest.mark.parametrize("role", ["evaluation", "fixture", "tuning"])
def test_augmentation_rejects_every_nontraining_role_without_consuming_rng(
    role: str,
) -> None:
    rejected = brainstate.random.RandomState(91)
    control = brainstate.random.RandomState(91)
    with pytest.raises(ValueError, match="training-only"):
        augment_training_task(_task(), rejected, role=role)

    rejected_next = np.asarray(rejected.randint(1000, size=(5,)))
    control_next = np.asarray(control.randint(1000, size=(5,)))
    assert np.array_equal(rejected_next, control_next)


def test_augmentation_does_not_consume_global_brainstate_stream() -> None:
    with brainstate.random.seed_context(2026):
        expected_before = np.asarray(brainstate.random.uniform(size=(4,)))
        expected_after = np.asarray(brainstate.random.uniform(size=(4,)))
    with brainstate.random.seed_context(2026):
        actual_before = np.asarray(brainstate.random.uniform(size=(4,)))
        augment_training_task(_task(), brainstate.random.RandomState(11))
        actual_after = np.asarray(brainstate.random.uniform(size=(4,)))

    assert np.array_equal(actual_before, expected_before)
    assert np.array_equal(actual_after, expected_after)


@pytest.mark.parametrize("dihedral_index", range(8))
def test_all_dihedral_transforms_preserve_cells_and_expected_orientation(
    dihedral_index: int,
) -> None:
    grid = ArcGrid(((1, 2, 3), (4, 5, 6)))
    transformed = task_module._transform_grid(
        grid, np.arange(10, dtype=np.int32), dihedral_index
    )

    assert sorted(transformed.as_array().ravel()) == [1, 2, 3, 4, 5, 6]
    if dihedral_index % 2:
        assert (transformed.height, transformed.width) == (3, 2)
    else:
        assert (transformed.height, transformed.width) == (2, 3)


def test_default_row_event_layout_is_bounded_and_nonoverlapping() -> None:
    config = RowEventConfig()
    slices = (
        config.valid_slice,
        config.phase_slice,
        config.demonstration_slice,
        config.side_valid_slice,
        config.normalized_slice,
        config.row_index_slice,
        config.input_height_slice,
        config.input_width_slice,
        config.output_height_slice,
        config.output_width_slice,
        config.input_mask_slice,
        config.output_mask_slice,
        config.input_color_slice,
        config.output_color_slice,
    )

    assert config.max_events == 330
    assert config.input_width == 830
    assert slices[0].start == 0
    assert all(left.stop == right.start for left, right in zip(slices, slices[1:]))
    assert slices[-1].stop == config.input_width


def test_standard_arc_associative_memory_feature_indices_are_exact() -> None:
    indices = associative_memory_feature_indices(RowEventConfig())

    expected_key = (
        (13, 15, 16, 17)
        + tuple(range(20, 50))
        + tuple(range(50, 80))
        + tuple(range(80, 110))
        + tuple(range(170, 200))
        + tuple(range(230, 530))
    )
    expected_value = (
        (14, 15, 18, 19)
        + tuple(range(20, 50))
        + tuple(range(110, 140))
        + tuple(range(140, 170))
        + tuple(range(200, 230))
        + tuple(range(530, 830))
    )
    assert isinstance(indices, AssociativeMemoryFeatureIndices)
    assert indices.key_indices == expected_key
    assert indices.value_indices == expected_value
    assert len(indices.key_indices) == len(indices.value_indices) == 424


def test_small_binding_associative_memory_feature_indices_are_exact() -> None:
    config = RowEventConfig(max_demonstrations=4, max_grid_size=1)

    indices = associative_memory_feature_indices(config)

    assert indices.key_indices == (
        7,
        9,
        10,
        11,
        14,
        15,
        16,
        19,
        *range(21, 31),
    )
    assert indices.value_indices == (
        8,
        9,
        12,
        13,
        14,
        17,
        18,
        20,
        *range(31, 41),
    )
    assert len(indices.key_indices) == len(indices.value_indices) == 18


def test_associative_memory_indices_preserve_side_semantics_without_identity() -> None:
    config = RowEventConfig()
    indices = associative_memory_feature_indices(config)
    row_indices = range(*config.row_index_slice.indices(config.input_width))
    shared = {config.normalized_slice.start, *row_indices}
    excluded = {
        *range(*config.valid_slice.indices(config.input_width)),
        *range(*config.phase_slice.indices(config.input_width)),
        *range(*config.demonstration_slice.indices(config.input_width)),
    }

    assert set(indices.key_indices) & set(indices.value_indices) == shared
    assert not excluded & set(indices.key_indices)
    assert not excluded & set(indices.value_indices)

    demonstration = encode_query_episode(_task(), 0, config).events[0]
    key_indices = np.asarray(indices.key_indices)
    value_indices = np.asarray(indices.value_indices)
    assert demonstration[key_indices].sum() > 0.0
    assert demonstration[value_indices].sum() > 0.0

    query = encode_query_episode(_task(), 0, config).events[
        config.max_demonstrations * config.max_grid_size
    ]
    assert query[key_indices].sum() > 0.0
    assert query[value_indices].sum() == pytest.approx(
        query[config.normalized_slice.start] + query[config.row_index_slice].sum()
    )


@pytest.mark.parametrize(
    ("key_indices", "value_indices", "message"),
    [
        ((), (), "non-empty"),
        ((0, 0), (1, 2), "unique"),
        ((0,), (1, 2), "same width"),
        ((-1,), (1,), "non-negative"),
        ((True,), (1,), "integers"),
    ],
)
def test_associative_memory_feature_index_record_fails_closed(
    key_indices: tuple[int, ...],
    value_indices: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AssociativeMemoryFeatureIndices(key_indices, value_indices)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_demonstrations": 0}, "max_demonstrations"),
        ({"max_demonstrations": True}, "max_demonstrations"),
        ({"max_grid_size": 31}, "max_grid_size"),
        ({"max_grid_size": 0}, "max_grid_size"),
        ({"color_count": 9}, "color_count"),
    ],
)
def test_invalid_row_event_configuration_fails_closed(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RowEventConfig(**kwargs)


def test_row_event_round_trip_is_lossless_for_unequal_shapes() -> None:
    task = smoke_arc_tasks()[0]

    encoded = encode_query_episode(task, 0)
    decoded = decode_row_events(encoded.events)

    assert isinstance(encoded, EncodedQueryEpisode)
    assert isinstance(decoded, DecodedQueryContext)
    assert decoded.demonstrations == task.train
    assert decoded.query_input == task.test[0].input
    assert encoded.target == task.test[0].output
    assert (
        encoded.valid_event_count
        == sum(max(pair.input.height, pair.output.height) for pair in task.train)
        + task.test[0].input.height
    )
    assert encoded.query_stop - encoded.query_start == task.test[0].input.height
    assert encoded.query_start == 10 * 30
    assert encoded.demonstration_spans == ((0, 30), (30, 60))


def test_multi_query_encoding_preserves_indices_and_shared_demonstrations() -> None:
    task = smoke_arc_tasks()[0]

    encoded = encode_task_queries(task, task_index=7)

    assert [episode.query_index for episode in encoded] == [0, 1]
    assert all(episode.task_index == 7 for episode in encoded)
    for episode, query in zip(encoded, task.test, strict=True):
        decoded = decode_row_events(episode.events)
        assert decoded.demonstrations == task.train
        assert decoded.query_input == query.input


def test_held_out_target_never_changes_model_input_bytes() -> None:
    first = _task()
    second = replace(
        first,
        test=(ArcPair(first.test[0].input, ArcGrid(((9, 8), (7, 6)))),),
    )

    first_encoded = encode_query_episode(first, 0)
    second_encoded = encode_query_episode(second, 0)

    assert first_encoded.events.tobytes() == second_encoded.events.tobytes()
    assert first_encoded.target != second_encoded.target
    assert first_encoded.task_fingerprint != second_encoded.task_fingerprint
    config = RowEventConfig()
    query_rows = first_encoded.events[
        first_encoded.query_start : first_encoded.query_stop
    ]
    assert not np.any(query_rows[:, config.output_mask_slice])
    assert not np.any(query_rows[:, config.output_color_slice])
    assert not np.any(query_rows[:, config.output_height_slice])
    assert not np.any(query_rows[:, config.output_width_slice])


def test_arc_query_episode_encoding_round_trips_context_and_parent_metadata() -> None:
    episode = leave_one_demonstration_out_episodes(_task(), task_index=7)[1]

    encoded = encode_arc_query_episode(episode)
    decoded = decode_row_events(encoded.events)

    assert decoded.demonstrations == episode.demonstrations
    assert decoded.query_input == episode.query_input
    assert encoded.task_index == episode.task_index
    assert encoded.query_index == episode.query_index
    assert encoded.task_id == episode.task_id
    assert encoded.task_fingerprint == episode.task_fingerprint
    assert encoded.target == episode.target


def test_arc_query_episode_target_never_changes_model_input_bytes() -> None:
    episode = leave_one_demonstration_out_episodes(_task())[0]
    changed = replace(episode, target=ArcGrid(((9, 8), (7, 6))))

    encoded = encode_arc_query_episode(episode)
    changed_encoded = encode_arc_query_episode(changed)

    assert encoded.events.tobytes() == changed_encoded.events.tobytes()
    assert encoded.target != changed_encoded.target


def test_arc_query_episode_encoding_does_not_require_official_query_label() -> None:
    episode = query_episodes(_task(target=False), task_index=3)[0]

    encoded = encode_arc_query_episode(episode)
    decoded = decode_row_events(encoded.events)

    assert decoded.demonstrations == episode.demonstrations
    assert decoded.query_input == episode.query_input
    assert encoded.target is None
    assert encoded.task_index == 3


def test_fixed_block_padding_is_all_zero_inert_and_read_only() -> None:
    encoded = encode_query_episode(_task(), 0)
    config = RowEventConfig()
    valid = encoded.events[:, config.valid_slice.start] == 1.0

    assert encoded.query_start == config.max_demonstrations * config.max_grid_size
    assert encoded.query_stop > encoded.valid_event_count
    assert np.any(~valid[: encoded.query_start])
    assert not np.any(encoded.events[~valid])
    assert not np.any(encoded.events[encoded.query_stop :])
    assert encoded.events.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        encoded.events[0, 0] = 0.0


def test_full_capacity_30x30_and_maximum_demo_count_round_trips() -> None:
    grid = ArcGrid(np.arange(900, dtype=np.int32).reshape(30, 30) % 10)
    pair = ArcPair(grid, grid)
    task = ArcTask(train=(pair,) * 10, test=(pair,))

    encoded = encode_query_episode(task, 0)

    assert encoded.valid_event_count == 330
    assert encoded.query_stop == encoded.events.shape[0]
    assert decode_row_events(encoded.events).demonstrations == task.train


def test_output_derangement_cannot_move_fixed_demo_or_query_blocks() -> None:
    first = ArcPair(ArcGrid(((1,), (2,), (3,))), ArcGrid(((4,),)))
    second = ArcPair(ArcGrid(((5,),)), ArcGrid(((6,), (7,), (8,))))
    query = ArcPair(ArcGrid(((9,),)), ArcGrid(((9,),)))
    intact = ArcTask(train=(first, second), test=(query,))
    deranged = ArcTask(
        train=(
            ArcPair(first.input, second.output),
            ArcPair(second.input, first.output),
        ),
        test=(query,),
    )

    intact_encoded = encode_query_episode(intact, 0)
    deranged_encoded = encode_query_episode(deranged, 0)

    assert intact_encoded.demonstration_spans == deranged_encoded.demonstration_spans
    assert intact_encoded.query_start == deranged_encoded.query_start == 300
    assert intact_encoded.query_stop == deranged_encoded.query_stop == 301
    assert (
        intact_encoded.events[300:].tobytes() == deranged_encoded.events[300:].tobytes()
    )


def test_encoding_rejects_demo_grid_and_query_overflow() -> None:
    pair = ArcPair(ArcGrid(((1,),)), ArcGrid(((1,),)))
    too_many = ArcTask(train=(pair,) * 11, test=(pair,))
    with pytest.raises(ValueError, match="demonstrations.*capacity"):
        encode_query_episode(too_many, 0)

    task = _task()
    with pytest.raises(ValueError, match="grid shapes.*capacity"):
        encode_query_episode(task, 0, RowEventConfig(max_grid_size=2))
    with pytest.raises(ValueError, match="query_index"):
        encode_query_episode(task, 1)
    with pytest.raises(ValueError, match="query_index"):
        encode_query_episode(task, True)
    with pytest.raises(ValueError, match="task_index"):
        encode_query_episode(task, 0, task_index=True)
    with pytest.raises(ValueError, match="task_index"):
        encode_query_episode(task, 0, task_index=-1)


@pytest.mark.parametrize("corruption", ["gap", "padding", "phase", "query-output"])
def test_decoder_fails_closed_on_corrupt_features(corruption: str) -> None:
    config = RowEventConfig()
    encoded = encode_query_episode(_task(), 0, config)
    events = encoded.events.copy()
    if corruption == "gap":
        events[1, config.valid_slice] = 0.0
    elif corruption == "padding":
        events[config.max_grid_size - 1, config.phase_slice.start] = 1.0
    elif corruption == "phase":
        events[0, config.phase_slice] = 0.0
    else:
        events[encoded.query_start, config.output_color_slice.start] = 1.0

    with pytest.raises(ValueError):
        decode_row_events(events, config)


def test_decoder_rejects_wrong_shape_and_nonfinite_values() -> None:
    config = RowEventConfig()
    with pytest.raises(ValueError, match="events shape"):
        decode_row_events(np.zeros((1, 1), dtype=np.float32), config)
    encoded = encode_query_episode(_task(), 0, config)
    events = encoded.events.copy()
    events[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        decode_row_events(events, config)


def test_target_padding_records_exact_shape_mask_and_is_read_only() -> None:
    target = encode_target_grid(ArcGrid(((9, 1, 2), (3, 4, 5))))

    assert isinstance(target, GridTarget)
    assert target.colors.shape == target.valid_mask.shape == (30, 30)
    assert (target.height_index, target.width_index) == (1, 2)
    assert target.colors[:2, :3].tolist() == [[9, 1, 2], [3, 4, 5]]
    assert int(target.valid_mask.sum()) == 6
    assert not target.colors.flags.writeable
    assert not target.valid_mask.flags.writeable


def test_target_padding_rejects_bad_capacity() -> None:
    with pytest.raises(ValueError, match="max_grid_size"):
        encode_target_grid(ArcGrid(((1,),)), max_grid_size=0)
    with pytest.raises(ValueError, match="exceeds capacity"):
        encode_target_grid(ArcGrid(((1, 2),)), max_grid_size=1)


def test_smoke_fixture_is_multiquery_plumbing_only_and_manifested() -> None:
    loaded = smoke_loaded_dataset()

    assert isinstance(loaded, LoadedDataset)
    assert loaded.tasks == smoke_arc_tasks()
    assert loaded.manifest.source.role == "fixture"
    assert loaded.manifest.plumbing_only is True
    assert loaded.manifest.valid_task_count == 2
    assert loaded.manifest.rejected_task_count == 0
    assert all(file.size_bytes > 0 for file in loaded.manifest.files)
    assert any(len(task.test) > 1 for task in loaded.tasks)


@pytest.mark.parametrize(
    "public_type",
    [
        ArcGrid,
        ArcPair,
        ArcTask,
        ArcQueryEpisode,
        DatasetSource,
        SourceManifest,
        LoadedDataset,
        RowEventConfig,
        AssociativeMemoryFeatureIndices,
        EncodedQueryEpisode,
        GridTarget,
    ],
)
def test_public_data_types_have_numpy_attribute_docstrings(public_type: type) -> None:
    assert "Attributes\n----------" in public_type.__doc__
