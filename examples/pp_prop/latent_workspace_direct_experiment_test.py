"""Tests for staged direct ARC corpus experiments."""

from __future__ import annotations

import importlib
import json

import brainstate
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_direct_experiment")


def _task(task_id: str, color: int) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((color,),)), ArcGrid(((color,),))),
            ArcPair(ArcGrid((((color + 1) % 10,),)), ArcGrid((((color + 1) % 10,),))),
        ),
        test=(ArcPair(ArcGrid(((color,),)), ArcGrid(((color,),))),),
        task_id=task_id,
    )


def test_deterministic_task_split_is_order_independent_and_disjoint() -> None:
    subject = _subject()
    tasks = tuple(_task(str(index), index) for index in range(5))

    fit_a, validation_a = subject.deterministic_task_split(tasks, 2)
    fit_b, validation_b = subject.deterministic_task_split(tuple(reversed(tasks)), 2)

    assert [task.task_id for task in fit_a] == [task.task_id for task in fit_b]
    assert [task.task_id for task in validation_a] == [
        task.task_id for task in validation_b
    ]
    assert {task.task_id for task in fit_a}.isdisjoint(
        task.task_id for task in validation_a
    )


def test_training_catalog_contains_loo_and_official_queries() -> None:
    catalog = _subject().training_episode_catalog((_task("a", 1),))

    assert len(catalog) == 3
    assert all(len(task.test) == 1 for task in catalog)
    assert all(task.test[0].output is not None for task in catalog)


def test_training_chunk_sampling_is_brainstate_seed_deterministic() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    catalog = subject.training_episode_catalog((_task("a", 1), _task("b", 3)))
    first = subject.sample_training_chunk(
        catalog,
        row_config,
        brainstate.random.RandomState(17),
        updates=2,
        batch_size=2,
        augment=True,
    )
    second = subject.sample_training_chunk(
        catalog,
        row_config,
        brainstate.random.RandomState(17),
        updates=2,
        batch_size=2,
        augment=True,
    )

    for name in first.__dataclass_fields__:
        assert np.asarray(getattr(first, name)).tobytes() == np.asarray(
            getattr(second, name)
        ).tobytes()


@pytest.mark.parametrize(
    "changes",
    [
        {"training_updates": 0},
        {"training_chunk_size": 3},
        {"device": "tpu"},
        {"learning_rate": 0.0},
        {"validation_task_count": True},
    ],
)
def test_experiment_config_fails_closed(tmp_path, changes: dict[str, object]) -> None:
    subject = _subject()
    values = {
        "source_manifest": tmp_path / "sources.json",
        "output_dir": tmp_path / "out",
        "device": "cpu",
        "training_updates": 4,
        "training_chunk_size": 2,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        subject.DirectExperimentConfig(**values)


def _write_task(path, task: ArcTask) -> None:
    payload = {
        "train": [
            {"input": pair.input.to_list(), "output": pair.output.to_list()}
            for pair in task.train
        ],
        "test": [
            {"input": pair.input.to_list(), "output": pair.output.to_list()}
            for pair in task.test
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tiny_manifest(tmp_path):
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    _write_task(training / "a.json", _task("a", 0))
    _write_task(training / "b.json", _task("b", 2))
    _write_task(training / "c.json", _task("c", 4))
    _write_task(evaluation / "z.json", _task("z", 7))
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "tiny training",
                        "role": "train",
                        "version": "1",
                        "path": str(training),
                        "license_reference": "test",
                        "format": "task_json",
                    },
                    {
                        "name": "tiny evaluation",
                        "role": "evaluation",
                        "version": "1",
                        "path": str(evaluation),
                        "license_reference": "test",
                        "format": "task_json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_tiny_end_to_end_run_writes_exact_validation_artifact(tmp_path) -> None:
    subject = _subject()
    manifest = _tiny_manifest(tmp_path)
    output = tmp_path / "out"
    config = subject.DirectExperimentConfig(
        source_manifest=manifest,
        output_dir=output,
        device="cpu",
        validation_task_count=1,
        training_updates=1,
        training_chunk_size=1,
        training_batch_size=1,
        encoder_width=4,
        hidden_width=8,
        decoder_width=6,
        recurrent_layers=1,
        augment=False,
    )

    result = subject.run_experiment(config)

    assert result["data"]["training_task_count"] == 3
    assert result["data"]["validation_task_count"] == 1
    assert result["data"]["scored_split"] == "training_task_validation"
    assert result["evaluation"]["task_count"] == 1
    assert result["evaluation"]["query_count"] == 1
    assert isinstance(result["evaluation"]["strict_task_pass_at_1_count"], int)
    assert result["model"]["parameters_moved"] is True
    assert (output / "result.json").is_file()


def test_source_manifest_and_sampling_fail_closed(tmp_path) -> None:
    subject = _subject()
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        subject.source_declarations(invalid)
    with pytest.raises(ValueError, match="smaller"):
        subject.deterministic_task_split((_task("a", 1),), 1)
    with pytest.raises(TypeError, match="RandomState"):
        subject.sample_training_chunk(
            (_task("a", 1),),
            RowEventConfig(max_demonstrations=2, max_grid_size=3),
            object(),
            updates=1,
            batch_size=1,
            augment=False,
        )


def test_cli_builds_config_and_reports_evaluation(monkeypatch, tmp_path, capsys) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {"evaluation": {"strict_task_pass_at_1_count": 0}}

    monkeypatch.setattr(subject, "run_experiment", fake_run)
    exit_code = subject.main(
        [
            "--source-manifest",
            str(tmp_path / "sources.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--training-updates",
            "2",
            "--training-chunk-size",
            "1",
            "--no-augment",
        ]
    )

    assert exit_code == 0
    assert observed["config"].augment is False
    assert "strict_task_pass_at_1_count" in capsys.readouterr().out
