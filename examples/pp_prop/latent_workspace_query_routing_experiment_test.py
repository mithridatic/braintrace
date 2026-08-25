"""Tests for the V48 query-routing model and experiment."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_expert_training import (
    TaskGatedPPPropTrainer,
    parameter_leaf_arrays,
)
from examples.pp_prop.latent_workspace_online_training import (
    OnlineTrainingChunk,
    load_online_checkpoint,
    parameter_digest,
    sample_online_training_chunk,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_query_routing_model"
    )


def _runner_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_query_routing_experiment"
    )


def _curriculum_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_diverse_curriculum"
    )


def _experiment_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_experiment"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _model(seed: int = 41):
    subject = _subject()
    return subject.QueryRoutingGatedMemoryRNN(
        subject.QueryRoutingConfig(
            memory_width=8, expert_count=12, program_count=16, seed=seed
        )
    )


def _config(seed: int = 41):
    subject = _subject()
    return subject.QueryRoutingConfig(
        memory_width=8, expert_count=12, program_count=16, seed=seed
    )


def test_event_extension_reconstructs_query_grid_losslessly() -> None:
    subject = _subject()
    row_config = RowEventConfig()
    rng = np.random.default_rng(5)
    events = np.zeros((4, 360, subject.BASE_INPUT_WIDTH), dtype=np.float32)
    colors = np.zeros((30, 30, 10), dtype=np.float32)
    mask = np.zeros((30, 30), dtype=np.float32)
    query = rng.integers(0, 10, size=(5, 7))
    for row in range(30):
        if row < 5:
            for column in range(7):
                colors[row, column, query[row, column]] = 1.0
                mask[row, column] = 1.0
        events[:, -30 + row, row_config.input_color_slice] = colors[
            row
        ].reshape(-1)
        events[:, -30 + row, row_config.input_mask_slice] = mask[row]

    extended = subject.extend_events_query_grid(events)

    assert extended.shape == (4, 360, subject.MODEL_INPUT_WIDTH)
    assert (extended[:, :-30, subject.BASE_INPUT_WIDTH :] == 0.0).all()
    block_colors = extended[0, -1, subject.BLOCK_COLOR_SLICE].reshape(900, 10)
    block_valid = extended[0, -1, subject.BLOCK_VALID_SLICE]
    assert np.array_equal(block_colors, colors.reshape(900, 10))
    assert np.array_equal(block_valid, mask.reshape(900))
    decoded = block_colors.argmax(axis=-1).reshape(30, 30)
    assert (decoded[:5, :7] == query).all()
    with pytest.raises(ValueError, match="3831"):
        subject.extend_events_query_grid(np.zeros((360, 12), dtype=np.float32))


def test_event_extension_depends_only_on_query_inputs() -> None:
    subject = _subject()
    rng = np.random.default_rng(11)
    base = rng.random((2, 360, subject.BASE_INPUT_WIDTH), dtype=np.float32)
    altered = base.copy()
    # Demonstration-output channels (offset 530..830) never enter the block.
    altered[..., 530:830] = rng.random((2, 360, 300), dtype=np.float32)
    base_extended = subject.extend_events_query_grid(base)
    altered_extended = subject.extend_events_query_grid(altered)
    assert np.array_equal(
        base_extended[..., subject.BASE_INPUT_WIDTH :],
        altered_extended[..., subject.BASE_INPUT_WIDTH :],
    )


def _match_at(model, row, height, width):
    return np.asarray(
        model._match_features(
            jnp.asarray([float(row)]),
            jnp.asarray([float(height)]),
            jnp.asarray([float(width)]),
        )
    )


def test_match_features_cover_representative_geometry() -> None:
    model = _model()
    matches = _match_at(model, row=2, height=4, width=3)
    source = lambda r, c: r * 30 + c

    assert matches[0, 1, source(2, 1), 0] == 1.0  # identity
    shift_index = 1 + (1 + 3) * 7 + (-1 + 3)
    assert matches[0, 1, source(3, 0), shift_index] == 1.0  # shift (1, -1)
    assert matches[0, 1, source(1, 2), 51] == 1.0  # transpose main
    assert matches[0, 1, source(2, 1), 54] == 1.0  # flip left-right
    assert matches[0, 3, source(1, 1), 58] == 1.0  # upscale factor 2
    assert matches[0, 1, source(1, 1), 61] == 1.0  # mirror-concat top
    assert matches.shape == (1, 30, 900, 65)
    assert set(np.unique(matches)).issubset({0.0, 1.0})
    # Exactly one source matches identity per output cell.
    assert matches[0, :, :, 0].sum(axis=-1).max() == 1.0


def test_routing_logits_are_zero_without_query_block() -> None:
    subject = _subject()
    model = _model()
    event = np.zeros((2, subject.MODEL_INPUT_WIDTH), dtype=np.float32)
    hidden = jnp.zeros((2, 3 * 8), dtype=jnp.float32)
    logits = model._routing_logits(hidden, jnp.asarray(event))
    assert logits.shape == (2, 30, 10)
    assert np.allclose(np.asarray(logits), 0.0)


def test_config_validation() -> None:
    subject = _subject()
    with pytest.raises(ValueError, match="13731"):
        subject.QueryRoutingConfig(input_width=3831)
    with pytest.raises(ValueError, match="architecture_version"):
        subject.QueryRoutingConfig(architecture_version="other")
    with pytest.raises(ValueError, match="program_count"):
        subject.QueryRoutingConfig(program_count=1)
    with pytest.raises(ValueError, match="routing_scale"):
        subject.QueryRoutingConfig(routing_scale=0.0)
    with pytest.raises(TypeError):
        subject.QueryRoutingGatedMemoryRNN(config=object())


def test_checkpoint_roundtrip_binds_v48_schema(tmp_path) -> None:
    subject = _subject()
    model = _model()
    path = tmp_path / "checkpoint.npz"
    digest = save_online_checkpoint(model, path)

    restored, metadata = load_online_checkpoint(path)

    assert metadata["architecture"]["architecture_version"] == (
        "query_routing_gated_memory_v48"
    )
    assert parameter_digest(restored) == digest
    leaves = parameter_leaf_arrays(model)
    assert len(leaves) == 19
    assert "routing_query.query#0" in leaves
    assert leaves["routing_query.query#0"].shape == (16, 76)


def test_routing_table_changes_emitted_logits() -> None:
    subject = _subject()
    model = _model()
    brainstate.nn.init_all_states(model, batch_size=2)
    event = np.zeros((2, subject.MODEL_INPUT_WIDTH), dtype=np.float32)
    row_config = RowEventConfig()
    event[:, row_config.phase_slice] = np.asarray([0.0, 1.0], dtype=np.float32)
    event[:, -900:] = 1.0
    event[:, subject.BLOCK_COLOR_SLICE] = np.tile(
        np.eye(10, dtype=np.float32).reshape(1, -1), (2, 900)
    )[:, :9000]
    hidden_event = jnp.asarray(event)
    before = np.asarray(model.update(hidden_event))
    table_state = model.routing_query.query
    table_state.value = jnp.asarray(
        np.random.default_rng(3).normal(size=(16, 76)), dtype=jnp.float32
    )
    after = np.asarray(model.update(hidden_event))
    assert not np.allclose(before, after)


def _extended_chunk(seed: int = 51) -> OnlineTrainingChunk:
    curriculum_module = _curriculum_subject()
    experiment_module = _experiment_subject()
    runner = _runner_subject()
    curriculum = curriculum_module.generate_diverse_curriculum(
        curriculum_module.DiverseCurriculumConfig(
            task_count=12, max_grid_size=6, min_demonstrations=2, max_demonstrations=2
        ),
        brainstate.random.RandomState(seed),
    )
    catalog = experiment_module.training_episode_catalog(curriculum.tasks)
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    chunk = sample_online_training_chunk(
        catalog,
        row_config,
        brainstate.random.RandomState(seed + 1),
        updates=2,
        batch_size=2,
        augment=False,
    )
    return runner._extend_chunk(chunk)


def test_compiler_registers_routing_table_without_exclusions() -> None:
    model = _model()
    trainer = TaskGatedPPPropTrainer(model, batch_size=2)
    report = trainer.learner.report
    # Decode-side heads are non-temporal current-step parameters by design;
    # the routing table must share that status and must never trigger a
    # weight-to-weight exclusion (a stacked trainable primitive).
    kinds = [item.kind.value for item in report.diagnostics]
    assert "relation_excluded_weight_to_weight" not in kinds
    excluded = [".".join(map(str, path)) for path, _ in report.excluded_weights]
    assert not any(path.startswith("recurrent") for path in excluded)
    assert "routing_query.query" in excluded
    recurrent_registered = [
        ".".join(map(str, path)) for path, _ in report.etrace_weights
    ]
    assert any(path.startswith("recurrent") for path in recurrent_registered)


def test_compiled_training_moves_every_leaf() -> None:
    model = _model()
    before = parameter_leaf_arrays(model)
    trainer = TaskGatedPPPropTrainer(model, batch_size=2)
    losses, norms = trainer.train_chunk(_extended_chunk())

    assert np.isfinite(np.asarray(losses)).all()
    assert all(np.isfinite(v) and v > 0.0 for v in norms.values())
    after = parameter_leaf_arrays(model)
    unmoved = {
        name
        for name in before
        if before[name].tobytes() == after[name].tobytes()
    }
    assert not unmoved, unmoved


def _task(index: int, *, role: str = "train") -> ArcTask:
    first = index % 9 + 1
    second = (index + 1) % 9 + 1
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((first, 0),)), ArcGrid(((second, 0),))),
            ArcPair(ArcGrid(((second, 0),)), ArcGrid(((first, 0),))),
        ),
        test=(ArcPair(ArcGrid(((first, 0),)), ArcGrid(((second, 0),))),),
        task_id=f"{role}-{index:03d}",
    )


def test_tiny_v48_run_binds_scopes_leaves_gates_and_artifact(
    tmp_path, monkeypatch
) -> None:
    runner = _runner_subject()
    training = tuple(_task(index) for index in range(4))
    evaluation = tuple(
        _task(index, role="evaluation") for index in range(2)
    )

    class Manifest:
        def __init__(self, role: str):
            self.role = role

        def to_dict(self):
            return {"source": {"role": self.role}, "valid_task_count": 4}

    corpora = SimpleNamespace(
        training=training,
        evaluation=evaluation,
        loaded=(
            SimpleNamespace(manifest=Manifest("train")),
            SimpleNamespace(manifest=Manifest("evaluation")),
        ),
    )
    monkeypatch.setattr(runner, "load_corpora", lambda _: corpora)
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "7" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = runner.QueryRoutingExperimentConfig(
        output_dir=tmp_path / "artifact",
        source_manifest=tmp_path / "sources.json",
        device="cpu",
        seed=41,
        synthetic_seed=51,
        holdout_seed=61,
        synthetic_task_count=12,
        holdout_task_count=12,
        max_grid_size=6,
        min_demonstrations=2,
        max_demonstrations=2,
        training_updates=2,
        training_chunk_size=2,
        training_batch_size=2,
        memory_width=8,
        expert_count=12,
        program_count=16,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
    )

    result = runner.run_query_routing_experiment(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "7" * 40
    assert result["model"]["architecture"]["architecture_version"] == (
        "query_routing_gated_memory_v48"
    )
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert all(result["model"]["parameter_leaves_moved"].values())
    assert len(result["model"]["parameter_leaves_moved"]) == 19
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert result["training"]["finite"] is True
    assert result["evaluation_in_library"]["task_count"] == 4
    assert result["evaluation_fold_zero"]["task_count"] == 2
    for gate in (
        "mechanism_gate_passed",
        "anti_collapse_gate_passed",
        "routing_gate_passed",
    ):
        assert isinstance(result[gate], bool)
    assert isinstance(result["routing_exact_families"], dict)
    assert (config.output_dir / "checkpoint.npz").exists()
    assert stored["configuration"] == config.to_dict()
