"""Tests for the synthetic row-binding qualification."""

from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pytest

try:
    from examples.pp_prop import latent_workspace_binding_qualification as qualification
    from examples.pp_prop.latent_workspace_binding_qualification import (
        BindingQualificationConfig,
        BindingQualificationError,
        require_binding_qualification,
        run_binding_qualification,
        synthetic_binding_tasks,
    )
except ImportError:
    import latent_workspace_binding_qualification as qualification
    from latent_workspace_binding_qualification import (
        BindingQualificationConfig,
        BindingQualificationError,
        require_binding_qualification,
        run_binding_qualification,
        synthetic_binding_tasks,
    )


def test_synthetic_families_cover_binding_shapes_and_context_rules() -> None:
    cases = synthetic_binding_tasks()
    assert tuple(case.name for case in cases) == (
        "identity",
        "recolor",
        "non_square_transpose",
        "demo_dependent_row_reverse",
    )
    assert all(len(case.task.train) == 2 for case in cases)
    assert all(len(case.task.test) == 1 for case in cases)

    identity = cases[0].task.test[0]
    assert identity.output is not None
    np.testing.assert_array_equal(identity.input.as_array(), identity.output.as_array())

    recolor = cases[1].task.test[0]
    assert recolor.output is not None
    assert recolor.input.cells == ((1, 0), (0, 1))
    assert recolor.output.cells == ((4, 0), (0, 4))

    transpose = cases[2].task.test[0]
    assert transpose.output is not None
    assert (transpose.input.height, transpose.input.width) == (2, 3)
    assert (transpose.output.height, transpose.output.width) == (3, 2)

    row_reverse = cases[3].task
    assert row_reverse.train[0].output is not None
    assert row_reverse.train[0].output.cells == tuple(
        reversed(row_reverse.train[0].input.cells)
    )
    assert row_reverse.test[0].output is not None
    assert row_reverse.test[0].output.cells == tuple(
        reversed(row_reverse.test[0].input.cells)
    )


def test_every_supervised_tick_is_latent_for_every_family() -> None:
    rows = qualification.RowEventConfig(max_demonstrations=2)
    batch = qualification._binding_batch(synthetic_binding_tasks(), rows)
    event_valid = batch.events[..., rows.valid_slice.start] > 0.5
    latent = batch.advances & ~event_valid
    supervised = batch.loss_mask > 0.0
    assert np.all(latent[supervised])
    np.testing.assert_array_equal(
        latent.sum(axis=0), np.full((len(synthetic_binding_tasks()),), 30)
    )


@pytest.mark.parametrize(
    ("updates", "learning_rate", "neuron_count", "recurrent_edges"),
    ((0, 0.01, 64, 64), (1, 0.0, 64, 64), (1, 0.01, 63, 64), (1, 0.01, 64, 0)),
)
def test_config_rejects_invalid_training_or_physical_sizes(
    updates: int,
    learning_rate: float,
    neuron_count: int,
    recurrent_edges: int,
) -> None:
    with pytest.raises(ValueError):
        BindingQualificationConfig(
            updates=updates,
            learning_rate=learning_rate,
            neuron_count=neuron_count,
            recurrent_edges=recurrent_edges,
        )


def test_qualification_gate_is_strict_about_learning_movement_and_completion() -> None:
    passing = qualification._qualification_gate(
        exact_before=0,
        exact_after=4,
        family_count=4,
        loss_before=3.0,
        loss_after=1.0,
        parameters_moved=True,
    )
    assert passing["qualified"] is True
    assert passing["violations"] == []

    no_learning = qualification._qualification_gate(
        exact_before=1,
        exact_after=1,
        family_count=4,
        loss_before=3.0,
        loss_after=1.0,
        parameters_moved=True,
    )
    assert no_learning["qualified"] is False
    assert "no_new_exact_grid" in no_learning["violations"]
    assert "not_all_families_exact" in no_learning["violations"]

    no_movement = qualification._qualification_gate(
        exact_before=0,
        exact_after=4,
        family_count=4,
        loss_before=1.0,
        loss_after=math.inf,
        parameters_moved=False,
    )
    assert no_movement["qualified"] is False
    assert set(no_movement["violations"]) == {
        "parameters_did_not_move",
        "loss_not_finite_or_improved",
    }


def test_compiled_training_driver_contains_no_python_iteration() -> None:
    source = inspect.getsource(qualification._compiled_training_losses)
    tree = ast.parse(source)
    assert "brainstate.transform.for_loop" in source
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))


def test_real_diagnostic_exercises_model_compiler_row_loss_and_model_candidates() -> (
    None
):
    report = run_binding_qualification(
        BindingQualificationConfig(
            updates=1,
            neuron_count=64,
            recurrent_edges=64,
            readout_width=8,
            context_memory_width=2,
            seed=37,
        )
    )

    assert report["mode"] == "synthetic_row_binding_diagnostic"
    assert report["family_count"] == 4
    assert report["model"] == {
        "decoder_mode": "row_refinement",
        "neuron_count": 64,
        "recurrent_edges": 64,
        "context_memory_width": 2,
        "training_output_width": 360,
        "checkpoint_output_width": 9060,
    }
    assert report["compiler"]["pp_prop_compiled"] is True
    assert report["compiler"]["learner_type"]
    assert report["movement"]["parameters_moved"] is True
    assert report["movement"]["changed_parameter_count"] > 0
    assert len(report["losses"]["updates"]) == 1
    assert math.isfinite(report["losses"]["before"])
    assert math.isfinite(report["losses"]["after"])

    assert len(report["families"]) == 4
    for family in report["families"]:
        assert family["candidate_provenance"] == "model"
        assert len(family["before_candidates"]) in (1, 2)
        assert len(family["after_candidates"]) in (1, 2)
        assert all("grid" in candidate for candidate in family["before_candidates"])
        assert all("grid" in candidate for candidate in family["after_candidates"])

    requirements = report["requirements"]
    assert report["qualified"] is all(requirements.values())
    if report["exact_grids"]["after"] == report["exact_grids"]["before"]:
        assert "no_new_exact_grid" in report["violations"]


def test_require_binding_qualification_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = {"qualified": False, "violations": ["no_new_exact_grid"]}
    monkeypatch.setattr(
        qualification, "run_binding_qualification", lambda _config: failed
    )
    with pytest.raises(BindingQualificationError, match="no_new_exact_grid"):
        require_binding_qualification(BindingQualificationConfig())

    passed = {"qualified": True, "violations": []}
    monkeypatch.setattr(
        qualification, "run_binding_qualification", lambda _config: passed
    )
    assert require_binding_qualification(BindingQualificationConfig()) is passed
