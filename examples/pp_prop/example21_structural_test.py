import ast
import copy
import hashlib
import importlib.util
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from examples.pp_prop.arc_contracts import decode_prediction
from examples.pp_prop.dale_candidates import DaleMeasurements, measure_dale_candidates

_SPEC = importlib.util.spec_from_file_location(
    "example21_structural", Path(__file__).with_name("example21_structural.py")
)
assert _SPEC is not None and _SPEC.loader is not None
structural = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(structural)


def _public_api_nodes(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = []

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    nodes.append((prefix + node.name, node))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")

    visit(tree.body)
    return nodes


def _has_value_return(node):
    class ReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Return(self, return_node):
            self.found |= return_node.value is not None

        def visit_FunctionDef(self, function_node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

    visitor = ReturnVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.found


def test_public_api_return_audit_finds_control_flow_returns_without_nested_callables():
    conditional = ast.parse(
        "def conditional(flag):\n"
        "    if flag:\n"
        "        return 1\n"
    ).body[0]
    nested = ast.parse(
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
    ).body[0]
    assert _has_value_return(conditional)
    assert not _has_value_return(nested)


@pytest.mark.parametrize(
    "path",
    (
        Path(__file__).with_name("21-braincell-arc.py"),
        Path(__file__).with_name("example21_structural.py"),
    ),
)
def test_public_apis_have_numpy_contract_sections(path):
    required_sections = {"Parameters", "Returns", "Yields", "Attributes", "Raises"}
    missing = []
    for name, node in _public_api_nodes(path):
        doc = ast.get_docstring(node, clean=False) or ""
        lines = [line.strip() for line in doc.splitlines()]
        sections = {line for line in lines if line in required_sections}
        if not doc.strip():
            missing.append(f"{name}: summary")
            continue
        if isinstance(node, ast.ClassDef):
            if not sections & {"Parameters", "Attributes"}:
                missing.append(f"{name}: Parameters or Attributes")
            continue
        arguments = [
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        ]
        if node.args.vararg:
            arguments.append(node.args.vararg.arg)
        if node.args.kwarg:
            arguments.append(node.args.kwarg.arg)
        parameter_names = set()
        section = None
        for line in lines:
            if line in required_sections:
                section = line
            elif section == "Parameters" and ":" in line:
                parameter_names.update(
                    name.strip().lstrip("*")
                    for name in line.split(":", 1)[0].split(",")
                )
        if arguments and "Parameters" not in sections:
            missing.append(f"{name}: Parameters")
        missing.extend(
            f"{name}: parameter {argument}"
            for argument in arguments
            if "Parameters" in sections and argument not in parameter_names
        )
        returns_value = _has_value_return(node)
        yields_value = any(
            isinstance(statement, (ast.Yield, ast.YieldFrom))
            for statement in ast.walk(node)
        )
        if returns_value and "Returns" not in sections:
            missing.append(f"{name}: Returns")
        if yields_value and "Yields" not in sections:
            missing.append(f"{name}: Yields")
    assert not missing


def test_rejected_validation_messages_include_corrective_actions():
    with pytest.raises(ValueError, match="pass at least one strict result"):
        structural.pruning_mask(np.ones(20), (False,))
    for path in (
        Path(__file__).with_name("21-braincell-arc.py"),
        Path(__file__).with_name("example21_structural.py"),
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        message_nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id in {
                    "AssertionError",
                    "IndexError",
                    "KeyError",
                    "RuntimeError",
                    "TypeError",
                    "ValueError",
                }:
                    message_nodes.append((node, node.exc.args[:1]))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "error"
            ):
                message_nodes.append((node, node.args[:1]))
        action_words = re.compile(
            r"\b(pass|provide|use|select|reduce|add|disable|restore|include|return|"
            r"choose|increase|assign|initialize|write|set|ensure|remove|declare|"
            r"load|supply|run|check|make|match|pick|keep|preserve|record|correct)\b",
            re.IGNORECASE,
        )
        for node, arguments in message_nodes:
            if not arguments or not isinstance(arguments[0], (ast.Constant, ast.JoinedStr)):
                continue
            expression = arguments[0]
            if isinstance(expression, ast.Constant):
                text = expression.value
            else:
                text = "".join(
                    value.value if isinstance(value, ast.Constant) else "value"
                    for value in expression.values
                )
            assert text and text[0].isupper(), f"{path}:{node.lineno}: {text}"
            assert ";" in text, f"{path}:{node.lineno}: {text}"
            assert action_words.search(text.split(";", 1)[1]), (
                f"{path}:{node.lineno}: {text}"
            )


class _EagerTransform:
    @staticmethod
    def for_loop(function, *values):
        outputs = [function(*items) for items in zip(*values)]
        if isinstance(outputs[0], tuple):
            return tuple(np.stack(items) for items in zip(*outputs))
        return np.stack(outputs)


def test_evidence_scores_owners_ranking_and_twins_are_direct_and_stable():
    rows = np.array([[0.0, 2.0, 1.0], [0.0, 4.0, 4.0]])
    np.testing.assert_allclose(
        structural.normalize_task_rows(rows), [[0.0, 1.0, 0.5], [0.0, 1.0, 1.0]]
    )
    score = structural.neuron_contribution(rows, rows * 2, rows * 3)
    np.testing.assert_allclose(score, [0.0, 1.0, 1.0])
    assert structural.task_owners(np.array([[0, 1, 1], [0, 1, 0]])) == (
        (), (0, 1), (0,)
    )
    assert structural.stable_rank(np.array([1.0, 1.0, 0.0])) == (2, 0, 1)
    twins = structural.structural_twins(
        3,
        input_sources=((0,), (0,), (1,)),
        recurrent_incoming=((2,), (2,), ()),
        recurrent_outgoing=((2,), (2,), (0, 1)),
        dale_labels=(0, 0, 0),
        mechanisms=(("hh",), ("hh",), ("hh",)),
    )
    assert twins == ((0, 1), (2,))


def test_dale_task_evidence_is_scale_invariant_for_both_selection_arms():
    activity = np.array([[1.0, 0.2, 0.8, 0.1], [0.3, 3.0, 0.2, 0.4]])
    gradients = np.array([[1.0, 0.1, 0.5, 0.2], [0.1, 3.0, 0.2, 0.4]])
    source = np.array([0, 0, 1, 1])
    base_activity, base_gradient = structural.dale_task_evidence(
        activity, source, gradients
    )
    common = {
        "parent_id": "accepted-parent",
        "rows": source,
        "weights": np.array([1.0, 1.0, -1.0, -1.0]),
        "task_ownership": np.ones(4),
        "lesion_evidence": np.ones(4),
    }
    base = measure_dale_candidates(DaleMeasurements(
        activity=base_activity, gradient_mass=base_gradient, **common
    ), fraction=0.5)
    for task in range(activity.shape[0]):
        for field in ("activity", "gradient"):
            scaled_activity = activity.copy()
            scaled_gradient = gradients.copy()
            values = scaled_activity if field == "activity" else scaled_gradient
            values[task] *= 100.0
            scaled_activity, scaled_gradient = structural.dale_task_evidence(
                scaled_activity, source, scaled_gradient
            )
            scaled = measure_dale_candidates(DaleMeasurements(
                activity=scaled_activity, gradient_mass=scaled_gradient, **common
            ), fraction=0.5)
            np.testing.assert_array_equal(base.excitatory, scaled.excitatory)
            np.testing.assert_array_equal(base.inhibitory, scaled.inhibitory)
    with pytest.raises(ValueError, match="task-by-item"):
        structural.dale_task_evidence(np.ones(4), source, gradients)
    with pytest.raises(ValueError, match="same task count"):
        structural.dale_task_evidence(activity, source, gradients[:1])
    with pytest.raises(ValueError, match="match gradient"):
        structural.dale_task_evidence(activity, source[:1], gradients)
    with pytest.raises(ValueError, match="nonnegative"):
        structural.dale_task_evidence(activity, [-1, 0, 1, 1], gradients)


def test_causal_block_lesion_evidence_measures_each_source_loss():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 0, 1]), np.array([1, 2, 2]),
        np.asarray(structural.encode_dale_weights(
            [0.5, 0.25, -0.75], [1, 1, -1]
        )), np.ones((3, 1)), np.array([1, 1, -1], dtype=np.int8), ((),) * 3,
    )
    evidence = structural.causal_block_lesion_evidence(
        topology,
        np.array([[2.0, 3.0, 0.0]]),
        task_output=lambda source: np.array([
            3.75 if source is None else
            (2.25 if source == 0 else 1.5 if source == 1 else 3.75)
        ]),
        transform=_EagerTransform,
    )
    np.testing.assert_allclose(evidence, [[2 / 3, 1.0, 0.0]])
    with pytest.raises(ValueError, match="task-by-neuron"):
        structural.causal_block_lesion_evidence(
            topology, np.ones(3), task_output=lambda _source: np.zeros(1)
        )


def test_causal_block_lesion_validates_outputs_and_normalizes_vectors():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0]), np.array([1]), np.array([1.0]),
        np.ones((2, 1)), np.zeros(2, dtype=np.int8), ((), ()),
    )
    with pytest.raises(TypeError, match="intervention"):
        structural.causal_block_lesion_evidence(
            topology, np.ones((1, 2)), task_output=object(), transform=_EagerTransform
        )
    with pytest.raises(ValueError, match="one row"):
        structural.causal_block_lesion_evidence(
            topology, np.ones((1, 2)),
            task_output=lambda _source: np.zeros(2), transform=_EagerTransform,
        )
    with pytest.raises(ValueError, match="match"):
        structural.causal_block_lesion_evidence(
            topology, np.ones((1, 2)),
            task_output=lambda source: np.zeros(1 if source is None else 2),
            transform=_EagerTransform,
        )

    def output(source):
        value = 0.0 if source is None else float(source + 1)
        return np.array([[value, 2.0 * value]])

    evidence = structural.causal_block_lesion_evidence(
        topology, np.ones((1, 2)), task_output=output, transform=_EagerTransform
    )
    np.testing.assert_allclose(evidence, [[0.5, 1.0]])


def test_connection_contribution_uses_csr_row_as_source():
    indptr = np.array([0, 2, 3])
    values = np.array([2.0, -1.0, 4.0])
    spikes = np.array([[3.0, 5.0], [1.0, 0.0]])
    gradients = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    scores = structural.connection_contribution(indptr, values, spikes, gradients)
    np.testing.assert_allclose(scores, [0.45, 0.6, 0.75])


def test_structural_evidence_aggregates_real_sparse_edges_by_source_and_target():
    topology = structural.SparseTopology(
        np.array([0]), np.array([0]), np.array([1.0]),
        np.array([0, 0, 1]), np.array([1, 2, 0]), np.array([2.0, 1.0, 3.0]),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    evidence = structural.structural_evidence(
        topology,
        readout_effect=np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]]),
        spikes=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
        gradient_mass=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
    )
    assert evidence["neuron_scores"].shape == (3,)
    assert evidence["connection_scores"].shape == (3,)
    assert evidence["owners"] == ((0,), (1,), (0,))
    with pytest.raises(ValueError, match="task-by-edge"):
        structural.structural_evidence(topology, np.zeros((1, 3)),
                                       np.zeros((1, 3)), np.zeros((1, 2)))


def test_pruning_masks_exact_ceiling_and_validation_guard():
    scores = np.arange(21.0)
    mask = structural.pruning_mask(scores, validation_strict=(True, False))
    assert np.flatnonzero(~mask).tolist() == [0, 1]
    with pytest.raises(ValueError, match="(?i)validation"):
        structural.pruning_mask(scores, validation_strict=(False, False))
    with pytest.raises(ValueError, match="positive"):
        structural.mutation_count(0)


def test_neuron_pruning_is_fail_closed_and_requires_one_score_per_neuron():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.ones((21, 1)), np.zeros(21), ((),) * 21,
    )
    with pytest.raises(ValueError, match="(?i)validation"):
        structural.prune_neurons(topology, np.arange(21.0), (False, False))
    with pytest.raises(ValueError, match="(?i)one score"):
        structural.prune_neurons(topology, np.arange(20.0), (True, False))
    mask = structural.prune_neurons(topology, np.arange(21.0), (True, False))
    assert np.flatnonzero(~mask).tolist() == [0, 1]


def test_compaction_remaps_sparse_edges_and_optimizer_rows():
    topology = structural.SparseTopology(
        input_source=np.array([0, 1, 1]), input_target=np.array([0, 1, 2]),
        input_value=np.array([1.0, 2.0, 3.0]),
        recurrent_source=np.array([0, 1, 2]), recurrent_target=np.array([1, 2, 0]),
        recurrent_value=np.array([4.0, 5.0, 6.0]),
        readout=np.arange(6.0).reshape(3, 2), dale=np.array([0, 1, -1]),
        mechanisms=(('a',), ('b',), ('c',)),
    )
    adam = structural.StructuralAdam(
        neuron_first=np.arange(6.0).reshape(3, 2),
        neuron_second=np.arange(6.0, 12.0).reshape(3, 2),
        input_first=np.array([10.0, 20.0, 30.0]), input_second=np.array([1.0, 2.0, 3.0]),
        recurrent_first=np.array([40.0, 50.0, 60.0]), recurrent_second=np.array([4.0, 5.0, 6.0]),
        step=7,
    )
    compact, mapped, reset = structural.compact(topology, np.array([True, False, True]), adam)
    assert compact.neuron_count == 2
    assert list(zip(compact.recurrent_source, compact.recurrent_target)) == [(1, 0)]
    np.testing.assert_array_equal(mapped.neuron_first, adam.neuron_first[[0, 2]])
    np.testing.assert_array_equal(mapped.recurrent_first, [60.0])
    assert mapped.step == 7 and reset


def test_canonicalization_sorts_sparse_rows_with_optimizer_values():
    topology = structural.SparseTopology(
        input_source=np.array([1, 0, 1]),
        input_target=np.array([2, 1, 0]),
        input_value=np.array([10.0, 20.0, 30.0]),
        recurrent_source=np.array([2, 0, 1, 0]),
        recurrent_target=np.array([1, 2, 0, 1]),
        recurrent_value=np.array([40.0, 50.0, 60.0, 70.0]),
        readout=np.arange(6.0).reshape(3, 2),
        dale=np.array([0, 1, -1], dtype=np.int8),
        mechanisms=((), (), ()),
        owner_codes=np.array([4, 5, 6], dtype=np.int16),
        neuron_ids=np.array([10, 20, 30], dtype=np.int32),
    )
    optimizer = structural.StructuralAdam(
        neuron_first=np.arange(6.0).reshape(3, 2),
        neuron_second=np.arange(6.0, 12.0).reshape(3, 2),
        input_first=np.array([100.0, 200.0, 300.0]),
        input_second=np.array([101.0, 201.0, 301.0]),
        recurrent_first=np.array([400.0, 500.0, 600.0, 700.0]),
        recurrent_second=np.array([401.0, 501.0, 601.0, 701.0]),
        input_step=11,
        recurrent_step=12,
        readout_step=13,
    )

    canonical, mapped = structural.canonicalize_topology_and_optimizer(
        topology, optimizer
    )

    assert list(zip(canonical.input_source, canonical.input_target)) == [
        (0, 1), (1, 0), (1, 2)
    ]
    assert list(zip(canonical.recurrent_source, canonical.recurrent_target)) == [
        (0, 1), (0, 2), (1, 0), (2, 1)
    ]
    np.testing.assert_array_equal(canonical.input_value, [20.0, 30.0, 10.0])
    np.testing.assert_array_equal(mapped.input_first, [200.0, 300.0, 100.0])
    np.testing.assert_array_equal(canonical.recurrent_value, [70.0, 50.0, 60.0, 40.0])
    np.testing.assert_array_equal(mapped.recurrent_first, [700.0, 500.0, 600.0, 400.0])
    np.testing.assert_array_equal(canonical.owner_codes, topology.owner_codes)
    np.testing.assert_array_equal(canonical.neuron_ids, topology.neuron_ids)
    assert (mapped.input_step, mapped.recurrent_step, mapped.readout_step) == (
        11, 12, 13
    )


def test_structural_mutations_preserve_stable_ids_and_task_owners():
    topology = structural.SparseTopology(
        input_source=np.array([0]), input_target=np.array([0]),
        input_value=np.array([2.0]), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 2]), recurrent_value=np.array([4.0, 5.0]),
        readout=np.ones((3, 1)), dale=np.zeros(3, dtype=np.int8),
        mechanisms=((), (), ()), owner_codes=np.array([4, 5, 6], dtype=np.int16),
        neuron_ids=np.array([10, 20, 30], dtype=np.int32),
    )
    adam = structural.StructuralAdam(
        np.ones((3, 1)), np.ones((3, 1)), np.ones(1), np.ones(1),
        np.ones(2), np.ones(2), 3,
    )

    pruned, _ = structural.prune_recurrent(topology, [1.0, 2.0], (True,))
    masked = structural.mask_topology(topology, np.array([True, False, True]))
    typed = structural.assign_dale_type(topology, [0], 1)
    connected = structural.add_recurrent_connections(topology, ((2, 0),))
    compacted, _, _ = structural.compact(
        topology, np.array([True, False, True]), adam
    )
    twins, donors = structural.add_twin_neurons(
        topology, [3.0, 2.0, 1.0], required=1
    )

    for candidate in (pruned, masked, typed, connected):
        np.testing.assert_array_equal(candidate.owner_codes, [4, 5, 6])
        np.testing.assert_array_equal(candidate.neuron_ids, [10, 20, 30])
    np.testing.assert_array_equal(compacted.owner_codes, [4, 6])
    np.testing.assert_array_equal(compacted.neuron_ids, [10, 30])
    assert donors == (0,)
    np.testing.assert_array_equal(twins.owner_codes, [4, 5, 6, 4])
    np.testing.assert_array_equal(twins.neuron_ids, [10, 20, 30, 31])


def test_neuron_mask_and_compaction_have_prediction_byte_identity():
    topology = structural.SparseTopology(
        input_source=np.array([0, 1]), input_target=np.array([0, 1]),
        input_value=np.array([1.0, 2.0]), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 0]), recurrent_value=np.array([3.0, 4.0]),
        readout=np.array([[5.0], [6.0]]), dale=np.array([1, -1]),
        mechanisms=(("hh",), ("hh",)),
    )
    adam = structural.StructuralAdam(
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
        np.zeros(2), np.zeros(2), 2,
    )
    alive = np.array([True, False])
    masked = structural.mask_topology(topology, alive)
    compacted, _, _ = structural.compact(topology, alive, adam)
    assert structural.prediction_bytes_identical(
        masked, compacted,
        lambda value: value.readout.sum(axis=0),
        lambda value: value.readout.sum(axis=0),
    )


def test_real_mask_compaction_identity_screens_fixed_tasks(monkeypatch):
    topology = structural.SparseTopology(
        input_source=np.array([0, 1]), input_target=np.array([0, 1]),
        input_value=np.array([1.0, 2.0]), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 0]), recurrent_value=np.array([3.0, 4.0]),
        readout=np.array([[1.0], [2.0]]), dale=np.array([1, -1]),
        mechanisms=((), ()),
    )
    adam = structural.StructuralAdam(
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
        np.zeros(2), np.zeros(2), 0,
    )

    class Model:
        def __init__(self, candidate):
            self.candidate = candidate
            self.readout_weight = SimpleNamespace(value=candidate.readout)
            self.readout_bias = SimpleNamespace(
                value=np.zeros(candidate.readout.shape[1])
            )

        def reset_episode(self, learner):
            pass

        def readout(self):
            return self.candidate.readout

    task = SimpleNamespace(targets=[np.array([[1]], dtype=np.uint8)])
    module = SimpleNamespace(
        TRAINING_TASK_IDS=("train",), VALIDATION_TASK_IDS=("valid",),
        load_task=lambda *_args: task,
        encode_episode=lambda *_args: (np.zeros((1, 1)), np.ones(1, dtype=bool)),
        run_event_sequence=lambda model, *_args: np.zeros(
            (31, model.candidate.neuron_count)
        ),
        decode_prediction=lambda value: np.asarray([value.sum()], dtype=np.uint8),
        strict_task_pass_at_1=lambda predictions, targets: True,
    )
    monkeypatch.setattr(
        structural, "_rebuild_real_candidate",
        lambda _module, candidate, _learner: (Model(candidate), object()),
    )
    result = structural._real_mask_compaction_identity(
        module, topology, adam, "data", transform=_EagerTransform
    )
    assert result["prediction_bytes_identical"]
    assert result["strict_identical"]
    assert result["masked_neurons"] == 2
    assert result["compacted_neurons"] == 1


def test_real_pp_prop_update_builds_episode_driver():
    import jax.numpy as jnp

    class Learner:
        def etrace_evolve(self, events, return_outputs):
            return (events,)

    class Model:
        input_weight = SimpleNamespace(value=jnp.ones(1))
        recurrent_weight = SimpleNamespace(value=jnp.ones(1))

        def reset_episode(self, learner):
            self.reset = learner

    class Trainer:
        def __init__(self, learner, parameters):
            self.learner = learner
            self.parameters = parameters

        def update_episode(self, events, step_fn, loss_mask):
            return events.shape, loss_mask.shape, step_fn(events[0])

    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    update = structural._real_pp_prop_update(
        module, Model(), Learner(), {"events": [[1.0]], "advances": [True]}
    )
    shape, mask_shape, value = update(0)
    assert shape == (1, 1)
    assert mask_shape == (1,)
    assert value.shape == ()


def test_recurrent_pruning_uses_exact_ceiling_and_preserves_other_arrays():
    topology = structural.SparseTopology(
        input_source=np.array([0]), input_target=np.array([0]), input_value=np.array([1.0]),
        recurrent_source=np.arange(21) % 3, recurrent_target=(np.arange(21) + 1) % 3,
        recurrent_value=np.arange(21.0), readout=np.ones((3, 1)),
        dale=np.ones(3), mechanisms=(("hh",),) * 3,
    )
    pruned, keep = structural.prune_recurrent(
        topology, np.arange(21.0), validation_strict=(False, True)
    )
    assert np.flatnonzero(~keep).tolist() == [0, 1]
    assert len(pruned.recurrent_value) == 19
    np.testing.assert_array_equal(pruned.input_value, topology.input_value)
    assert structural.prune_neurons(topology, [3.0, 2.0, 1.0], (True,)).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="(?i)validation"):
        structural.prune_neurons(topology, [1.0, 2.0, 3.0], (False,))
    with pytest.raises(ValueError, match="per neuron"):
        structural.prune_neurons(topology, [1.0], (True,))
    with pytest.raises(ValueError, match="(?i)validation"):
        structural.prune_recurrent(topology, np.arange(21.0), (False,))
    with pytest.raises(ValueError, match="per recurrent"):
        structural.prune_recurrent(topology, np.arange(2.0), (True,))


def test_real_model_snapshot_and_task_mass_helpers_are_sparse():
    class Value:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Csr:
        def __init__(self, indices, indptr):
            self.indices = np.asarray(indices)
            self.indptr = np.asarray(indptr)

    class Model:
        input_csr = Csr([0, 1, 1], [0, 2, 3])
        recurrent_csr = Csr([1, 0], [0, 1, 2])
        input_weight = Value([1.0, 2.0, 3.0])
        recurrent_weight = Value([4.0, 5.0])
        readout_weight = Value([[6.0], [7.0]])

    topology = structural.topology_from_model(Model())
    assert topology.neuron_count == 2
    np.testing.assert_array_equal(topology.input_source, [0, 0, 1])
    np.testing.assert_array_equal(topology.recurrent_source, [0, 1])
    mass = structural.task_gradient_mass({"recurrent": [[-1.0, 2.0], [3.0, -4.0]]}, "recurrent", 2)
    np.testing.assert_array_equal(mass, [[1.0, 2.0], [3.0, 4.0]])
    assert structural.task_gradient_mass({}, "missing", 2).shape == (2, 0)
    with pytest.raises(ValueError, match="task dimension"):
        structural.task_gradient_mass({"recurrent": [[1.0]]}, "recurrent", 2)
    assert structural.resident_tile_pairs(256) == 65_536
    with pytest.raises(ValueError, match="between"):
        structural.resident_tile_pairs(0)

    evidence = structural.structural_evidence(
        topology,
        readout_effect=np.array([[1.0, 2.0], [2.0, 0.0]]),
        spikes=np.array([[1.0, 0.0], [0.5, 2.0]]),
        gradient_mass=np.array([[3.0, 1.0], [2.0, 4.0]]),
    )
    assert evidence["neuron_scores"].shape == (2,)
    assert evidence["connection_scores"].shape == (2,)
    assert len(evidence["owners"]) == 2
    with pytest.raises(ValueError, match="task-by-neuron"):
        structural.structural_evidence(topology, [[1.0]], [[1.0]], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="task-by-edge"):
        structural.structural_evidence(
            topology, [[1.0, 2.0]], [[1.0, 2.0]], [[1.0]]
        )


def test_plot_topology_uses_checkpoint_counts_and_labels_without_mutation(tmp_path):
    input_indptr = np.zeros(442, dtype=np.int32)
    input_indptr[-1] = 3
    recurrent_indptr = np.array([0, 2, 3, 3], dtype=np.int32)
    arrays = {
        "neuron_count": np.asarray(3, dtype=np.int32),
        "input_indptr": input_indptr,
        "input_indices": np.array([0, 1, 2], dtype=np.int32),
        "input_values": np.ones(3, dtype=np.float32),
        "recurrent_indptr": recurrent_indptr,
        "recurrent_indices": np.array([1, 2, 0], dtype=np.int32),
        "recurrent_values": np.ones(3, dtype=np.float32),
        "readout_weight": np.arange(6, dtype=np.float32).reshape(3, 2),
        "dale_codes": np.array([0, 1, -1], dtype=np.int8),
        "owner_codes": np.array([-1, 0, -2], dtype=np.int16),
        "neuron_ids": np.array([2, 0, 1], dtype=np.int32),
    }
    module = SimpleNamespace(load_checkpoint=lambda _path: arrays)
    topology = structural.topology_from_checkpoint(module, "accepted.npz")
    request_logits = np.zeros((31, 360), dtype=np.float32)
    request_logits[0, 1] = 1.0
    request_logits[0, 32] = 1.0
    request_logits[1:, 60:] = np.arange(30, dtype=np.float32)[None, :, None]
    prediction_before = decode_prediction(request_logits).tobytes()
    logits_before = request_logits.tobytes()
    output = tmp_path / "topology.png"
    result = structural.plot_topology(topology, output)
    assert output.exists() and output.stat().st_size > 0
    assert result["neuron_count"] == 3
    assert result["input_connection_count"] == 3
    assert result["recurrent_connection_count"] == 3
    assert result["recurrent_plot_edge_count"] == 3
    assert result["dale_groups"] == [-1, 0, 1]
    assert result["owner_groups"] == [-2, -1, 0]
    assert topology.readout is None
    assert request_logits.tobytes() == logits_before
    assert decode_prediction(request_logits).tobytes() == prediction_before


def test_structural_file_entry_point_imports_before_parsing():
    root = Path(__file__).parents[2]
    script = root / "examples" / "pp_prop" / "example21_structural.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_plot_command_is_explicit_and_ordinary_baseline_does_not_plot(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(structural, "plot_topology", lambda *_args, **_kwargs: calls.append(1))

    class Value:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Csr:
        def __init__(self, indices, indptr):
            self.indices = np.asarray(indices)
            self.indptr = np.asarray(indptr)

    class Model:
        input_csr = Csr([0], [0, 1])
        recurrent_csr = Csr([0], [0, 1])
        input_weight = Value([1.0])
        recurrent_weight = Value([1.0])
        readout_weight = Value([[1.0]])

    monkeypatch.setattr(
        structural, "_load_example21_model",
        lambda: SimpleNamespace(
            BrainCellArcModel=lambda: Model(), VALIDATION_TASK_IDS=("valid",)
        ),
    )
    output = tmp_path / "baseline.json"
    structural.main(["baseline", "--output", str(output)])
    assert calls == []
    monkeypatch.setattr(
        structural, "topology_from_checkpoint", lambda *_args: object()
    )
    structural.main([
        "plot", "--checkpoint", "accepted.npz", "--output", str(tmp_path / "plot.png")
    ])
    assert calls == [1]


def test_plot_topology_rejects_invalid_checkpoint_and_topology_shapes(tmp_path):
    def checkpoint(input_rows, recurrent_rows):
        return {
            "neuron_count": np.asarray(1, dtype=np.int32),
            "input_indptr": np.zeros(input_rows, dtype=np.int32),
            "recurrent_indptr": np.zeros(recurrent_rows, dtype=np.int32),
        }

    with pytest.raises(ValueError, match="441 rows"):
        structural.topology_from_checkpoint(
            SimpleNamespace(load_checkpoint=lambda _path: checkpoint(2, 2)), "bad"
        )
    with pytest.raises(ValueError, match="invalid rows"):
        structural.topology_from_checkpoint(
            SimpleNamespace(load_checkpoint=lambda _path: checkpoint(442, 3)), "bad"
        )
    with pytest.raises(ValueError, match="counts must match"):
        structural.plot_topology(structural.SparseTopology(
            np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
            np.array([0], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
            np.ones((1, 1)), np.zeros(1, dtype=np.int8), ((),),
        ), tmp_path / "bad.png")
    with pytest.raises(ValueError, match="Dale labels"):
        structural.plot_topology(SimpleNamespace(
            neuron_count=2,
            recurrent_source=np.array([], dtype=np.int32),
            recurrent_target=np.array([], dtype=np.int32),
            input_value=np.array([], dtype=np.float32),
            recurrent_value=np.array([], dtype=np.float32),
            dale=np.zeros(1, dtype=np.int8),
        ), tmp_path / "bad.png")
    with pytest.raises(ValueError, match="one neuron"):
        structural.plot_topology(structural.SparseTopology(
            np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
            np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
            np.ones((0, 1)), np.zeros(0, dtype=np.int8), (),
        ), tmp_path / "bad.png")
    topology = structural.SparseTopology(
        np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
        np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
        np.ones((1, 1)), np.zeros(1, dtype=np.int8), ((),),
        owner_codes=np.zeros(2, dtype=np.int16),
    )
    with pytest.raises(ValueError, match="identifiers"):
        structural.plot_topology(
            topology.__class__(
                topology.input_source, topology.input_target, topology.input_value,
                topology.recurrent_source, topology.recurrent_target,
                topology.recurrent_value, topology.readout, topology.dale,
                topology.mechanisms, topology.owner_codes, np.zeros(2, dtype=np.int32),
            ), tmp_path / "bad.png"
        )
    with pytest.raises(ValueError, match="(?i)owner labels"):
        structural.plot_topology(topology, tmp_path / "bad.png")


def test_plot_topology_defaults_missing_labels_and_identifiers(tmp_path):
    topology = structural.SparseTopology(
        np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
        np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32),
        np.ones((1, 1)), np.zeros(1, dtype=np.int8), ((),),
    )
    result = structural.plot_topology(topology, tmp_path / "default.png")
    assert result["owner_groups"] == [-1]


def test_truth_documents_separate_observations_and_implementation_boundary():
    root = Path(__file__).parents[2] / "docs" / "specs"
    causal = (root / "2026-08-24-example21-causal-explanation.md").read_text()
    system = (root / "2026-08-24-example21-system-model.md").read_text()
    assert "## Observations" in causal
    assert "## Inferences" in causal
    assert "no claim of real ARC accuracy" in causal
    for term in (
        "BrainCell", "BrainState", "BrainTrace", "neuron", "connection",
        "layer", "Dale-type", "model-cell", "prediction", "output-shape",
    ):
        assert term in system
    assert "## Implementation boundary" in system
    assert "MiniLSTM" not in system


def test_twin_addition_is_connected_splits_values_and_zeros_new_moments():
    topology = structural.SparseTopology(
        input_source=np.array([0]), input_target=np.array([0]), input_value=np.array([2.0]),
        recurrent_source=np.array([0]), recurrent_target=np.array([1]), recurrent_value=np.array([4.0]),
        readout=np.array([[6.0], [2.0]]), dale=np.array([1, 0]),
        mechanisms=(('hh',), ('hh',)),
    )
    grown, donors = structural.add_twin_neurons(
        topology, np.array([[3.0, 1.0]]), required=1
    )
    assert donors == (0,) and grown.neuron_count == 3
    assert (0, 2.0) in list(zip(grown.input_target, grown.input_value))
    np.testing.assert_array_equal(grown.readout[[0, 2]], [[3.0], [3.0]])
    assert (2, 1) in list(zip(grown.recurrent_source, grown.recurrent_target))
    np.testing.assert_array_equal(
        grown.recurrent_value[np.isin(grown.recurrent_source, [0, 2])], [2.0, 2.0]
    )
    adam = structural.StructuralAdam(
        np.ones((2, 1)), np.ones((2, 1)), np.ones(1), np.ones(1),
        np.ones(1), np.ones(1), 9,
    )
    mapped = structural.grow_adam_for_twins(adam, topology, grown)
    assert mapped.step == 9
    assert mapped.neuron_first[-1, 0] == 0
    assert np.all(mapped.recurrent_first[1:] == 0)
    with pytest.raises(ValueError, match="budget"):
        structural.add_twin_neurons(topology, [1.0, 0.0], required=0)
    with pytest.raises(ValueError, match="connected"):
        structural.add_twin_neurons(topology, [2.0, 1.0], required=2)


def test_neuron_addition_rejects_connection_ceiling_before_candidate_build(monkeypatch):
    topology = structural.SparseTopology(
        input_source=np.zeros(1024, dtype=int),
        input_target=np.zeros(1024, dtype=int),
        input_value=np.ones(1024),
        recurrent_source=np.zeros(1024, dtype=int),
        recurrent_target=np.ones(1024, dtype=int),
        recurrent_value=np.ones(1024),
        readout=np.ones((2, 1)), dale=np.zeros(2), mechanisms=((), ()),
    )
    monkeypatch.setattr(
        np, "vstack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate arrays were constructed before the ceiling check")
        ),
    )
    with pytest.raises(ValueError, match="biological-connection ceiling"):
        structural.add_twin_neurons(topology, [1.0, 0.0], required=1)


def test_twin_connectivity_guard_handles_numpy_edge_arrays():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([2, 3]), np.ones(2),
        np.ones((4, 1)), np.zeros(4), ((),) * 4,
    )
    grown, donors = structural.add_twin_neurons(topology, np.arange(4.0), required=2)
    assert donors == (3, 2)
    assert grown.neuron_count == 6


def test_tiled_connection_addition_is_global_stable_and_never_dense(monkeypatch):
    original_empty = np.empty
    def guarded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (300, 300):
            raise AssertionError("dense pair allocation")
        return original_empty(shape, *args, **kwargs)
    monkeypatch.setattr(np, "empty", guarded_empty)
    selected = structural.select_connection_additions(
        300, existing={(0, 1)}, source_evidence=np.arange(300.0, 0.0, -1.0),
        target_evidence=np.arange(300.0, 0.0, -1.0), required=3, tile_size=256,
    )
    assert selected == ((1, 0), (0, 2), (2, 0))
    assert structural.resident_tile_pairs(256) == 65536
    with pytest.raises(ValueError, match="256"):
        structural.resident_tile_pairs(257)


def test_connection_addition_appends_exact_items_and_zero_moments():
    topology = structural.SparseTopology(
        input_source=np.array([], dtype=int), input_target=np.array([], dtype=int),
        input_value=np.array([]), recurrent_source=np.array([0]),
        recurrent_target=np.array([1]), recurrent_value=np.array([2.0]),
        readout=np.ones((3, 1)), dale=np.zeros(3), mechanisms=((), (), ()),
    )
    grown = structural.add_recurrent_connections(topology, ((1, 2), (2, 0)))
    assert list(zip(grown.recurrent_source, grown.recurrent_target))[-2:] == [(1, 2), (2, 0)]
    np.testing.assert_array_equal(grown.recurrent_value[-2:], [0.0, 0.0])
    adam = structural.StructuralAdam(
        np.ones((3, 1)), np.ones((3, 1)), np.array([]), np.array([]),
        np.ones(1), np.ones(1), 4,
    )
    mapped = structural.grow_adam_for_connections(adam, 2)
    np.testing.assert_array_equal(mapped.recurrent_first, [1.0, 0.0, 0.0])
    typed = structural.add_recurrent_connections(topology, ((2, 1),), typed=True)
    np.testing.assert_array_equal(typed.recurrent_value[-1:], [0.0])
    with pytest.raises(ValueError, match="distinct"):
        structural.add_recurrent_connections(topology, ((0, 1),))
    with pytest.raises(ValueError, match="65,536"):
        structural.select_connection_additions(2, set(), [1, 1], [1, 1], 1, 257)


def test_connection_addition_rejects_connection_ceiling_before_concatenation(
    monkeypatch,
):
    topology = structural.SparseTopology(
        input_source=np.zeros(1024, dtype=int),
        input_target=np.zeros(1024, dtype=int),
        input_value=np.ones(1024),
        recurrent_source=np.zeros(1024, dtype=int),
        recurrent_target=np.zeros(1024, dtype=int),
        recurrent_value=np.ones(1024),
        readout=np.ones((2, 1)), dale=np.zeros(2), mechanisms=((), ()),
    )
    monkeypatch.setattr(
        np, "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate arrays were constructed before the ceiling check")
        ),
    )
    with pytest.raises(ValueError, match="biological-connection ceiling"):
        structural.add_recurrent_connections(topology, ((0, 1),))


def test_dale_assignment_and_structural_operations_preserve_effective_signs():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([1, 2]), np.array([-2.0, -4.0]),
        np.ones((3, 1)), np.zeros(3, dtype=np.int8), ((),) * 3,
    )
    typed = structural.assign_dale_type(topology, [0], 1)
    np.testing.assert_array_equal(typed.dale, [1, 0, 0])
    assert structural.effective_topology_recurrent_values(typed)[0] > 0
    grown = structural.add_recurrent_connections(typed, ((0, 2),), typed=True)
    assert structural.validate_topology_dale(grown)
    assert structural.effective_topology_recurrent_values(grown)[-1] > 0
    compacted, _, _ = structural.compact(
        grown, np.array([True, False, True]), structural.StructuralAdam(
            np.ones((3, 1)), np.ones((3, 1)), np.array([]), np.array([]),
            np.ones(3), np.ones(3), 1,
        )
    )
    assert compacted.dale.tolist() == [1, 0]
    assert structural.validate_topology_dale(compacted)


def test_inhibitory_addition_uses_source_label_and_zero_new_moments():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0]), np.array([1]), np.array([2.0]), np.ones((3, 1)),
        np.array([-1, 0, 0], dtype=np.int8), ((),) * 3,
    )
    grown = structural.add_recurrent_connections(topology, ((0, 2),), typed=True)
    assert structural.effective_topology_recurrent_values(grown)[-1] < 0
    adam = structural.StructuralAdam(
        np.ones((2, 1)), np.ones((2, 1)), np.array([]), np.array([]),
        np.ones(1), np.ones(1), 4,
    )
    mapped = structural.grow_adam_for_connections(adam, 1)
    assert mapped.recurrent_first.tolist() == [1.0, 0.0]


def test_recurrent_addition_ignores_caller_type_flag_and_uses_source_label():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0]), np.array([1]), np.array([2.0]), np.ones((3, 1)),
        np.array([1, 0, 0], dtype=np.int8), ((),) * 3,
    )
    grown = structural.add_recurrent_connections(topology, ((0, 2),), typed=False)
    assert structural.effective_topology_recurrent_values(grown)[-1] == pytest.approx(1e-6)
    untyped = structural.add_recurrent_connections(
        topology, ((1, 2),), typed=True
    )
    assert untyped.recurrent_value[-1] == 0.0
    with pytest.raises(ValueError, match="source Dale"):
        structural.add_recurrent_connections(
            topology, ((0, 2),), source_dale=np.array([-1])
        )


def test_production_dale_wrapper_supplies_a_serialized_checkpoint(monkeypatch):
    captured = {}

    def runner(parent, measurements, build, update, strict, **kwargs):
        captured.update({"parent": parent, "measurements": measurements, **kwargs})
        return {"arms": ()}

    monkeypatch.setattr(structural, "_run_dale_candidates", runner)
    parent = SimpleNamespace(parent_id="accepted-parent")
    measurements = SimpleNamespace(parent_id="accepted-parent")
    result = structural.run_dale_candidate_arms(
        parent, measurements, object(), object(), object()
    )
    assert result == {"arms": ()}
    assert pickle.loads(captured["checkpoint"]).parent_id == parent.parent_id


def test_dale_arm_is_exposed_by_the_production_structural_route(monkeypatch):
    module = object()
    parent = object()
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "load_parent_checkpoint", lambda *_: parent)
    monkeypatch.setattr(
        structural,
        "_measure_real_dale",
        lambda received_module, received_parent, data_root, checkpoint_output, clock: {
            "module": received_module, "parent": received_parent,
            "data_root": data_root, "checkpoint_output": checkpoint_output,
            "clock": clock,
        },
    )
    def clock():
        return 1.0

    result = structural.measure_real_arm(
        "dale", data_root="arc", parent_checkpoint="accepted",
        checkpoint_output="children.npz", clock=clock
    )
    assert result == {
        "module": module, "parent": parent, "data_root": "arc", "clock": clock,
        "checkpoint_output": "children.npz",
    }


def test_real_dale_measurement_builds_checkpoint_arms_and_records_evidence(monkeypatch):
    import jax.numpy as jnp

    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    topology = structural.SparseTopology(
        np.array([0]), np.array([0]), np.array([1.0]),
        np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([2.0, -3.0, 1.0]),
        np.ones((3, 1)), np.zeros(3, dtype=np.int8), ((),) * 3,
    )

    class Model:
        def __init__(self, source):
            self.input_csr = type("CSR", (), {
                "indptr": np.array([0, 1]), "indices": np.array([0]),
            })()
            self.recurrent_csr = type("CSR", (), {
                "indptr": np.array([0, 1, 2, 3]),
                "indices": np.asarray(source.recurrent_target),
            })()
            self.input_weight = State(source.input_value)
            self.recurrent_weight = State(source.recurrent_value)
            self.readout_weight = State(source.readout)
            self.readout_bias = State(np.zeros(1))
            self.dale = np.asarray(source.dale)
            self.mechanisms = ((),) * 3
            self.biology_options = structural.deferred_biology_defaults()

        def reset_episode(self, _learner=None):
            return None

    parent = structural.ParentCheckpoint(
        topology, structural.StructuralAdam(
            np.ones((3, 1)), np.ones((3, 1)), np.ones(1), np.ones(1),
            np.ones(3), np.ones(3), 1,
        ), np.zeros(1), "accepted-digest", True,
    )
    module = type(
        "Module", (), {
            "BrainCellArcModel": Model,
            "compile_pp_prop_model": staticmethod(lambda model: object()),
            "run_event_sequence": staticmethod(
                lambda _model, events, _advances, block_source=None: jnp.ones(
                    (events.shape[0], 3)
                ) * (1.0 if block_source is None else 1.0 + block_source * 0.1),
            ),
        },
    )
    monkeypatch.setattr(
        structural, "_fixed_task_evidence", lambda *_args: {
            "preclip_gradient_mass": [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            "task_spike_evidence": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],
            "owners": ((), (0,), (1,)),
            "task_readout_evidence": [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            "strict": (False,),
            "training_events": np.zeros((2, 3, 441), dtype=np.float32),
            "training_advances": np.ones((2, 3), dtype=bool),
        },
    )
    monkeypatch.setattr(
        structural, "_rebuild_real_candidate",
        lambda _module, candidate, _learner: (Model(candidate), object()),
    )
    monkeypatch.setattr(
        structural, "_real_pp_prop_update",
        lambda *_args, **_kwargs: lambda _index: None,
    )
    monkeypatch.setattr(structural, "_fixed_strict_screen", lambda *_args, **_kwargs: (True,))

    def run(parent_state, measurements, build, update, strict, **kwargs):
        assert measurements.parent_id == parent_state.parent_id == "accepted-digest"
        candidate = build(parent_state, np.array([0]), 1)
        update(candidate, 0)
        assert strict(candidate) == (True,)
        selection = type("Selection", (), {
            "parent_id": measurements.parent_id,
            "excitatory": np.array([0]), "inhibitory": np.array([1]),
            "excitatory_scores": np.array([1.0]),
            "inhibitory_scores": np.array([0.5]),
        })()
        return {
            "selection": selection,
            "arms": [{
                "candidate": candidate, "sign": 1, "updates": 128,
                "before_strict": [False], "after_strict": [True],
                "promoted": True,
            }, {
                "candidate": candidate, "sign": -1, "updates": 128,
                "before_strict": [False], "after_strict": [True],
                "promoted": True,
            }],
            "parent_checkpoint_unchanged": True,
        }

    monkeypatch.setattr(structural, "run_dale_candidate_arms", run)
    result = structural._measure_real_dale(module, parent, "arc")
    assert result["candidate_selection"]["parent_id"] == "accepted-digest"
    assert result["parent_checkpoint_unchanged"]
    assert all(arm["typed_signs_valid"] for arm in result["arms"])
    assert all(not any(result["deferred_biology"].values()) for _ in result["arms"])
    monkeypatch.setattr(
        structural, "_fixed_task_evidence",
        lambda *_args: {
            "preclip_gradient_mass": [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            "task_spike_evidence": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],
            "owners": ((), (0,), (1,)),
            "task_readout_evidence": [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            "strict": (False,),
        },
    )
    fallback = structural._measure_real_dale(module, parent, "arc")
    assert fallback["lesion_evidence"]


def test_promoted_dale_child_checkpoint_is_distinct_and_array_backed(
    monkeypatch, tmp_path
):
    topology = structural.SparseTopology(
        np.array([0]), np.array([0]), np.array([1.0]),
        np.array([0]), np.array([1]),
        np.asarray(structural.encode_dale_weights([0.25], [1])),
        np.ones((2, 1)), np.array([1, 0], dtype=np.int8), ((), ()),
    )
    candidate = SimpleNamespace(
        model=SimpleNamespace(), update=SimpleNamespace(trainer=object())
    )
    optimizer = structural.StructuralAdam(
        np.ones((2, 1)), np.ones((2, 1)), np.ones(1), np.ones(1),
        np.ones(1), np.ones(1), bias_first=np.ones(1), bias_second=np.ones(1),
        input_step=1, recurrent_step=1, readout_step=1,
    )
    arrays = {"recurrent_values": np.array([0.25], dtype=np.float32)}
    monkeypatch.setattr(structural, "topology_from_model", lambda _model: topology)
    monkeypatch.setattr(
        structural, "optimizer_from_muon_groups", lambda _trainer: optimizer
    )
    monkeypatch.setattr(
        structural, "checkpoint_arrays", lambda *_args: arrays
    )

    def write_checkpoint(path, _arrays):
        Path(path).write_bytes(b"child-checkpoint")

    module = SimpleNamespace(
        write_checkpoint=write_checkpoint,
        load_checkpoint=lambda _path: arrays,
    )
    parent_path = tmp_path / "accepted.npz"
    parent_path.write_bytes(b"parent")
    child_path, digest = structural._write_dale_child_checkpoint(
        module, candidate, {}, parent_path, 1
    )
    assert child_path == tmp_path / "accepted-dale-excitatory.npz"
    assert child_path != parent_path
    assert child_path.read_bytes() == b"child-checkpoint"
    assert digest == hashlib.sha256(b"child-checkpoint").hexdigest()


def test_default_example21_construction_keeps_deferred_biology_inactive():
    module = structural._load_example21_model()
    model = module.BrainCellArcModel()
    assert model.biology_options == {
        name: False for name in module.deferred_biology_defaults()
    }
    assert not any(model.mechanisms)


def test_preclip_mass_is_taken_directly_from_real_etrace_boundary():
    class Learner:
        def etrace_grad(self, events, **kwargs):
            assert kwargs["return_value"]
            return {("model", "recurrent_weight"): np.array([2.0, -3.0])}, 7.0

    mass, loss = structural.preclip_gradient_mass(
        Learner(), np.ones((1, 2)), lambda value: value, 1, 3,
        reduction="sum",
    )
    np.testing.assert_array_equal(
        mass["model/recurrent_weight"], [[0.0, 0.0], [2.0, 3.0], [0.0, 0.0]]
    )
    assert loss == 7.0


def test_collect_model_evidence_keeps_preclip_mass_and_task_scores():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = type("CSR", (), {"indptr": np.array([0, 1]), "indices": np.array([0])})()
        recurrent_csr = type("CSR", (), {
            "indptr": np.array([0, 1, 2]), "indices": np.array([1, 0])
        })()
        input_weight = State([1.0])
        recurrent_weight = State([2.0, 3.0])
        readout_weight = State(np.ones((2, 1)))

    class Learner:
        def etrace_grad(self, events, **kwargs):
            return {("model", "recurrent_weight"): np.array([2.0, -3.0])}, np.array([4.0])

    result = structural.collect_model_evidence(
        Model(), Learner(), np.ones((1, 1)), lambda value: value,
        np.ones((2, 2)), np.ones((2, 2)), reduction="sum",
    )
    assert result["model_neurons"] == 2
    assert result["preclip_exceeds_clip"]
    np.testing.assert_array_equal(result["gradient_mass"], [[2.0, 3.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="recurrent weight"):
        structural.collect_model_evidence(
            Model(), type("L", (), {"etrace_grad": lambda *a, **k: ({}, 0)})(),
            np.ones((1, 1)), lambda value: value, np.ones((2, 2)), np.ones((2, 2)),
        )


def test_addition_driver_and_one_arm_execution_use_compiled_128_updates():
    class Transform:
        calls: ClassVar[list] = []

        @staticmethod
        def for_loop(function, values):
            Transform.calls.append(("for_loop", len(values)))
            return np.asarray([function(value) for value in values])

        @staticmethod
        def jit(function):
            def compiled(values):
                Transform.calls.append(("jit", len(values)))
                return function(values)
            return compiled

    ticks = []
    evidence = structural.execute_one_arm(
        "connection-add", (False, True), lambda: (object(), 2),
        lambda candidate: (True, True), updates=128, transform=Transform,
        update=lambda index: ticks.append(int(index)) or index,
        clock=iter((10.0, 12.5)).__next__,
    )
    assert ticks == list(range(128))
    assert Transform.calls == [("jit", 128), ("for_loop", 128)]
    assert evidence["promoted"] and evidence["mutated_item_count"] == 2
    assert evidence["elapsed_seconds"] == 2.5
    with pytest.raises(ValueError, match="128"):
        structural.run_addition_updates(Transform, lambda value: value, updates=63)
    with pytest.raises(ValueError, match="recognized"):
        structural.execute_one_arm("two-arms", (), lambda: (None, 0), lambda _: ())
    with pytest.raises(ValueError, match="compiled"):
        structural.execute_one_arm(
            "neuron-add", (False,), lambda: (None, 1), lambda _: (True,), updates=128
        )
    pruning = structural.execute_one_arm(
        "neuron-prune", (True,), lambda: (None, 1), lambda _: (True,),
        clock=iter((0.0, 1.0)).__next__,
    )
    assert not pruning["promoted"] and pruning["updates"] == 0


def test_complete_process_timing_controls_bounded_result_and_promotion():
    evidence = {
        "arm": "neuron-add",
        "updates": 128,
        "before_strict": [False, True],
        "after_strict": [True, True],
        "within_300_seconds": True,
        "promoted": True,
    }
    structural.apply_complete_process_timing(evidence, 300.25)
    assert evidence["complete_process_seconds"] == 300.25
    assert not evidence["within_300_seconds"]
    assert not evidence["promoted"]


def test_artifact_is_canonical_and_records_environment(tmp_path):
    target = tmp_path / "arm.json"
    digest = structural.write_artifact(target, {"arm": "neuron-prune"})
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    document = json.loads(target.read_text())
    assert document["environment"]["seeds"] == [21, 22, 23]
    assert document["arm"] == "neuron-prune"


def test_arm_gate_requires_gain_no_regression_limits_and_fixed_updates():
    assert structural.promote_arm((False, True), (True, True), 299.0, "addition", 128)
    assert not structural.promote_arm((False, True), (True, False), 1.0, "addition", 128)
    assert not structural.promote_arm((False,), (True,), 301.0, "addition", 128)
    assert not structural.promote_arm((True,), (True,), 1.0, "pruning", 0)
    with pytest.raises(ValueError, match="128"):
        structural.promote_arm((False,), (True,), 1.0, "addition", 63)


def test_integrated_arm_rebuilds_model_remaps_adam_and_resets_eligibility():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        def __init__(self):
            self.input_csr = type("CSR", (), {
                "indptr": np.array([0, 1, 2]),
                "indices": np.array([0, 1]),
            })()
            self.recurrent_csr = type("CSR", (), {
                "indptr": np.array([0, 1, 2, 3]),
                "indices": np.array([1, 2, 0]),
            })()
            self.input_weight = State([1.0, 1.0])
            self.recurrent_weight = State([1.0, 1.0, 1.0])
            self.readout_weight = State(np.ones((3, 1)))
            self.reset_count = 0

        def reset_episode(self, learner):
            self.reset_count += 1

    def rebuild(topology, adam):
        model = Model()
        model.input_csr.indices = topology.input_target
        model.recurrent_csr.indices = topology.recurrent_target
        model.input_weight.value = topology.input_value
        model.recurrent_weight.value = topology.recurrent_value
        model.readout_weight.value = topology.readout
        return model, object()

    result = structural.run_integrated_arm(
        "connection-prune", Model, lambda model: object(),
        lambda model, learner: (False, True), rebuild,
        before_strict=(False, True),
        evidence={
            "neuron_scores": np.array([1.0, 1.0, 1.0]),
            "connection_scores": np.array([0.1, 0.2, 0.3]),
            "validation_strict": (False, True),
        },
        clock=iter((0.0, 1.0)).__next__,
    )
    assert result["real_model"]
    assert result["adam_remapped"]
    assert result["eligibility_reset"]
    assert result["mutated_item_count"] == 1
    assert not result["promoted"]


def test_real_pp_prop_update_seeds_remapped_adam_state():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.adam_groups = {
                "readout": type("Adam", (), {"first": np.zeros(1), "second": np.zeros(1), "step": 0})(),
                "input": type("Adam", (), {"first": np.zeros(2), "second": np.zeros(2), "step": 0})(),
                "recurrent": type("Adam", (), {"first": np.zeros(3), "second": np.zeros(3), "step": 0})(),
            }

        def update_episode(self, *args, **kwargs):
            return 0.0, 0.0

    model = SimpleNamespace(input_weight=State([1, 1]), recurrent_weight=State([1, 1, 1]))
    model.reset_episode = lambda learner: None
    learner = SimpleNamespace()
    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    adam = structural.StructuralAdam(
        np.array([1.0]), np.array([2.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0]),
        np.array([6.0, 7.0, 8.0]), np.array([9.0, 10.0, 11.0]), step=4,
    )
    update = structural._real_pp_prop_update(
        module, model, learner,
        {"events": np.zeros((1, 2)), "advances": np.ones(1)}, adam,
    )
    update()
    trainer = update.trainer
    np.testing.assert_array_equal(trainer.adam_groups["readout"].first, [1])
    np.testing.assert_array_equal(trainer.adam_groups["readout"].second, [2])
    assert trainer.adam_groups["readout"].step == 4
    np.testing.assert_array_equal(trainer.adam_groups["input"].first, [2, 3])
    np.testing.assert_array_equal(trainer.adam_groups["recurrent"].second, [9, 10, 11])
    assert trainer.adam_groups["recurrent"].step == 4


def test_real_pp_prop_update_then_structural_addition_preserves_dale_signs():
    import jax.numpy as jnp

    module = structural._load_example21_model()
    topology = structural.SparseTopology(
        np.array([0]), np.array([0]), np.array([1.0]),
        np.array([0, 1]), np.array([1, 2]), np.asarray(structural.encode_dale_weights(
            jnp.array([0.25, -0.5]), jnp.array([1, -1])
        )),
        np.ones((3, 360)), np.array([1, -1, 0], dtype=np.int8), ((),) * 3,
    )
    model = module.BrainCellArcModel(topology)
    learner = module.compile_pp_prop_model(model)
    before = np.asarray(model.recurrent_weight.value)
    evidence = {
        "events": np.zeros((3, 441), dtype=np.float32),
        "advances": np.ones(3, dtype=bool),
    }
    evidence["events"][0, 0] = 1.0
    update = structural._real_pp_prop_update(
        module, model, learner, evidence,
    )
    update()
    assert not np.array_equal(model.recurrent_weight.value, before)
    updated = structural.topology_from_model(model)
    assert structural.validate_topology_dale(updated)
    grown = structural.add_recurrent_connections(updated, ((0, 2), (1, 0)))
    effective = structural.effective_topology_recurrent_values(grown)
    np.testing.assert_allclose(effective[-2:], [1e-6, -1e-6], atol=1e-10)
    assert structural.validate_topology_dale(grown)
    optimizer = structural.optimizer_from_muon_groups(update.trainer)
    mapped = structural.grow_adam_for_connections(optimizer, 2)
    np.testing.assert_array_equal(mapped.recurrent_first[-2:], [0.0, 0.0])
    np.testing.assert_array_equal(mapped.recurrent_second[-2:], [0.0, 0.0])


def test_muon_groups_are_remapped_for_structural_candidate():
    state = SimpleNamespace(
        mu=np.arange(6, dtype=float).reshape(3, 2),
        count=np.array(7, dtype=np.int32),
    )
    mapped = structural.remap_muon_groups(
        {"readout_weight": state},
        {"readout_weight": (np.array([True, False, True]), (2, 2))},
    )
    np.testing.assert_array_equal(mapped["readout_weight"].mu, [[0, 1], [4, 5]])
    assert mapped["readout_weight"].count == 7


def test_real_update_hands_remapped_muon_groups_to_trainer():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.muon_groups = {}

        def update_episode(self, *args, **kwargs):
            return 0.0, 0.0

    model = SimpleNamespace(
        input_weight=State([1, 1]), recurrent_weight=State([1, 1, 1])
    )
    model.reset_episode = lambda learner: None
    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    state = SimpleNamespace(mu=np.arange(6, dtype=float).reshape(3, 2))
    update = structural._real_pp_prop_update(
        module, model, SimpleNamespace(),
        {"events": np.zeros((1, 2)), "advances": np.ones(1)},
        muon_groups={"readout_weight": state},
        parameter_maps={"readout_weight": (np.array([True, False, True]), (2, 2))},
    )
    np.testing.assert_array_equal(update.trainer.muon_groups["readout_weight"].mu,
                                  [[0, 1], [4, 5]])


def test_real_mask_compaction_identity_reuses_pruning_episode_snapshot(monkeypatch):
    episodes = []

    class Module:
        TRAINING_TASK_IDS = ("train",)
        VALIDATION_TASK_IDS = ("valid",)

        def load_task(self, root, task_id, split):
            return SimpleNamespace(targets=(task_id,))

        def encode_episode(self, task, query_index):
            episodes.append((task.targets[0], query_index))
            return np.ones((1, 1)), np.ones(1, dtype=bool)

        def strict_task_pass_at_1(self, predictions, targets):
            return True

        def decode_prediction(self, value):
            return np.asarray(value)

        def run_event_sequence(self, model, events, advances):
            return np.zeros((31, 2))

    class Model:
        def __init__(self):
            self.readout_weight = SimpleNamespace(value=np.ones((2, 1)))
            self.readout_bias = SimpleNamespace(value=np.zeros(1))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.ones(1)

    module = Module()
    topology = SimpleNamespace(
        neuron_count=2, input_source=np.array([0, 0]), input_target=np.array([0, 1]),
        input_value=np.ones(2), recurrent_source=np.array([0]),
        recurrent_target=np.array([1]), recurrent_value=np.ones(1),
        readout=np.ones((2, 1)), dale=np.ones(2), mechanisms=((), ()),
    )
    monkeypatch.setattr(structural, "_rebuild_real_candidate", lambda *args: (Model(), object()))
    structural._real_mask_compaction_identity(
        module, topology, structural.StructuralAdam(
            np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
            np.zeros(1), np.zeros(1),
        ), "data", alive=np.array([True, False]), transform=_EagerTransform,
    )
    assert episodes == [("train", 0), ("valid", 0)]


def test_real_arm_runner_covers_all_bounded_paths(monkeypatch, tmp_path):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1, 2, 3, 4]), indices=np.arange(4))
        recurrent_csr = SimpleNamespace(indptr=np.arange(5), indices=np.array([1, 2, 3, 0]))
        input_weight = State(np.ones(4))
        recurrent_weight = State(np.ones(4))
        readout_weight = State(np.ones((4, 1)))

    fake_module = SimpleNamespace(
        BrainCellArcModel=Model, TRAINING_TASK_IDS=("a",), VALIDATION_TASK_IDS=("b",)
    )
    fake_evidence = {
        "strict": [False, True], "neuron_scores": np.ones(4),
        "connection_scores": np.ones(4), "gradient_mass": np.ones(4),
        "preclip_gradient_mass": [], "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: fake_module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: fake_evidence)
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args: lambda index: index)
    monkeypatch.setattr(structural, "run_addition_updates", lambda *args, **kwargs: None)
    for arm in ("neuron-prune", "connection-prune", "neuron-add", "connection-add"):
        result = structural.measure_real_arm(arm)
        assert result["real_model"] and result["updates"] == (128 if arm.endswith("add") else 0)
    structural.main(["baseline", "--output", str(tmp_path / "baseline.json")])


def test_fixed_task_evidence_decodes_model_readout_not_neuron_voltage(monkeypatch):
    class Model:
        readout_weight = SimpleNamespace(value=np.ones((2, 360)))
        readout_bias = SimpleNamespace(value=np.zeros(360))
        previous_spikes = SimpleNamespace(value=np.array([0.0, 1.0]))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.zeros(360)

    task = SimpleNamespace(targets=(np.zeros((1, 1), dtype=np.uint8),))
    def run_event_sequence(_model, _events, _advances, *, return_spikes=False):
        voltages = np.full((31, 2), 10.0)
        direct_spikes = np.tile(np.array([0.0, 1.0]), (31, 1))
        return (voltages, direct_spikes) if return_spikes else voltages

    module = SimpleNamespace(
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
        load_task=lambda *args: task,
        encode_episode=lambda *args: (np.ones((31, 441)), np.ones(31, dtype=bool)),
        run_event_sequence=run_event_sequence,
        decode_prediction=lambda value: (
            np.zeros((1, 1), dtype=np.uint8)
            if np.asarray(value).shape == (31, 360)
            else (_ for _ in ()).throw(AssertionError("decoded neuron voltage"))
        ),
        strict_task_pass_at_1=lambda predictions, targets: True,
    )
    learner = SimpleNamespace(
        etrace_grad=lambda *args, **kwargs: ({
            ("model", "input_weight"): np.ones(2),
            ("model", "recurrent_weight"): np.ones(2),
        }, 0.0),
        etrace_evolve=lambda event, return_outputs: (event,),
    )
    monkeypatch.setattr(
        structural,
        "topology_from_model",
        lambda model: SimpleNamespace(neuron_count=2),
    )
    monkeypatch.setattr(
        structural,
        "structural_evidence",
        lambda *args: {
            "neuron_scores": np.ones(2),
            "connection_scores": np.ones(2),
            "target_incident_gradient": np.zeros((1, 2)),
        },
    )
    evidence = structural._fixed_task_evidence(
        module, Model(), learner, "data", transform=_EagerTransform
    )
    assert evidence["strict"] == [True, True]
    np.testing.assert_array_equal(evidence["task_spike_evidence"], [[0.0, 1.0]])


def test_real_model_uses_canonical_validation_task_ids():
    module = structural._load_example21_model()
    assert module.VALIDATION_TASK_IDS == (
        "46f33fce", "3428a4f5", "d8c310e9", "09629e4f"
    )


def test_real_pruning_arm_records_closed_validation_gate(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.arange(2))
        recurrent_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.array([1, 0]))
        input_weight = State(np.ones(2))
        recurrent_weight = State(np.ones(2))
        readout_weight = State(np.ones((2, 1)))

    module = SimpleNamespace(
        BrainCellArcModel=Model,
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
    )
    evidence = {
        "strict": [False, False],
        "neuron_scores": np.ones(2),
        "connection_scores": np.ones(2),
        "gradient_mass": np.ones(2),
        "preclip_gradient_mass": [],
        "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: evidence)
    result = structural.measure_real_arm("neuron-prune")
    assert result["pruning_blocked"]
    assert result["mutated_item_count"] == 0
    assert result["candidate_neurons"] == result["baseline_neurons"]


def test_real_arm_evaluates_rebuilt_candidate_model_after_mutation(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        def __init__(self, topology=None):
            self.is_candidate = topology is not None
            self.input_csr = SimpleNamespace(
                indptr=np.array([0, 1, 2]), indices=np.array([0, 1])
            )
            self.recurrent_csr = SimpleNamespace(
                indptr=np.array([0, 1, 2]), indices=np.array([1, 0])
            )
            self.input_weight = State(np.ones(2))
            self.recurrent_weight = State(np.ones(2))
            self.readout_weight = State(np.ones((2, 1)))

        def reset_episode(self, learner):
            pass

    module = SimpleNamespace(
        BrainCellArcModel=Model,
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
        compile_pp_prop_model=lambda model: object(),
    )
    evidence = {
        "strict": [False], "neuron_scores": np.array([1.0, 0.0]),
        "connection_scores": np.ones(2), "gradient_mass": np.ones(2),
        "preclip_gradient_mass": [], "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(
        structural, "_fixed_task_evidence",
        lambda _module, model, _learner, _root: {
            **evidence, "strict": [True] if model.is_candidate else [False]
        },
    )
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args: lambda _: None)
    monkeypatch.setattr(structural, "run_addition_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(structural, "_fixed_strict_screen", lambda *args, **kwargs: (True,))
    monkeypatch.setattr(
        structural,
        "_real_mask_compaction_identity",
        lambda *args, **kwargs: {
            "prediction_bytes_identical": True,
            "strict_identical": True,
            "masked_strict": [False],
            "compacted_strict": [False],
        },
        raising=False,
    )
    result = structural.measure_real_arm("neuron-add", data_root="data")
    assert result["candidate_neurons"] == 3
    assert result["after_strict"] == [True]
    assert result["pruning_blocked"] is False
    assert result["mask_compaction"]["not_measured"] is True


def test_neuron_evidence_includes_input_and_recurrent_incident_gradient_mass():
    topology = structural.SparseTopology(
        input_source=np.array([0, 1]), input_target=np.array([0, 1]),
        input_value=np.ones(2), recurrent_source=np.array([0]),
        recurrent_target=np.array([1]), recurrent_value=np.ones(1),
        readout=np.zeros((2, 1)), dale=np.zeros(2), mechanisms=((), ()),
    )
    evidence = structural.structural_evidence(
        topology,
        readout_effect=np.zeros((1, 2)),
        spikes=np.zeros((1, 2)),
        gradient_mass=np.array([[2.0]]),
        input_gradient_mass=np.array([[3.0, 5.0]]),
    )
    np.testing.assert_allclose(
        evidence["neuron_task_scores"], [[5.0 / 21.0, 1.0 / 3.0]]
    )
    np.testing.assert_array_equal(evidence["target_incident_gradient"], [[5.0, 7.0]])


def test_wrong_output_readout_evidence_uses_only_incorrect_supervised_groups():
    voltages = np.full((31, 2), -65.0)
    voltages[0] = [-45.0, -65.0]
    weights = np.zeros((2, 360))
    weights[0, :30] = 2.0
    weights[0, 30:60] = 3.0
    bias = np.zeros(360)
    bias[1] = 4.0
    bias[30] = 4.0
    target = np.zeros((1, 1), dtype=np.uint8)
    result = structural.wrong_output_readout_evidence(
        voltages, weights, bias, target
    )
    assert result[0] == pytest.approx(60.0 * np.tanh(1.0))
    assert result[1] == 0.0


def test_connection_selector_stops_at_strict_next_tile_bound():
    selected, statistics = structural.select_connection_additions(
        600,
        existing=set(),
        source_evidence=np.arange(600.0, 0.0, -1.0),
        target_evidence=np.arange(600.0, 0.0, -1.0),
        required=1,
        return_statistics=True,
    )
    assert selected == ((0, 1),)
    assert statistics["tiles_scanned"] == 1
    assert statistics["tiles_total"] == 9
    assert statistics["stopped_by_bound"]
    assert statistics["max_resident_pairs"] == 65_536
    assert statistics["next_tile_upper_bound"] < statistics["worst_selected_score"]

    tied, tied_statistics = structural.select_connection_additions(
        4, set(), np.ones(4), np.ones(4), 1, tile_size=2,
        return_statistics=True,
    )
    assert tied == ((0, 1),)
    assert tied_statistics["tiles_scanned"] == 4
    assert not tied_statistics["stopped_by_bound"]


def test_fixed_task_evidence_measures_all_eight_training_tasks(monkeypatch):
    training_ids = tuple(f"train-{index}" for index in range(8))
    task_index = {task_id: index for index, task_id in enumerate(training_ids)}

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.array([0, 1]))
        recurrent_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.array([1, 0]))
        input_weight = SimpleNamespace(value=np.ones(2))
        recurrent_weight = SimpleNamespace(value=np.ones(2))
        readout_weight = SimpleNamespace(value=np.ones((2, 360)))
        readout_bias = SimpleNamespace(value=np.zeros(360))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.zeros(360)

    class Learner:
        def __init__(self):
            self.calls = []

        def etrace_grad(self, events, **kwargs):
            index = int(np.asarray(events)[0, 0])
            self.calls.append(index)
            return {
                ("model", "input_weight"): np.full(2, index + 1.0),
                ("model", "recurrent_weight"): np.full(2, index + 2.0),
            }, np.asarray(index, dtype=float)

        def etrace_evolve(self, event, return_outputs):
            return (event,)

    def load_task(_root, task_id, _role):
        return SimpleNamespace(
            task_id=task_id,
            targets=(np.zeros((1, 1), dtype=np.uint8),),
        )

    def encode_episode(task, _query_index):
        index = task_index.get(task.task_id, 0)
        events = np.zeros((31, 441), dtype=float)
        events[0, 0] = index
        return events, np.ones(31, dtype=bool)

    module = SimpleNamespace(
        TRAINING_TASK_IDS=training_ids,
        VALIDATION_TASK_IDS=("valid",),
        load_task=load_task,
        encode_episode=encode_episode,
        run_event_sequence=lambda *_args, **kwargs: (
            (np.full((31, 2), -45.0), np.ones((31, 2)))
            if kwargs.get("return_spikes") else np.full((31, 2), -45.0)
        ),
        decode_prediction=lambda _value: np.zeros((1, 1), dtype=np.uint8),
        strict_task_pass_at_1=lambda _predictions, _targets: False,
    )
    learner = Learner()
    evidence = structural._fixed_task_evidence(
        module, Model(), learner, "data", transform=_EagerTransform
    )
    assert evidence["training_task_ids"] == list(training_ids)
    assert learner.calls == list(range(8))
    assert np.asarray(evidence["task_spike_evidence"]).shape == (8, 2)
    assert np.asarray(evidence["preclip_gradient_mass"]).shape == (8, 2)
    assert np.asarray(evidence["input_preclip_gradient_mass"]).shape == (8, 2)
    assert np.asarray(evidence["training_events"]).shape == (8, 31, 441)


def test_parent_checkpoint_loads_nonzero_optimizer_state_and_distinct_steps(tmp_path):
    arrays = {
        "neuron_ids": np.arange(2, dtype=np.int32),
        "dale_codes": np.zeros(2, dtype=np.int8),
        "owner_codes": np.full(2, -1, dtype=np.int16),
        "mechanism_codes": np.zeros(2, dtype=np.uint8),
        "neuron_count": np.asarray(2, dtype=np.int32),
        "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.concatenate((np.array([0, 1]), np.ones(440))).astype(np.int32),
        "input_indices": np.array([0], dtype=np.int32),
        "input_values": np.array([1.0], dtype=np.float32),
        "input_m1": np.array([2.0], dtype=np.float32),
        "input_m2": np.array([3.0], dtype=np.float32),
        "recurrent_indptr": np.array([0, 1, 1], dtype=np.int32),
        "recurrent_indices": np.array([1], dtype=np.int32),
        "recurrent_values": np.array([4.0], dtype=np.float32),
        "recurrent_m1": np.array([5.0], dtype=np.float32),
        "recurrent_m2": np.array([6.0], dtype=np.float32),
        "readout_weight": np.ones((2, 360), dtype=np.float32),
        "readout_bias": np.ones(360, dtype=np.float32),
        "readout_weight_m1": np.full((2, 360), 7.0, dtype=np.float32),
        "readout_weight_m2": np.full((2, 360), 8.0, dtype=np.float32),
        "readout_bias_m1": np.full(360, 9.0, dtype=np.float32),
        "readout_bias_m2": np.full(360, 10.0, dtype=np.float32),
        "input_step": np.asarray(11, dtype=np.int64),
        "recurrent_step": np.asarray(12, dtype=np.int64),
        "readout_step": np.asarray(13, dtype=np.int64),
    }
    checkpoint_path = tmp_path / "parent.npz"
    checkpoint_path.write_bytes(b"exact-parent-checkpoint")
    module = SimpleNamespace(load_checkpoint=lambda path: arrays)
    parent = structural.load_parent_checkpoint(module, checkpoint_path)
    np.testing.assert_array_equal(parent.topology.input_source, [0])
    np.testing.assert_array_equal(parent.topology.recurrent_source, [0])
    assert parent.optimizer.input_step == 11
    assert parent.optimizer.recurrent_step == 12
    assert parent.optimizer.readout_step == 13
    assert parent.nonzero_optimizer_values
    assert parent.digest == hashlib.sha256(b"exact-parent-checkpoint").hexdigest()
    np.testing.assert_array_equal(parent.topology.owner_codes, [-1, -1])
    np.testing.assert_array_equal(parent.topology.neuron_ids, [0, 1])

    typed_parent = structural.load_parent_checkpoint(
        SimpleNamespace(load_checkpoint=lambda path: {
            **arrays, "dale_codes": np.array([1, 0], dtype=np.int8)
        }), checkpoint_path
    )
    np.testing.assert_array_equal(typed_parent.topology.dale, [1, 0])
    assert structural.validate_topology_dale(typed_parent.topology)
    with pytest.raises(ValueError, match="Dale codes"):
        structural.load_parent_checkpoint(
            SimpleNamespace(load_checkpoint=lambda path: {
                **arrays, "dale_codes": np.array([2, 0], dtype=np.int8)
            }), checkpoint_path
        )
    with pytest.raises(ValueError, match="mechanism codes"):
        structural.load_parent_checkpoint(
            SimpleNamespace(load_checkpoint=lambda path: {
                **arrays, "mechanism_codes": np.array([1, 0], dtype=np.uint8)
            }), checkpoint_path
        )

    zero_arrays = {**arrays, "input_m1": np.zeros(1, dtype=np.float32),
                   "input_m2": np.zeros(1, dtype=np.float32),
                   "recurrent_m1": np.zeros(1, dtype=np.float32),
                   "recurrent_m2": np.zeros(1, dtype=np.float32),
                   "readout_weight_m1": np.zeros((2, 360), dtype=np.float32),
                   "readout_weight_m2": np.zeros((2, 360), dtype=np.float32),
                   "readout_bias_m1": np.zeros(360, dtype=np.float32),
                   "readout_bias_m2": np.zeros(360, dtype=np.float32)}
    with pytest.raises(ValueError, match="nonzero optimizer"):
        structural.load_parent_checkpoint(
            SimpleNamespace(load_checkpoint=lambda path: zero_arrays), "zero.npz"
        )


def test_active_muon_state_is_loaded_from_parent_arrays_and_remapped():
    import jax
    import jax.numpy as jnp

    trainer = SimpleNamespace(
        parameters={
            "input": jnp.ones(2),
            "recurrent": jnp.ones(3),
            "readout_weight": jnp.ones((2, 4)),
            "readout_bias": jnp.ones(4),
        },
        learning_rates={"input": 0.1, "recurrent": 0.1, "readout": 0.1},
    )
    optimizer = structural.StructuralAdam(
        np.full((2, 4), 7.0), np.full((2, 4), 8.0),
        np.full(2, 2.0), np.full(2, 3.0),
        np.full(3, 4.0), np.full(3, 5.0),
        bias_first=np.full(4, 9.0), bias_second=np.full(4, 10.0),
        input_step=11, recurrent_step=12, readout_step=13,
    )
    groups = structural.initialize_muon_groups(trainer, optimizer)
    assert set(groups) == set(trainer.parameters)
    leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(groups)]
    assert any(np.any(value == 2.0) for value in leaves)
    assert any(np.any(value == 7.0) for value in leaves)
    mapped = structural.remap_muon_groups(
        groups,
        {
            "input": (np.arange(2), (3,)),
            "recurrent": (np.arange(3), (4,)),
            "readout_weight": (np.array([True, False]), (1, 4)),
            "readout_bias": (np.arange(4), (4,)),
        },
    )
    checks = structural.muon_remap_checks(
        groups,
        mapped,
        {
            "input": (np.arange(2), (3,)),
            "recurrent": (np.arange(3), (4,)),
            "readout_weight": (np.array([True, False]), (1, 4)),
            "readout_bias": (np.arange(4), (4,)),
        },
    )
    assert checks == {
        "loaded": True,
        "source_nonzero": True,
        "surviving_values_preserved": True,
    }
    input_shapes = [
        getattr(value, "shape", None)
        for value in jax.tree_util.tree_leaves(mapped["input"])
    ]
    assert input_shapes.count((3,)) >= 3
    trainer.muon_groups = groups
    restored = structural.optimizer_from_muon_groups(trainer)
    np.testing.assert_array_equal(restored.input_first, np.full(2, 2.0))
    np.testing.assert_array_equal(restored.neuron_first, np.full((2, 4), 7.0))
    assert restored.input_step == 11
    assert restored.recurrent_step == 12
    assert restored.readout_step == 13


def test_real_update_adds_target_dependent_direct_readout_gradients():
    import jax.numpy as jnp

    class Model:
        input_weight = SimpleNamespace(value=jnp.ones(1))
        recurrent_weight = SimpleNamespace(value=jnp.ones(1))
        readout_weight = SimpleNamespace(value=jnp.zeros((2, 360)))
        readout_bias = SimpleNamespace(value=jnp.zeros(360))

        def reset_episode(self, learner):
            pass

    class Learner:
        model4compile = Model()

        def etrace_evolve(self, event, return_outputs):
            return (jnp.zeros((1, 2)),)

    class Trainer:
        def __init__(self, learner, parameters):
            self.parameters = {
                **parameters,
                "readout_weight": learner.model4compile.readout_weight.value,
                "readout_bias": learner.model4compile.readout_bias.value,
            }
            self.gradients = None

        def update_episode(self, events, **kwargs):
            self.gradients = kwargs["direct_grad_fn"]()
            return 0.0, 0.0

    module = SimpleNamespace(
        PPPropEpisodeTrainer=Trainer,
        run_event_sequence=lambda model, events, mask: jnp.full((31, 2), -45.0),
    )
    evidence = {
        "training_events": np.zeros((1, 31, 441)),
        "training_advances": np.ones((1, 31), dtype=bool),
        "events": np.zeros((31, 441)),
        "advances": np.ones(31, dtype=bool),
        "training_target_colors": np.zeros((1, 30, 30), dtype=np.int32),
        "training_target_valid": np.pad(
            np.ones((1, 1, 1), dtype=bool), ((0, 0), (0, 29), (0, 29))
        ),
        "training_target_heights": np.zeros(1, dtype=np.int32),
        "training_target_widths": np.zeros(1, dtype=np.int32),
    }
    update = structural._real_pp_prop_update(
        module, Learner.model4compile, Learner(), evidence
    )
    update(0)
    assert np.any(np.asarray(update.trainer.gradients[("readout_weight",)]) != 0)
    assert np.any(np.asarray(update.trainer.gradients[("readout_bias",)]) != 0)


def test_checkpoint_arrays_preserve_sparse_topology_and_optimizer_values():
    class Model:
        input_csr = SimpleNamespace(
            indptr=np.concatenate((np.array([0, 1, 2]), np.full(439, 2))),
            indices=np.array([0, 1]),
        )
        recurrent_csr = SimpleNamespace(
            indptr=np.array([0, 1, 2]), indices=np.array([1, 0])
        )
        input_weight = SimpleNamespace(value=np.array([1.0, 2.0], dtype=np.float32))
        recurrent_weight = SimpleNamespace(value=np.array([3.0, 4.0], dtype=np.float32))
        readout_weight = SimpleNamespace(value=np.ones((2, 360), dtype=np.float32))
        readout_bias = SimpleNamespace(value=np.ones(360, dtype=np.float32))
        owner_codes = np.array([7, 8], dtype=np.int16)
        neuron_ids = np.array([41, 55], dtype=np.int32)

    optimizer = structural.StructuralAdam(
        np.full((2, 360), 1.0), np.full((2, 360), 2.0),
        np.full(2, 3.0), np.full(2, 4.0),
        np.full(2, 5.0), np.full(2, 6.0),
        bias_first=np.full(360, 7.0), bias_second=np.full(360, 8.0),
        input_step=9, recurrent_step=10, readout_step=11,
    )
    arrays = structural.checkpoint_arrays(
        Model(), optimizer, {"owners": ((), (0, 1))}
    )
    assert arrays["input_indptr"].shape == (442,)
    np.testing.assert_array_equal(arrays["input_indices"], [0, 1])
    np.testing.assert_array_equal(arrays["recurrent_indices"], [1, 0])
    np.testing.assert_array_equal(arrays["owner_codes"], [7, 8])
    np.testing.assert_array_equal(arrays["neuron_ids"], [41, 55])
    assert int(arrays["input_step"]) == 9
    assert int(arrays["recurrent_step"]) == 10
    assert int(arrays["readout_step"]) == 11


def test_parent_writer_uses_real_update_state_and_writes_digest(
    monkeypatch, tmp_path
):
    model = SimpleNamespace()
    learner = SimpleNamespace()
    trainer = SimpleNamespace()
    def update(index):
        return index

    update.trainer = trainer
    module = SimpleNamespace(
        BrainCellArcModel=lambda: model,
        compile_pp_prop_model=lambda value: learner,
        write_checkpoint=lambda path, arrays: Path(path).write_bytes(b"checkpoint"),
    )
    optimizer = structural.StructuralAdam(
        np.ones((1, 1)), np.ones((1, 1)), np.ones(1), np.ones(1),
        np.ones(1), np.ones(1), bias_first=np.ones(1), bias_second=np.ones(1),
        input_step=128, recurrent_step=128, readout_step=128,
    )
    arrays = {
        "input_m1": np.ones(1), "input_m2": np.ones(1),
        "recurrent_m1": np.ones(1), "recurrent_m2": np.ones(1),
        "readout_weight_m1": np.ones(1), "readout_weight_m2": np.ones(1),
        "readout_bias_m1": np.ones(1), "readout_bias_m2": np.ones(1),
        "input_step": np.asarray(128), "recurrent_step": np.asarray(128),
        "readout_step": np.asarray(128),
    }
    monkeypatch.setattr(
        structural, "_fixed_task_evidence",
        lambda *args: {"strict": [False, True]},
    )
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args: update)
    monkeypatch.setattr(structural, "run_addition_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        structural, "_fixed_strict_screen", lambda *args: (True, True)
    )
    monkeypatch.setattr(
        structural, "optimizer_from_muon_groups", lambda value: optimizer
    )
    monkeypatch.setattr(
        structural, "checkpoint_arrays", lambda *args: arrays
    )
    monkeypatch.setattr(structural, "_git_commit", lambda: "commit")
    target = tmp_path / "parent.npz"
    result = structural.write_parent_checkpoint(module, target, "data")
    assert result["updates"] == 128
    assert result["optimizer_nonzero"]
    assert result["optimizer_steps"] == {
        "input": 128, "recurrent": 128, "readout": 128
    }
    assert result["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()

    monkeypatch.setattr(
        structural, "_fixed_strict_screen", lambda *args: (False, False)
    )
    with pytest.raises(ValueError, match="strict regression"):
        structural.write_parent_checkpoint(module, target, "data")


def test_merge_validation_accepts_honest_nonpromotion_and_rejects_invalid_evidence():
    task_ids = [f"task-{index}" for index in range(8)]

    def arm(name, pid):
        neuron_arm = name.startswith("neuron")
        addition = name.endswith("add")
        result = {
            "arm": name,
            "environment": {
                "pid": 1,
                "pid_namespace": "pid:[shared]",
                "process_start_ticks": pid,
            },
            "implementation_commit": "abc",
            "real_model": True,
            "baseline_neurons": 100,
            "candidate_neurons": 105 if name == "neuron-add" else (95 if name == "neuron-prune" else 100),
            "baseline_recurrent_items": 100,
            "candidate_recurrent_items": 105 if name == "connection-add" else (95 if name == "connection-prune" else 100),
            "mutated_item_count": 5,
            "updates": 128 if addition else 0,
            "before_strict": [False] + [True] * 11,
            "after_strict": [True] * 12,
            "promoted": True,
            "strict_regression_rejected": True,
            "within_300_seconds": True,
            "complete_process_seconds": 299.0,
            "dense_neuron_pair_array": False,
            "parent_checkpoint_sha256": "parent",
            "parent_checkpoint_sha256_after": "parent",
            "parent_checkpoint_unchanged": True,
            "parent_optimizer_nonzero": True,
            "training_evidence_task_ids": task_ids,
            "adam_remapped": True,
            "muon_remapped": True,
            "optimizer_remap": {
                "surviving_values_preserved": True,
                "new_values_zero": True,
                "step_counts_preserved": True,
            },
            "mask_compaction": {
                "prediction_bytes_identical": True,
                "strict_identical": True,
            } if name == "neuron-prune" else {"not_measured": True},
            "connection_selection": {
                "stopped_by_bound": True,
                "max_resident_pairs": 65_536,
            } if name == "connection-add" else {
                "stopped_by_bound": False,
                "max_resident_pairs": 0,
            },
        }
        assert neuron_arm or name.startswith("connection")
        return result

    arms = [arm(name, index) for index, name in enumerate(
        ("neuron-prune", "connection-prune", "neuron-add", "connection-add"), 1
    )]
    structural.validate_merged_arms(arms)
    for arm_evidence in arms:
        arm_evidence["environment"]["pid"] = 1
    structural.validate_merged_arms(arms)
    for arm_evidence in arms:
        arm_evidence["before_strict"] = [False] * 12
        arm_evidence["after_strict"] = [False] * 12
        arm_evidence["promoted"] = False
    for arm_evidence in arms[:2]:
        arm_evidence.update({
            "candidate_neurons": arm_evidence["baseline_neurons"],
            "candidate_recurrent_items": arm_evidence["baseline_recurrent_items"],
            "mutated_item_count": 0,
            "pruning_blocked": True,
            "muon_remapped": False,
            "mask_compaction": {"not_measured": True},
        })
    structural.validate_merged_arms(arms)

    invalid_promotion = copy.deepcopy(arms)
    invalid_promotion[2]["promoted"] = True
    with pytest.raises(ValueError, match="promotion record"):
        structural.validate_merged_arms(invalid_promotion)
    invalid_parent = copy.deepcopy(arms)
    invalid_parent[3]["parent_checkpoint_unchanged"] = False
    with pytest.raises(ValueError, match="parent checkpoint"):
        structural.validate_merged_arms(invalid_parent)

    arms = [arm(name, index) for index, name in enumerate(
        ("neuron-prune", "connection-prune", "neuron-add", "connection-add"), 1
    )]
    arms[3]["connection_selection"]["stopped_by_bound"] = False
    with pytest.raises(ValueError, match="tile bound"):
        structural.validate_merged_arms(arms)

    arms = [arm(name, index) for index, name in enumerate(
        ("neuron-prune", "connection-prune", "neuron-add", "connection-add"), 1
    )]
    mutations = (
        (lambda values: values.reverse(), "fixed order"),
        (lambda values: values[1].update(
            environment=values[0]["environment"].copy()
        ), "separate process"),
        (lambda values: values[1].update(implementation_commit="def"), "implementation commit"),
        (lambda values: values[1].update(parent_checkpoint_sha256="other"), "parent checkpoint"),
        (lambda values: values[0].update(
            after_strict=values[0]["before_strict"].copy()
        ), "promotion record"),
        (lambda values: values[0].update(after_strict=[True, False] + [True] * 10), "strict regression"),
        (lambda values: values[0].update(within_300_seconds=False), "300-second"),
        (lambda values: values[0].update(complete_process_seconds=300.01), "complete process"),
        (lambda values: values[0].update(dense_neuron_pair_array=True), "sparse pair"),
        (lambda values: values[0].update(training_evidence_task_ids=[]), "eight training"),
        (lambda values: values[0].update(parent_optimizer_nonzero=False), "nonzero parent"),
        (lambda values: values[0]["optimizer_remap"].update(new_values_zero=False), "optimizer state"),
        (lambda values: values[0].update(muon_remapped=False), "active optimizer"),
        (lambda values: values[2].update(updates=63), "update count"),
        (lambda values: values[2].update(mutated_item_count=4), "mutation count"),
        (lambda values: values[0]["mask_compaction"].update(strict_identical=False), "compaction identity"),
    )
    for mutate, message in mutations:
        invalid = copy.deepcopy(arms)
        mutate(invalid)
        with pytest.raises(ValueError, match=message):
            structural.validate_merged_arms(invalid)


def test_coverage_summary_requires_branch_data_above_ninety(monkeypatch):
    class Data:
        def __init__(self, branches):
            self.branches = branches

        def has_arcs(self):
            return self.branches

    class Coverage:
        branches = True
        percent = 91.25

        def __init__(self, config_file):
            assert config_file is False

        def load(self):
            pass

        def get_data(self):
            return Data(self.branches)

        def report(self, show_missing, include):
            assert not show_missing
            assert include == ["examples/pp_prop/example21_structural.py"]
            return self.percent

    monkeypatch.setitem(sys.modules, "coverage", SimpleNamespace(Coverage=Coverage))
    assert structural._coverage_summary() == {
        "line_and_branch_percent": 91.25,
        "branch_data": True,
    }
    Coverage.branches = False
    with pytest.raises(ValueError, match="line-plus-branch"):
        structural._coverage_summary()


def test_peak_process_resident_memory_reads_linux_high_water_mark(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmHWM:\t1234 kB\n")
    assert structural._peak_process_resident_memory_bytes(status) == 1_263_616
    status.write_text("Name:\tpython\n")
    assert structural._peak_process_resident_memory_bytes(status) is None
    assert structural._peak_process_resident_memory_bytes(tmp_path / "missing") is None


def test_peak_process_resident_memory_uses_process_fallback(tmp_path):
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(peak_wset=9_876_543, rss=1_234_567)
    )
    assert structural._peak_process_resident_memory_bytes(
        tmp_path / "missing", process=process
    ) == 9_876_543

    stat = tmp_path / "stat"
    fields = ["R"] + ["0"] * 18 + ["9876"]
    stat.write_text(f"13 (python worker) {' '.join(fields)}\n")
    assert structural._process_start_ticks(stat) == 9876
    assert structural._process_start_ticks(tmp_path / "missing-stat") is None


def test_merge_cli_uses_measured_files_and_arm_cli_requires_parent(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    names = ("neuron-prune", "connection-prune", "neuron-add", "connection-add")
    for name in names:
        (tmp_path / f".gate5-{name}.json").write_text(json.dumps({
            "arm": name,
            "implementation_commit": "code-commit",
            "within_300_seconds": True,
        }))
    (tmp_path / ".gate5-baseline.json").write_text(json.dumps({
        "baseline": {"neurons": 2048}
    }))
    monkeypatch.setattr(structural, "validate_merged_arms", lambda arms: None)
    monkeypatch.setattr(
        structural, "_coverage_summary",
        lambda: {"line_and_branch_percent": 91.5, "branch_data": True},
    )
    monkeypatch.setattr(structural, "_git_commit", lambda: "artifact-commit")
    structural.main([
        "merge", "--output", "merged.json", "--focused-passed", "37"
    ])
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert merged["implementation_commit"] == "code-commit"
    assert merged["artifact_build_commit"] == "artifact-commit"
    assert merged["focused_tests"]["line_and_branch_percent"] == 91.5
    assert merged["complete_process_seconds"] >= 0.0
    assert merged["peak_process_resident_memory_bytes"] is not None
    with pytest.raises(SystemExit):
        structural.main(["neuron-add", "--output", "arm.json"])


def test_git_commit_fails_closed_when_git_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        structural.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git")),
    )
    assert structural._git_commit() == "unknown"
