"""Tests for the post-training neuron-pruning example."""

import importlib.util
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLE = (
    pathlib.Path(__file__).resolve().with_name("20-post-training-neuron-pruning.py")
)


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_neuron_pruning", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_task_rows_handles_zero_rows_without_nan():
    values = np.array([[0.0, 2.0], [0.0, 0.0]])
    normalized = _load()._normalize_task_rows(values)
    assert normalized.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_contribution_scores_protect_each_task_and_break_ties_by_index():
    example = _load()
    scores, task_scores, owners = example._contribution_scores(
        rates=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        readout_weight=np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]]),
        rows=np.array([], dtype=int),
        cols=np.array([], dtype=int),
        values=np.array([], dtype=float),
        task_mass=np.zeros((2, 0)),
        n_rec=3,
    )
    assert np.all(np.isfinite(scores))
    assert owners.tolist() == [0, 1, 0]
    assert task_scores[:, 2].tolist() == [0.0, 0.0]
    assert example._removal_order(scores).tolist() == [2, 0, 1]


def test_edge_contribution_scores_mask_dead_coordinates_and_protect_tasks():
    example = _load()
    scores, task_scores, owners = example._edge_contribution_scores(
        rates=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        rows=np.array([1, 0, 0]),
        cols=np.array([0, 1, 2]),
        values=np.array([2.0, 3.0, 100.0]),
        task_mass=np.array([[1.0, 0.0, 10.0], [0.0, 2.0, 10.0]]),
        neuron_alive=np.array([1.0, 1.0, 0.0]),
        edge_alive=np.array([1.0, 1.0, 1.0]),
    )
    assert owners.tolist() == [0, 1, 0]
    assert scores[0] > 0.0
    assert scores[1] > 0.0
    assert scores[2] == 0.0
    assert np.all(task_scores[:, 2] == 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rates": np.zeros((2, 2))}, "rates"),
        ({"readout_weight": np.zeros((3, 3))}, "readout"),
        ({"task_mass": np.zeros((3, 0))}, "task_mass"),
        ({"values": np.array([np.nan])}, "aligned"),
    ],
)
def test_contribution_scores_reject_invalid_contracts(kwargs, message):
    inputs = {
        "rates": np.zeros((2, 3)),
        "readout_weight": np.zeros((3, 2)),
        "rows": np.array([], dtype=int),
        "cols": np.array([], dtype=int),
        "values": np.array([], dtype=float),
        "task_mass": np.zeros((2, 0)),
        "n_rec": 3,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        _load()._contribution_scores(**inputs)


def test_candidate_counts_and_alive_masks_include_both_endpoints():
    example = _load()
    counts = example._coarse_removed_counts(10, 0.25)
    assert counts.tolist() == [0, 3, 6, 9]
    masks = example._alive_masks(np.array([2, 0, 1]), np.array([0, 1, 2]))
    assert masks.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_device_binding_fails_closed_on_backend_mismatch(monkeypatch):
    example = _load()
    updates = []
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(
            update=lambda name, value: updates.append((name, value))
        ),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setattr(example, "jax", fake_jax)
    assert example._bind_device("auto") == "cpu"
    with pytest.raises(
        RuntimeError, match="requested device gpu, bound backend is cpu"
    ):
        example._bind_device("gpu")
    assert updates == [("jax_platform_name", "gpu")]


def test_device_binding_rejects_invalid_and_unavailable_backends(monkeypatch):
    example = _load()
    with pytest.raises(ValueError, match="device must be"):
        example._bind_device("tpu")

    def unavailable():
        raise RuntimeError("backend missing")

    fake_jax = SimpleNamespace(
        config=SimpleNamespace(update=lambda name, value: None),
        default_backend=unavailable,
    )
    monkeypatch.setattr(example, "jax", fake_jax)
    with pytest.raises(RuntimeError, match="JAX backend is unavailable"):
        example._bind_device("gpu")


@pytest.mark.parametrize("fraction", [0.0, 1.0, np.nan])
def test_candidate_counts_reject_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        _load()._coarse_removed_counts(4, fraction)


def test_frontier_is_contiguous_and_reports_later_recovery():
    example = _load()
    counts = np.array([0, 2, 4, 6])
    accuracies = np.array([[1.0, 1.0], [1.0, 0.9], [0.8, 0.9], [1.0, 1.0]])
    frontier = example._select_safe_frontier(counts, accuracies, 0.9)
    assert frontier == {
        "baseline_eligible": True,
        "safe_removed": 2,
        "first_failed_removed": 4,
        "later_recovery": True,
    }
    assert example._refinement_counts(frontier).tolist() == [3, 4]


def test_frontier_stops_when_baseline_is_below_target():
    frontier = _load()._select_safe_frontier(
        np.array([0, 2]), np.array([[0.75, 1.0], [1.0, 1.0]]), 1.0
    )
    assert frontier["baseline_eligible"] is False
    assert frontier["safe_removed"] == 0
    assert frontier["first_failed_removed"] == 0
    assert frontier["later_recovery"] is True
    assert _load()._refinement_counts(frontier).size == 0


def test_twin_and_task_alignment_distinguishes_full_and_partial_classes():
    alignment = _load()._alignment_summary(
        class_of=np.array([0, 0, 2, 2, 4]),
        removed=np.array([True, False, True, True, False]),
        owners=np.array([0, 1, 0, 1, 1]),
        n_tasks=2,
    )
    assert alignment == {
        "removed_task_counts": [2, 1],
        "retained_task_counts": [0, 2],
        "removed_twin_neurons": 3,
        "fully_removed_twin_classes": 1,
        "partially_pruned_twin_classes": 1,
    }


def test_alignment_owners_keep_removed_initial_and_retained_final_assignments():
    owners = _load()._alignment_owners(
        initial_owners=np.array([1, 1, 0, 0]),
        final_owners=np.array([0, 0, 1, 1]),
        removed=np.array([True, False, True, False]),
    )
    assert owners.tolist() == [1, 0, 0, 1]


def test_all_dead_mask_blocks_every_neuron_output():
    import brainstate
    import brainunit as u

    example = _load()
    config = replace(example.EX18._EvolutionConfig.smoke(), eval_trials_per_task=1)
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example.EX18._initial_experiment(config)
        accuracies, rates = example._evaluate_alive_masks(
            experiment,
            config,
            np.stack([np.ones(config.n_rec), np.zeros(config.n_rec)]),
        )
    assert accuracies.shape == (2, config.num_tricks)
    assert rates.shape == (2, config.num_tricks, config.n_rec)
    assert np.all(rates[1] == 0.0)


def test_edge_lesion_evaluation_restores_trained_recurrent_values():
    import brainstate
    import brainunit as u

    example = _load()
    config = replace(example.EX18._EvolutionConfig.smoke(), eval_trials_per_task=1)
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example.EX18._initial_experiment(config)
        _, _, before = example.EX18._current_topology(experiment)
        accuracies, rates = example._evaluate_structural_masks(
            experiment,
            config,
            np.ones((1, config.n_rec)),
            np.zeros((1, experiment.task_mass.shape[1])),
        )
        _, _, after = example.EX18._current_topology(experiment)
    assert accuracies.shape == (1, config.num_tricks)
    assert rates.shape == (1, config.num_tricks, config.n_rec)
    np.testing.assert_array_equal(after, before)


def test_compact_topology_remaps_only_active_live_live_edges():
    retained, rows, cols, values = _load()._compact_topology(
        rows=np.array([0, 1, 2, 3]),
        cols=np.array([1, 3, 3, 1]),
        values=np.array([0.1, 0.2, 0.3, 0.4]),
        neuron_alive=np.array([0, 1, 0, 1]),
        edge_alive=np.array([0, 1, 0, 1]),
    )
    assert retained.tolist() == [1, 3]
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]
    assert values.tolist() == [0.2, 0.4]


def test_compacted_bundle_matches_masked_model_and_reloads(tmp_path):
    import brainstate
    import brainunit as u

    example = _load()
    config = replace(example.EX18._EvolutionConfig.smoke(), eval_trials_per_task=1)
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example.EX18._initial_experiment(config)
        rows, cols, _ = example.EX18._current_topology(experiment)
        neuron_alive = np.ones(config.n_rec, dtype=np.float32)
        neuron_alive[-1] = 0.0
        edge_alive = neuron_alive[rows] * neuron_alive[cols]
        compact = example._build_compact_model(
            experiment, config, neuron_alive, edge_alive
        )
        masked_logits, masked_accuracy = example._evaluate_probe_logits(
            experiment, config, neuron_alive, edge_alive
        )
        compact_logits, compact_accuracy = example._evaluate_probe_logits(
            compact["experiment"],
            compact["config"],
            np.ones(compact["config"].n_rec),
            np.ones(compact["rows"].size),
        )
        np.testing.assert_allclose(compact_logits, masked_logits, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(compact_accuracy, masked_accuracy)

        output = tmp_path / "compact.npz"
        example._save_compact_bundle(compact, output)
        loaded = example._load_compact_bundle(output)
        loaded_logits, loaded_accuracy = example._evaluate_probe_logits(
            loaded["experiment"],
            loaded["config"],
            np.ones(loaded["config"].n_rec),
            np.ones(loaded["rows"].size),
        )
        np.testing.assert_allclose(loaded_logits, compact_logits, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(loaded_accuracy, compact_accuracy)

        compaction = example._analyze_compaction(
            experiment,
            config,
            {
                "final_alive_mask": neuron_alive.tolist(),
                "final_edge_alive_mask": edge_alive.tolist(),
            },
            target=0.0,
            output=tmp_path / "analyzed-compact.npz",
            benchmark_repetitions=1,
        )
    assert loaded["original_neuron_indices"].tolist() == list(range(config.n_rec - 1))
    assert compaction["status"] == "complete"
    assert compaction["predictions_identical"] is True
    assert (
        compaction["compact_storage"]["total_bytes"]
        < compaction["original_storage"]["total_bytes"]
    )
    assert compaction["masked_probe_ms"] > 0.0
    assert compaction["compact_probe_ms"] > 0.0
    assert (tmp_path / "analyzed-compact.npz").exists()


def test_compaction_rejects_active_edge_incident_to_dead_neuron():
    with pytest.raises(ValueError, match="active edge"):
        _load()._compact_topology(
            rows=np.array([0]),
            cols=np.array([1]),
            values=np.array([1.0]),
            neuron_alive=np.array([1, 0]),
            edge_alive=np.array([1]),
        )


def test_compact_model_supports_zero_recurrent_edges():
    import brainstate
    import brainunit as u

    example = _load()
    config = replace(example.EX18._EvolutionConfig.smoke(), eval_trials_per_task=1)
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example.EX18._initial_experiment(config)
        neuron_alive = np.ones(config.n_rec, dtype=np.float32)
        edge_alive = np.zeros(experiment.task_mass.shape[1], dtype=np.float32)
        compact = example._build_compact_model(
            experiment, config, neuron_alive, edge_alive
        )
        masked_logits, _ = example._evaluate_probe_logits(
            experiment, config, neuron_alive, edge_alive
        )
        compact_logits, _ = example._evaluate_probe_logits(
            compact["experiment"],
            compact["config"],
            neuron_alive,
            np.zeros(0, dtype=np.float32),
        )
    assert compact["rows"].size == 0
    np.testing.assert_allclose(compact_logits, masked_logits, rtol=1e-5, atol=1e-6)


def test_compaction_skips_an_empty_neuron_network():
    result = _load()._analyze_compaction(
        None,
        None,
        {"final_alive_mask": [0], "final_edge_alive_mask": []},
        target=0.0,
        output=None,
        benchmark_repetitions=1,
    )
    assert result["status"] == "skipped_empty_network"


def test_compaction_benchmark_rejects_nonpositive_repetitions():
    with pytest.raises(ValueError, match="repetitions"):
        _load()._benchmark_compaction(None, None, None, None, None, repetitions=0)


def _small_experiment(trials_per_task=2):
    """Build a tiny untrained Example 18 experiment for evaluator equivalence."""
    import brainstate
    import brainunit as u

    example = _load()
    config = replace(
        example.EX18._EvolutionConfig.smoke(), eval_trials_per_task=trials_per_task
    )
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example.EX18._initial_experiment(config)
    return example, config, experiment


def test_batched_evaluator_matches_the_unbatched_rollout():
    """Putting the probe trials on the batch axis must not change the network.

    Any leakage between trials, or any state shared across the batch, would
    show up here: the unbatched reference resets between every trial, so the
    two agree only if the batched rollout keeps them independent.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    example, config, experiment = _small_experiment()
    edge_count = int(experiment.task_mass.shape[1])
    rng = np.random.default_rng(3)
    alive = jnp.asarray((rng.random(config.n_rec) > 0.25).astype(np.float32))
    edge_alive = jnp.asarray((rng.random(edge_count) > 0.2).astype(np.float32))
    with brainstate.environ.context(dt=1.0 * u.ms):
        serial = brainstate.transform.jit(
            example._probe_logit_evaluator(experiment, config)
        )
        expected_logits, expected_rates = serial(alive, edge_alive)
        batched = brainstate.transform.jit(
            example._batched_logit_evaluator(experiment, config)
        )
        logits, rates = batched(alive, edge_alive)
    np.testing.assert_array_equal(np.asarray(rates), np.asarray(expected_rates))
    np.testing.assert_array_equal(
        np.argmax(np.asarray(logits), axis=1),
        np.argmax(np.asarray(expected_logits), axis=1),
    )
    np.testing.assert_allclose(
        np.asarray(logits), np.asarray(expected_logits), rtol=1e-6, atol=1e-7
    )


def test_batched_evaluator_silences_every_neuron_under_a_dead_mask():
    """A dead mask must zero the rates of every trial in the batch, not one."""
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    example, config, experiment = _small_experiment()
    edge_count = int(experiment.task_mass.shape[1])
    with brainstate.environ.context(dt=1.0 * u.ms):
        batched = brainstate.transform.jit(
            example._batched_logit_evaluator(experiment, config)
        )
        dead, dead_rates = batched(
            jnp.zeros(config.n_rec), jnp.ones(edge_count)
        )
        live, live_rates = batched(jnp.ones(config.n_rec), jnp.ones(edge_count))
    assert np.all(np.asarray(dead_rates) == 0.0)
    assert np.any(np.asarray(live_rates) > 0.0)
    assert np.asarray(dead).shape == (
        config.num_tricks * config.eval_trials_per_task,
        config.num_tricks,
    )


def test_priming_sizes_states_to_the_probe_batch():
    """The carrying loops need the state batch set before they are traced."""
    import brainstate
    import brainunit as u

    example, config, experiment = _small_experiment()
    trials = config.num_tricks * config.eval_trials_per_task
    with brainstate.environ.context(dt=1.0 * u.ms):
        evaluator = example._mask_evaluator(experiment, config)
        example._prime_evaluator(evaluator)
        membrane = experiment.model.neu.V.value
        # An evaluator that drives no model exposes no prime and needs none.
        example._prime_evaluator(lambda alive, edges: None)
    assert membrane.shape[0] == trials


def test_probe_arrays_are_memoized_per_configuration():
    example, config, _ = _small_experiment()
    first = example._probe_arrays(config)
    assert example._probe_arrays(config) is first


def test_joint_fixed_point_revisits_neurons_after_edge_pruning(monkeypatch):
    import jax.numpy as jnp

    example = _load()
    config = SimpleNamespace(n_rec=3, num_tricks=1)

    def fake_evaluator(experiment, current_config):
        del experiment, current_config

        def evaluate(neuron_alive, edge_alive):
            neuron_code = jnp.sum(neuron_alive * jnp.array([4.0, 2.0, 1.0]))
            edge_code = jnp.sum(edge_alive * jnp.array([2.0, 1.0]))
            safe = jnp.logical_or(
                jnp.logical_and(neuron_code == 7.0, edge_code >= 1.0),
                jnp.logical_and(neuron_code == 3.0, edge_code == 0.0),
            )
            return jnp.array([safe], dtype=jnp.float32), neuron_alive[None, :]

        return evaluate

    monkeypatch.setattr(example, "_mask_evaluator", fake_evaluator)
    monkeypatch.setattr(example, "_readout_weight", lambda model: np.zeros((3, 1)))
    result = example._joint_fixed_point_prune(
        SimpleNamespace(model=object()),
        config,
        np.ones(3),
        1.0,
        np.array([1, 0]),
        np.array([2, 2]),
        np.zeros(2),
        np.zeros((1, 2)),
    )
    assert result["converged"] is True
    assert result["cycle_count"] == 2
    assert result["accepted_neurons"] == [0]
    assert result["accepted_edges"] == [0]
    assert result["retained_indices"] == [1, 2]
    assert result["final_active_edge_count"] == 0
    assert result["final_original_live_live_edge_count"] == 1
    assert result["incident_edge_count"] == 1
    assert result["causally_removed_live_live_edge_count"] == 1
    assert result["retained_single_ablation_accuracies"] == [[0.0], [0.0]]
    assert result["retained_single_edge_ablation_accuracies"] == []


def test_joint_fixed_point_rejects_invalid_initial_mask(monkeypatch):
    example = _load()
    config = SimpleNamespace(n_rec=2, num_tricks=1)
    with pytest.raises(ValueError, match="initial_alive"):
        example._joint_fixed_point_prune(
            SimpleNamespace(model=object()),
            config,
            np.array([1.0, 0.5]),
            1.0,
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.zeros((1, 0)),
        )


def test_report_and_plot_show_selected_frontier(tmp_path, capsys, monkeypatch):
    from matplotlib.axes import Axes

    example = _load()
    analysis = {
        "status": "complete",
        "target": 1.0,
        "names": ["fetch", "roll_over"],
        "baseline_accuracies": [1.0, 1.0],
        "scores": [0.0, 1.0, 0.5],
        "owners": [0, 1, 0],
        "removed_counts": [0, 1, 2],
        "accuracies": [[1.0, 1.0], [1.0, 1.0], [0.5, 1.0]],
        "initial_frontier_removed": 1,
        "initial_frontier_retained": 2,
        "fixed_point_removed_counts": [1, 2],
        "fixed_point_accuracies": [[1.0, 1.0], [1.0, 1.0]],
        "fixed_point": {
            "accepted_per_pass": [1, 0],
            "final_accuracies": [1.0, 1.0],
            "retained_zero_score_count": 0,
        },
        "compaction": {
            "status": "complete",
            "n_rec": 1,
            "n_edges": 2,
            "accuracies": [1.0, 1.0],
            "max_abs_logit_error": 1e-7,
            "original_storage": {"total_bytes": 1000},
            "compact_storage": {"total_bytes": 100},
            "storage_reduction_fraction": 0.9,
            "benchmark_repetitions": 3,
            "masked_probe_ms": 4.0,
            "compact_probe_ms": 1.0,
            "speedup": 4.0,
            "bundle_path": "/tmp/compact.npz",
            "bundle_bytes": 256,
        },
        "safe_removed": 2,
        "safe_retained": 1,
        "first_failed_removed": 2,
        "later_recovery": False,
        "alignment": {
            "removed_task_counts": [1, 0],
            "retained_task_counts": [1, 1],
            "removed_twin_neurons": 1,
            "fully_removed_twin_classes": 0,
            "partially_pruned_twin_classes": 1,
        },
    }
    report = example._format_report(analysis)
    assert "Initial fixed-ranking frontier: 2/3 neurons" in report
    assert "locally minimal network: 1/3 neurons" in report
    assert "[1, 0]" in report
    assert "first failed" in report
    assert "Physical compact model: 1 neurons, 2 recurrent edges" in report
    assert "speedup=4.00x" in report
    output = tmp_path / "pruning.png"
    scales = []
    original_set_xscale = Axes.set_xscale

    def record_xscale(axis, value, **kwargs):
        scales.append(value)
        return original_set_xscale(axis, value, **kwargs)

    monkeypatch.setattr(Axes, "set_xscale", record_xscale)
    example._plot_pruning(analysis, output)
    example._print_report(report)
    assert output.exists()
    assert scales == ["log"]
    assert "20-neuron-pruning" in capsys.readouterr().out


def test_main_forwards_example_18_args_and_attaches_analysis(monkeypatch, tmp_path):
    example = _load()
    seen = {}
    arm = {
        "rows": np.array([], dtype=int),
        "cols": np.array([], dtype=int),
        "task_mass": np.zeros((2, 0)),
        "trick_names": ["fetch", "roll_over"],
    }
    result = {
        "config": SimpleNamespace(n_rec=2),
        "evolve": arm,
        "control": arm,
    }

    def fake_main(argv, evolve_posthoc=None, evolve_checkpoint=None):
        seen["argv"] = argv
        evolve_checkpoint(
            "experiment", arm["trick_names"], result["config"], 3, (1.0, 1.0)
        )
        evolve_posthoc("experiment", arm, result["config"])
        return result

    analysis = {
        "status": "baseline_below_target",
        "target": 1.0,
        "names": arm["trick_names"],
        "baseline_accuracies": [0.5, 1.0],
        "scores": [0.0, 0.0],
        "owners": [0, 0],
        "removed_counts": [0],
        "accuracies": [[0.5, 1.0]],
        "safe_removed": 0,
        "safe_retained": 2,
        "first_failed_removed": 0,
        "later_recovery": False,
        "alignment": {
            "removed_task_counts": [0, 0],
            "retained_task_counts": [2, 0],
            "removed_twin_neurons": 0,
            "fully_removed_twin_classes": 0,
            "partially_pruned_twin_classes": 0,
        },
    }
    monkeypatch.setattr(example.EX18, "main", fake_main)
    monkeypatch.setattr(example, "_analyze_pruning", lambda *args, **kwargs: analysis)
    plot = tmp_path / "pruning.png"
    output = example.main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--pruning-plot-output",
            str(plot),
            "--prune-target",
            "1",
        ]
    )
    assert seen["argv"] == [
        "--smoke",
        "--task-style",
        "temporal-credit",
        "--num-tricks",
        "4",
    ]
    assert output is result
    assert output["neuron_pruning"] is analysis
    assert analysis["checkpoint_index"] == 3
    assert plot.exists()


def test_smoke_entry_point_runs_training_and_pruning(tmp_path):
    result = _load().main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--prune-target",
            "0",
            "--prune-step-fraction",
            "0.5",
            "--plot-output",
            str(tmp_path / "evolution.png"),
            "--pruning-plot-output",
            str(tmp_path / "pruning.png"),
        ]
    )
    pruning = result["neuron_pruning"]
    assert pruning["status"] == "complete"
    assert pruning["fixed_point"]["converged"] is True
    assert pruning["safe_retained"] >= 0
    assert (tmp_path / "evolution.png").exists()
    assert (tmp_path / "pruning.png").exists()
