"""Tests for the multi-task structural evolution example."""

import importlib.util
import pathlib

import numpy as np

EXAMPLE = pathlib.Path(__file__).resolve().with_name("18-structural-evolution.py")


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_structural_evolution", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_valid_csr(example, rows, cols, n_rec):
    indptr = example._indptr_from_rows(rows, n_rec)
    assert np.all(np.diff(indptr) >= 0)
    assert indptr[-1] == rows.size
    assert np.all(cols >= 0) and np.all(cols < n_rec)
    assert np.all(rows >= 0) and np.all(rows < n_rec)
    assert np.all(rows != cols)
    for row in np.unique(rows):
        row_cols = cols[rows == row]
        assert np.unique(row_cols).size == row_cols.size


def test_prune_removes_weakest_edges_and_preserves_budget():
    example = _load()
    values = np.asarray([0.9, -0.05, 0.3, 0.01, -0.7, 0.2, -0.02, 0.5])

    survivors = example._prune_survivors(values, 3)

    assert survivors.tolist() == [0, 2, 4, 5, 7]
    removed = np.setdiff1d(np.arange(values.size), survivors)
    assert np.all(np.abs(values[removed]) <= np.min(np.abs(values[survivors])))

    n_rec, n_edges = 16, 12
    rng = np.random.default_rng(0)
    rows, cols = example._sample_irregular_topology(n_rec, n_edges, rng)
    budget_values = np.asarray(
        [0.9, -0.05, 0.3, 0.01, -0.7, 0.2, -0.02, 0.5, 0.4, -0.6, 0.8, -0.1]
    )
    prune_count = 3
    survivors = example._prune_survivors(budget_values, prune_count)
    new_rows, new_cols = example._respawn_endpoints(
        n_rec, prune_count, np.zeros(n_rec), 1e-3,
        rows[survivors], cols[survivors], rng,
    )
    all_rows = np.concatenate([rows[survivors], new_rows])
    all_cols = np.concatenate([cols[survivors], new_cols])
    all_values = np.concatenate(
        [budget_values[survivors], np.full(prune_count, 9.0)]
    )
    all_rows, all_cols, (all_values,) = example._sort_edges(
        all_rows, all_cols, all_values
    )

    assert all_rows.size == n_edges
    csr = example._to_csr(all_rows, all_cols, all_values, n_rec, None)
    assert csr.data.shape[0] == n_edges
    kept_pairs = set(zip(rows[survivors].tolist(), cols[survivors].tolist()))
    final_pairs = list(zip(all_rows.tolist(), all_cols.tolist()))
    for pair, value in zip(final_pairs, all_values.tolist()):
        if pair in kept_pairs:
            expected = budget_values[(rows == pair[0]) & (cols == pair[1])]
            assert value == expected[0]
        else:
            assert value == 9.0


def test_respawn_produces_valid_csr_and_respects_activity_floor():
    example = _load()
    n_rec = 32
    rng = np.random.default_rng(1)
    rows, cols = example._sample_irregular_topology(n_rec, 24, rng)

    new_rows, new_cols = example._respawn_endpoints(
        n_rec, 8, np.zeros(n_rec), 1e-3, rows, cols, rng
    )

    all_rows = np.concatenate([rows, new_rows])
    all_cols = np.concatenate([cols, new_cols])
    all_rows, all_cols, _ = example._sort_edges(all_rows, all_cols)
    _assert_valid_csr(example, all_rows, all_cols, n_rec)

    hot = np.full(n_rec, 1e-6)
    hot[0] = 1.0
    biased_rows, biased_cols = example._respawn_endpoints(
        n_rec, 16, hot, 1e-3, rows, cols, rng
    )
    touches_hot = (biased_rows == 0) | (biased_cols == 0)
    assert np.mean(touches_hot) > 0.5


def test_gradient_endpoints_follow_marginals_and_avoid_existing():
    example = _load()
    n_rec = 32
    rng = np.random.default_rng(2)

    # Zero mass everywhere: floor-only uniform draw, still a valid CSR.
    rows, cols = example._sample_irregular_topology(n_rec, 24, rng)
    new_rows, new_cols = example._gradient_endpoints(
        n_rec, 8, np.zeros(rows.size), rows, cols, 1e-3, rng
    )
    all_rows = np.concatenate([rows, new_rows])
    all_cols = np.concatenate([cols, new_cols])
    all_rows, all_cols, _ = example._sort_edges(all_rows, all_cols)
    _assert_valid_csr(example, all_rows, all_cols, n_rec)

    # One edge (3 <- 0) carrying all the mass: demand concentrates on row 3,
    # supply on col 0, and (3, 0) is taken — draws must land on its free
    # neighbors almost surely.
    rows = np.array([3])
    cols = np.array([0])
    mass = np.array([1.0])
    hot_rows, hot_cols = example._gradient_endpoints(
        n_rec, 16, mass, rows, cols, 1e-6, rng
    )
    touches = (hot_rows == 3) | (hot_cols == 0)
    assert np.mean(touches) > 0.9
    assert not np.any((hot_rows == 3) & (hot_cols == 0))
    assert not np.any(hot_rows == hot_cols)


def test_draw_free_pairs_falls_back_to_uniform_when_weights_saturate():
    example = _load()
    n_rec = 16
    rng = np.random.default_rng(7)

    # All mass on the self-loop (0, 0): every weighted draw is invalid, so
    # only the uniform fallback can satisfy the request.
    row_weight = np.zeros(n_rec)
    col_weight = np.zeros(n_rec)
    row_weight[0] = 1.0
    col_weight[0] = 1.0
    flat = example._draw_free_pairs(
        n_rec, 6, row_weight, col_weight, np.array([1]), np.array([2]), rng
    )
    rows, cols = divmod(flat, n_rec)
    assert flat.size == 6
    assert np.unique(flat).size == 6
    assert not np.any(rows == cols)
    assert not np.any((rows == 1) & (cols == 2))
    assert not np.any((rows == 0) & (cols == 0))


def test_attribution_classification_applies_sixty_percent_rule():
    example = _load()
    fetch_mass = np.asarray([7.0, 3.0, 0.0, 5.0, 1.0, 10.0])
    roll_mass = np.asarray([3.0, 7.0, 0.0, 5.0, 9.0, 0.0])

    labels = example._classify_attribution(fetch_mass, roll_mass, threshold=0.6)

    assert labels.tolist() == [
        example.FETCH,
        example.ROLL_OVER,
        example.SHARED,
        example.SHARED,
        example.ROLL_OVER,
        example.FETCH,
    ]


def test_adaptive_controller_grows_below_target_and_shrinks_above():
    example = _load()
    config = example._EvolutionConfig(
        n_rec=32, n_edges=32, max_edges=64, min_edges=16
    )

    # Below target: grow by the factor, at least one edge, never past the cap.
    assert example._next_budget(32, 0.50, config) == 35
    assert example._next_budget(48, 0.50, config) == 53
    assert example._next_budget(64, 0.50, config) == 64
    # At or above target: shrink by 10%, never below the floor.
    assert example._next_budget(63, 0.95, config) == 57
    assert example._next_budget(17, 1.00, config) == 16
    assert example._next_budget(16, 1.00, config) == 16
    # A budget already under the floor holds rather than growing via shrink.
    assert example._next_budget(12, 1.00, config) == 12


def test_adaptive_budget_three_round_forced_growth_and_shrink(tmp_path):
    example = _load()
    base = dict(
        n_rec=32,
        n_edges=32,
        n_rounds=3,
        trials_per_round=4,
        eval_trials_per_task=2,
        rate_probe_trials=2,
        max_edges=64,
        min_edges=16,
    )

    grow = example.run(
        example._EvolutionConfig(**base, target_accuracy=1.01),
        tmp_path / "grow.png",
    )
    grow_counts = grow["evolve"]["edge_counts"]
    assert grow_counts == [32, 35, 38, 42]
    assert [event["kind"] for event in grow["evolve"]["events"]] == ["grow"] * 3
    assert grow["control"]["edge_counts"] == [32, 32, 32, 32]
    assert grow["evolve"]["attribution"].shape == (grow_counts[-1],)
    _assert_valid_csr(
        example, grow["evolve"]["rows"], grow["evolve"]["cols"], base["n_rec"]
    )
    assert (tmp_path / "grow.png").exists()

    shrink = example.run(
        example._EvolutionConfig(**base, target_accuracy=0.0),
        tmp_path / "shrink.png",
    )
    shrink_counts = shrink["evolve"]["edge_counts"]
    assert shrink_counts == [32, 29, 26, 23]
    assert all(
        event["kind"] == "shrink" for event in shrink["evolve"]["events"]
    )
    assert min(shrink_counts) >= base["min_edges"]
    assert shrink["evolve"]["attribution"].shape == (shrink_counts[-1],)
    _assert_valid_csr(
        example, shrink["evolve"]["rows"], shrink["evolve"]["cols"], base["n_rec"]
    )


def test_adaptive_budget_respects_max_growth_events(tmp_path):
    example = _load()
    config = example._EvolutionConfig(
        n_rec=32,
        n_edges=32,
        n_rounds=3,
        trials_per_round=4,
        eval_trials_per_task=2,
        rate_probe_trials=2,
        max_edges=64,
        min_edges=16,
        target_accuracy=1.01,  # demand stays high: would grow every round
        max_growth_events=1,
    )
    result = example.run(config, tmp_path / "capped.png")
    counts = result["evolve"]["edge_counts"]
    assert counts == [32, 35, 35, 35]
    kinds = [event["kind"] for event in result["evolve"]["events"]]
    assert kinds == ["grow"]
    _assert_valid_csr(
        example, result["evolve"]["rows"], result["evolve"]["cols"], 32
    )


def test_temporal_credit_smoke_round_runs(tmp_path):
    example = _load()
    config = example._EvolutionConfig(
        task_style="temporal-credit",
        n_rec=32,
        n_edges=32,
        n_rounds=1,
        trials_per_round=8,
        eval_trials_per_task=2,
        rate_probe_trials=2,
        min_edges=16,
        max_edges=64,
    )
    layout = example._layout(config)
    assert (layout.n_in, layout.n_step) == (17, 30)
    assert layout.response_start == {example.FETCH: 6, example.ROLL_OVER: 26}
    assert layout.go_channel == 16

    result = example.run(config, tmp_path / "temporal.png")

    for arm in (result["evolve"], result["control"]):
        assert np.all(np.isfinite(arm["round_losses"]))
        for key in ("fetch_accuracy", "roll_over_accuracy"):
            accuracies = np.asarray(arm[key])
            assert accuracies.shape == (config.n_rounds + 1,)
            assert np.all(accuracies >= 0.0) and np.all(accuracies <= 1.0)
        assert arm["attribution"].shape == (arm["edge_counts"][-1],)
        _assert_valid_csr(
            example, arm["rows"], arm["cols"], config.n_rec
        )
    evolve_counts = result["evolve"]["edge_counts"]
    assert max(evolve_counts) <= config.max_edges
    assert min(evolve_counts) >= min(config.min_edges, config.n_edges)
    assert (tmp_path / "temporal.png").exists()


def test_attribution_classification_generalizes_to_n_tasks():
    example = _load()
    masses = [
        np.asarray([7.0, 1.0, 2.0, 5.0, 0.0]),
        np.asarray([1.0, 7.0, 2.0, 5.0, 0.0]),
        np.asarray([1.0, 1.0, 2.0, 0.0, 9.0]),
        np.asarray([1.0, 1.0, 2.0, 0.0, 1.0]),
    ]

    labels = example._classify_attribution(*masses, threshold=0.6)

    # edge 0: fetch 7/10; edge 1: roll over 7/10; edge 2: 2/8 each (shared);
    # edge 3: fetch 5/10 and roll over 5/10 tie (shared); edge 4: sit 9/10.
    assert labels.tolist() == [0, 1, 4, 4, 2]


def test_temporal_credit_four_trick_smoke_runs(tmp_path):
    example = _load()
    config = example._EvolutionConfig(
        task_style="temporal-credit",
        num_tricks=4,
        n_rec=32,
        n_edges=32,
        n_rounds=1,
        trials_per_round=8,
        eval_trials_per_task=2,
        rate_probe_trials=4,
        min_edges=16,
        max_edges=64,
    )
    layout = example._layout(config)
    assert (layout.n_in, layout.n_step) == (33, 31)
    assert layout.response_start == {0: 6, 1: 13, 2: 20, 3: 27}
    assert layout.go_channel == 32
    starts = sorted(layout.response_start.values())
    for first, second in zip(starts, starts[1:]):
        assert first + layout.response_ticks <= second
    assert starts[-1] <= example._TEMPORAL_MAX_RESPONSE_START

    result = example.run(config, tmp_path / "four.png")

    for arm in (result["evolve"], result["control"]):
        assert np.all(np.isfinite(arm["round_losses"]))
        assert len(arm["accuracies"]) == 4
        for history in arm["accuracies"]:
            values = np.asarray(history)
            assert values.shape == (config.n_rounds + 1,)
            assert np.all(values >= 0.0) and np.all(values <= 1.0)
        assert arm["attribution"].shape == (arm["edge_counts"][-1],)
        assert arm["task_mass"].shape == (config.num_tricks, arm["rows"].size)
        assert len(arm["split"]) == 5
        _assert_valid_csr(example, arm["rows"], arm["cols"], config.n_rec)
    evolve_counts = result["evolve"]["edge_counts"]
    assert max(evolve_counts) <= config.max_edges
    assert min(evolve_counts) >= min(config.min_edges, config.n_edges)
    assert (tmp_path / "four.png").exists()


def _context_config(example):
    return example._EvolutionConfig(
        task_style="context",
        num_tricks=4,
        n_rec=32,
        n_edges=32,
        n_rounds=1,
        trials_per_round=8,
        eval_trials_per_task=2,
        rate_probe_trials=4,
        min_edges=16,
        max_edges=64,
    )


def test_context_layout_is_valid_and_bounded():
    example = _load()
    config = _context_config(example)
    layout = example._layout(config)

    assert layout.n_in == 25
    assert layout.go_channel == 24
    assert layout.response_start == {0: 12, 1: 16, 2: 20, 3: 24}
    assert layout.n_step == 28
    starts = sorted(layout.response_start.values())
    for first, second in zip(starts, starts[1:]):
        assert first + layout.response_ticks <= second
    assert starts[-1] <= example._TEMPORAL_MAX_RESPONSE_START


def test_context_rate_templates_differ_only_in_x_channels():
    example = _load()
    config = _context_config(example)
    a_alone = example._rate_template(0, config)
    a_context = example._rate_template(1, config)
    b_alone = example._rate_template(2, config)
    b_context = example._rate_template(3, config)

    # Same cue, different context: identical outside the X channels (16-23).
    assert np.array_equal(a_alone[:, :16], a_context[:, :16])
    assert np.array_equal(a_alone[:, 24:], a_context[:, 24:])
    assert not np.array_equal(a_alone, a_context)
    assert np.all(a_context[:4, 16:24] > 0)
    assert np.all(a_alone[:, 16:24] == 0.0)
    # Different cue, no context: identical outside the cue channels (0-15).
    assert np.array_equal(a_alone[:, 16:], b_alone[:, 16:])
    assert not np.array_equal(a_alone[:, :16], b_alone[:, :16])
    # Go inputs are label-independent across all four conditions.
    go = a_alone[:, 24]
    for template in (a_context, b_alone, b_context):
        assert np.array_equal(template[:, 24], go)


def test_context_smoke_run(tmp_path):
    example = _load()
    config = _context_config(example)

    result = example.run(config, tmp_path / "context.png")

    for arm in (result["evolve"], result["control"]):
        assert np.all(np.isfinite(arm["round_losses"]))
        assert len(arm["accuracies"]) == 4
        for history in arm["accuracies"]:
            values = np.asarray(history)
            assert values.shape == (config.n_rounds + 1,)
            assert np.all(values >= 0.0) and np.all(values <= 1.0)
        assert arm["attribution"].shape == (arm["edge_counts"][-1],)
        _assert_valid_csr(example, arm["rows"], arm["cols"], config.n_rec)
    assert result["evolve"]["trick_names"] == list(example._CONTEXT_NAMES)
    evolve_counts = result["evolve"]["edge_counts"]
    assert max(evolve_counts) <= config.max_edges
    assert min(evolve_counts) >= min(config.min_edges, config.n_edges)
    assert (tmp_path / "context.png").exists()


def test_smoke_run_completes_with_finite_losses_and_valid_accuracies(tmp_path):
    example = _load()
    plot_path = tmp_path / "smoke.png"

    result = example.main(["--smoke", "--plot-output", str(plot_path)])

    config = result["config"]
    assert (config.n_rec, config.n_edges, config.n_rounds) == (32, 32, 1)
    assert config.trials_per_round == 8
    for arm in (result["evolve"], result["control"]):
        assert np.all(np.isfinite(arm["round_losses"]))
        for key in ("fetch_accuracy", "roll_over_accuracy"):
            accuracies = np.asarray(arm[key])
            assert accuracies.shape == (config.n_rounds + 1,)
            assert np.all(accuracies >= 0.0) and np.all(accuracies <= 1.0)
        counts = arm["edge_counts"]
        assert len(counts) == config.n_rounds + 1
        assert counts[0] == config.n_edges
        assert arm["attribution"].shape == (counts[-1],)
        assert arm["rows"].shape == (counts[-1],)
    evolve_counts = result["evolve"]["edge_counts"]
    assert max(evolve_counts) <= config.max_edges
    assert min(evolve_counts) >= min(config.min_edges, config.n_edges)
    assert len(set(result["control"]["edge_counts"])) == 1
    assert plot_path.exists()
    assert "plain English" in result["report"]


def test_optimizer_moments_survive_rebuild():
    import brainstate
    import brainunit as u

    example = _load()
    config = example._EvolutionConfig.smoke()
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example._initial_experiment(config)
        example._train_round(experiment, config, 0)

        rec_opt = experiment.optimizers["recurrent"]
        donor_adam = next(
            s for s in rec_opt.opt_state.value if hasattr(s, "mu")
        )
        donor_mu = np.asarray(
            u.get_mantissa(donor_adam.mu[experiment.rec_key]["weight"]),
            dtype=np.float64,
        )
        donor_step = int(rec_opt.step_count.value)
        readout_mu = _readout_mu(experiment)
        old_rows, old_cols, _ = example._current_topology(experiment)

        rows, cols, values, mass = example._shrink_edges(experiment, 4)
        rebuilt = example._rebuild_experiment(
            config, experiment, rows, cols, values, mass
        )

        # Recurrent mu lands on the same (row, col) pairs after the rebuild.
        expected = {
            (int(r), int(c)): m
            for r, c, m in zip(old_rows, old_cols, donor_mu)
        }
        rebuilt_adam = next(
            s
            for s in rebuilt.optimizers["recurrent"].opt_state.value
            if hasattr(s, "mu")
        )
        new_mu = np.asarray(
            u.get_mantissa(rebuilt_adam.mu[rebuilt.rec_key]["weight"]),
            dtype=np.float64,
        )
        new_rows, new_cols, _ = example._current_topology(rebuilt)
        assert new_mu.shape == (new_rows.shape[0],)
        for j, (r, c) in enumerate(zip(new_rows, new_cols)):
            assert np.isclose(new_mu[j], expected[(int(r), int(c))])
        # Step counts and dense-group moments carry untouched.
        assert int(rebuilt.optimizers["recurrent"].step_count.value) == donor_step
        assert np.array_equal(readout_mu, _readout_mu(rebuilt))


def _readout_mu(experiment):
    import brainunit as u

    adam = next(
        s for s in experiment.optimizers["readout"].opt_state.value
        if hasattr(s, "mu")
    )
    key = next(iter(adam.mu))
    return np.asarray(u.get_mantissa(adam.mu[key]), dtype=np.float64)
