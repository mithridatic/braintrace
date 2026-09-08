# H01 population programme: SP9.4 merge readiness (2026-09-07)

Branch `feat/h01-braincell`, tip recorded as `ede78a5` (the final programme
merge). The jobs below started while the worktree was at `15e270c`; sibling
sessions advanced the branch (`49c434e`, `ede78a5`) during the runs, so the
files under test were the working tree as it stood between those commits.
No simulation was run. Read-only apart from this file.

Interpreter: `.cache\validation\Scripts\python.exe` (CPython 3.13.14) with
`PYTHONPATH` set to the worktree root; `braintrace.__file__` resolved to this
worktree. Every pytest invocation used the Abseil workaround from
`h01-verified-network-validation.md`:

```
python -u -c "import braintrace, braincell; import pytest; raise SystemExit(pytest.main([...]))"
```

Jobs ran as detached `pwsh` processes with redirected logs; the four jobs ran
concurrently on the same host, so wall clocks include contention.

## Summary

| # | Suite | pytest args | Summary line | Wall clock | Exit |
|---|-------|-------------|--------------|-----------|------|
| 1 | `braintrace/datasets` (+coverage) | `braintrace/datasets --cov=braintrace.datasets.h01_network --cov=...h01_cell_types --cov=...h01_ei_profiles --cov=...h01_ei_cell --cov=...h01_l2_cell --cov=...h01_network_init --cov=..._h01_ei_parameters --cov=...h01_construction --cov-report=term-missing -q -p no:cacheprovider` | `410 passed in 752.70s (0:12:32)` | 12:49 | 0 |
| 2a | `examples` (first attempt) | `examples -m "not diagnostic and not slow" -q -p no:cacheprovider` | `3 errors in 5.41s` (collection interrupted) | 0:18 | 2 |
| 2b | `examples` (rerun) | as 2a plus `--ignore` of the three `temporal_benchmark_*_test.py` files | `30 failed, 691 passed, 5 skipped, 25 warnings in 1040.27s (0:17:20)` | 17:32 | 1 |
| 3 | `repo_conventions_test.py setup_test.py` | `-q -p no:cacheprovider` | `1 failed, 3 passed in 3.85s` | 0:16 | 1 |
| 4 | `docs/evidence/h01_*_test.py` (71 files, explicit list, `docs/evidence` on `sys.path`) | `-q -p no:cacheprovider` | `2 failed, 798 passed, 4 skipped in 203.37s (0:03:23)` | 3:37 | 1 |

Skipped in suite 2: `-m "not slow"` deselected the `slow` marker on top of the
default `not diagnostic`; no GPU marker exists in `pyproject.toml`. The rerun
also ignored `examples/pp_prop/temporal_benchmark_config_test.py`,
`temporal_benchmark_data_test.py`, `temporal_benchmark_topology_test.py`,
which fail at import (`ModuleNotFoundError: No module named 'hypothesis'`;
the validation venv lacks the `testing` extra). Suite 4 was restricted to the
explicit `h01_*_test.py` file list from the start (not by `-k`); no
collection of unrelated evidence tests was attempted.

## Coverage (suite 1, modules changed by the programme; rule: >= 90 %)

| Module | Stmts | Miss | Cover | Missing | 90 % rule |
|--------|-------|------|-------|---------|-----------|
| `braintrace/datasets/_h01_ei_parameters.py` | 9 | 0 | 100% | | pass |
| `braintrace/datasets/h01_cell_types.py` | 51 | 0 | 100% | | pass |
| `braintrace/datasets/h01_construction.py` | 207 | 9 | 96% | 35, 76, 92-95, 210, 285-286 | pass |
| `braintrace/datasets/h01_ei_cell.py` | 90 | 2 | 98% | 50, 124 | pass |
| `braintrace/datasets/h01_ei_profiles.py` | 50 | 0 | 100% | | pass |
| `braintrace/datasets/h01_l2_cell.py` | 31 | 0 | 100% | | pass |
| `braintrace/datasets/h01_network.py` | 179 | 0 | 100% | | pass |
| `braintrace/datasets/h01_network_init.py` | 43 | 0 | 100% | | pass |
| TOTAL | 660 | 11 | 98% | | pass |

## Failures (verbatim; not fixed)

### Suite 3

`setup_test.py::test_source_distribution_excludes_scratch_directories`
```
archives = sorted((pathlib.Path(__file__).parent / "dist").glob("*.tar.gz"))
>       assert archives, "Build an sdist before running the packaging test. Update the fixture or expected result to satisfy this assertion."
E       AssertionError: Build an sdist before running the packaging test. Update the fixture or expected result to satisfy this assertion.
E       assert []
```
No `dist/` directory exists in the worktree; precondition, not a code regression.

### Suite 4

`docs/evidence/h01_ie_functional_inhibition_test.py::test_manifest_commands_parse_and_match_the_driver`
```
>       assert manifest["cap"] == 6 and manifest["halving_cap"] == 2 and manifest["prior_evaluations"] == 0
E       assert (6 == 6 and 2 == 2 and 1 == 0)
docs\evidence\h01_ie_functional_inhibition_test.py:185: AssertionError
```
`h01-ie-inhibition-manifest.json` records `prior_evaluations` = 1; the test pins 0.

`docs/evidence/h01_pv_inactivation_equilibrium_test.py::EquilibriumObservationTest::test_uses_accessible_mechanism_state`
```
>       runpy.run_path(str(Path(__file__).with_name("h01_pv_inactivation_equilibrium.py")))
>   from neuron import h
E   ModuleNotFoundError: No module named 'neuron'
docs\evidence\h01_pv_inactivation_equilibrium.py:7: ModuleNotFoundError
```
NEURON is not installed in the validation venv.

### Suite 2 (30 failed; none in H01 files)

All three `examples/h01_*_test.py` modules passed. Every failure is an
environment gap of the validation venv (no `numba`, no `scikit-learn`, no
examples extra), grouped by final error line:

`ModuleNotFoundError: No module named 'numba'` (direct import):
- `examples/compile_modes_test.py::test_snn_cell_compiles_and_runs_in_both_modes[False-gif]`
- `examples/compile_modes_test.py::test_snn_cell_compiles_and_runs_in_both_modes[True-gif]`
- `examples/compile_modes_test.py::test_gif_neuron_init_state_accepts_batch_size`
- `examples/pp_prop/smoke_test.py::test_example_runs[03-neurons-gif-working-memory.py]`
- `examples/smoke_test.py::test_example_runs[001-gif-snn-for-dms.py-tiny_kwargs0]`

`brainevent.KernelCompilationError: Backend 'numba' failed to construct a kernel for primitive 'csrmv' on platform 'cpu': ModuleNotFoundError: No module named 'numba'. Available backend(s) for platform 'cpu': ['numba', 'jax_raw']. Switch with backend='<name>' on this call, or csrmv.set_default('cpu', '<name>') / brainevent.set_backend('cpu', '<name>') to change the default.` (primitive `csrmm` where marked):
- `examples/pp_prop/21-braincell-arc_test.py::test_cli_smoke_writes_report` (csrmm)
- `examples/pp_prop/21-braincell-arc_test.py::test_finite_difference_fixture_has_declared_tolerance` (csrmm)
- `examples/pp_prop/21-braincell-arc_test.py::test_false_advance_preserves_biological_state_bitwise`
- `examples/pp_prop/21-braincell-arc_test.py::test_compiled_event_sequence_freezes_padding_and_returns_outputs`
- `examples/pp_prop/21-braincell-arc_test.py::test_padding_does_not_change_a_valid_sequence_result`
- `examples/pp_prop/21-braincell-arc_test.py::test_scoring_sequence_matches_full_history_with_masked_prefix`
- `examples/pp_prop/21-braincell-arc_test.py::test_matched_integration_and_decoder_boundary_are_explicit`
- `examples/pp_prop/21-braincell-arc_test.py::test_pp_prop_sequence_skips_false_events`
- `examples/pp_prop/21-braincell-arc_test.py::test_event_sequence_uses_candidate_neuron_count_for_false_events`
- `examples/pp_prop/21-braincell-arc_test.py::test_pp_prop_sequence_preserves_eligibility_across_interspersed_padding`
- `examples/pp_prop/21-braincell-arc_test.py::test_real_compiled_episode_updates_grouped_parameters_and_param_states`
- `examples/pp_prop/example21_structural_test.py::test_real_pp_prop_update_then_structural_addition_preserves_dale_signs`
- `examples/pp_prop/sparse_gradient_test.py::test_sparse_backends_support_nonzero_batched_pp_prop_gradients[None]` (csrmm)

`RuntimeError: Example 15 requires scikit-learn; install the BrainTrace examples extra`:
- `examples/pp_prop/15-sparse-temporal-learning_test.py::test_sparse_digit_learning_beats_chance_across_seeds`
- `examples/pp_prop/15-sparse-temporal-learning_test.py::test_digit_split_is_fixed_and_stratified`

`RuntimeError: Poisson digit examples require the BrainTrace examples extra. Provide the required value for Poisson digit examples.`:
- `examples/pp_prop/shared_data_test.py::test_make_poisson_mnist_shape_and_labels`
- `examples/pp_prop/smoke_test.py::test_example_runs[04-neurons-coba-ei-rsnn.py]`
- `examples/pp_prop/smoke_test.py::test_example_runs[09-operator-sparse.py]`
- `examples/pp_prop/smoke_test.py::test_example_runs[10-operator-lora.py]`
- `examples/pp_prop/smoke_test.py::test_example_runs[11-operator-conv.py]`
- `examples/pp_prop/smoke_test.py::test_example_runs[12-classification-neuromorphic.py]`
- `examples/pp_prop/sparse_operator_test.py::test_sparse_example_trains_with_jax_raw_backend`

Subprocess of `examples/pp_prop/16-configurable-sparse-benchmark.py` (child
stderr is not captured in the log; same venv, so the numba gap is the likely
cause but is unverified):
- `examples/pp_prop/configurable_sparse_benchmark_integration_test.py::test_tiny_supervised_worker_emits_learning_schema`
  `E subprocess.CalledProcessError: Command '[...python.exe, ...\examples\pp_prop\16-configurable-sparse-benchmark.py, '--mode', 'fixed-work', '--device', 'c...]' returned non-zero exit status 1`
- `examples/pp_prop/configurable_sparse_benchmark_integration_test.py::test_the_host_backend_can_be_pinned_over_an_inherited_platform`
  `E assert 1 == 0` / `E +  where 1 = CompletedProcess(args=[...], stdout='{... "scope": "cpu_process_tree_rss" }, "schema_version": 2, "status": "failed" }', stderr='').returncode`

## Reading

- The programme's own surface is green: 410/410 datasets tests, 98 % line
  coverage with every changed module at or above 96 %, and all
  `examples/h01_*` tests pass.
- Three failures touch H01 evidence or packaging preconditions and need a
  decision before merge: the stale `prior_evaluations` pin in
  `h01_ie_functional_inhibition_test.py`, the NEURON dependency of
  `h01_pv_inactivation_equilibrium_test.py`, and the missing sdist for
  `setup_test.py`.
- The 30 `examples` failures and the 3 collection errors are dependency gaps
  of `.cache/validation` (`numba`, `scikit-learn`, `hypothesis`, examples
  extra), not regressions attributable to the branch; a run in a venv with the
  `testing` extra installed would settle that.

Logs: `.cache/merge-readiness/{datasets,examples,examples2,conventions,evidence}.log` (untracked).
