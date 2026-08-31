# Branch reconciliation

## Objective

Reconcile the BrainTrace repository onto the freshly fetched `origin/main`
without importing obsolete branch history or losing recoverable worktree state.
The publication must leave one local worktree, one local branch (`main`), one
remote branch (`main`), and one remote archive tag.

This change does not alter a public API, model behavior, data schema,
dependency, lint gate, or Paperclip configuration.

## Safety preconditions

The live Paperclip check on 2026-08-30 reported service health `ok`, zero live
runs, fourteen agents, zero enabled heartbeats, and zero enabled routines. One
agent was in an idle error state; it had no active run. Recheck these conditions
before publication and cleanup, and stop if automation becomes active.

The remote was fetched and pruned before creating this reconciliation branch.
The refreshed canonical base is `b3c4569791dea1aecbb31c7c205aebd359373f6a`
(`origin/main`). The six topic heads present in the planning inventory had
already been deleted remotely. No new divergent remote head appeared.

## Inventory

The refreshed local inventory contains:

- 91 local branches;
- 83 registered worktrees;
- 30 dirty worktrees;
- one remote topic head, `main` (plus the `origin/HEAD` symbolic ref).

The recovery manifest records each dirty entry's original two-character Git
status, path, size, and SHA-256. It also records excluded transient files and
exact-copy deduplication. Recovery branch names and commit hashes will be added
after rescue completes.

## Dispositions

| Ref group | Disposition |
| --- | --- |
| Refreshed `origin/main` | Canonical accepted BrainCell release base. |
| `paperclip/bra-36-speedups` | Requalify only the two `TestCoupledTraceBoundedness` loop conversions. Port them manually if the performance gate passes. Do not merge old history or profiling files. |
| Gate 4 and Gate 5 lines | Archive. Preserve the later supervised-loss implementation and accepted non-promotion evidence already on `main`. |
| Lint lines, including dirty BRA-35 and BRA-132 state | Archive. Do not land tooling, configuration, or source rewrites. Preserve the current lint baseline. |
| Retired Example 21 latent-reasoning lines | Archive divergent work and remove ancestor refs after verification. |
| BRA-125 rejected optimization and BRA-2 planning | Archive. Do not land code or the obsolete plan. |
| Refs already reachable from `main` | Record the name-to-SHA mapping, then remove their worktrees and refs after publication and archive verification. |

## Recovery rules

Create one named recovery branch for each worktree containing meaningful dirty
state. Preserve modified and deleted tracked files and untracked source, spec,
evidence, configuration, and OpenSpec files. Hash but do not commit `.err`,
`.payload`, untracked `uv.lock`, cache content, or exact repository copies.

For BRA-73's seven repository-like benchmark trees, preserve each unique delta
blob once and record all duplicates and exact source-tree copies in the
manifest. Fingerprint every dirty worktree again after rescue. Retry once if it
changed; if it changes twice, leave that worktree intact and stop cleanup.

Recovery completed with 4,381 entries across all 30 dirty worktrees. Of these,
228 meaningful entries were committed on 23 recovery branches. The remaining
4,153 entries were hashed and excluded as transient files or exact repository
copies. All 30 post-rescue content fingerprints matched their originals, and
every recovery commit has its original worktree HEAD as its parent.

The complete per-path record is
`docs/evidence/reconciliation/recovery-manifest.json` (SHA-256
`ff0960b50725a18fa27a6a28b6197e239b89f85b34b5ce34837cc24bada00ae2`).
The recovery refs are:

- `e6f0fe84921e8f18dac3e9340fc7cc126e6f0e74` (`paperclip-bra-105-lint-code`)
- `d16b68801e48f90630f7cd442fa9c57e3e011b86` (`paperclip-bra-106-speed-up-code-and-tests`)
- `4fddd42d5f80e6a2ce83993136120c2c33646b0d` (`paperclip-bra-107-update-tasks-and-notify-agent-s`)
- `1c5f3f0f9b2e90433c5660b151ecb8f54e70cc43` (`paperclip-bra-116-lint-code`)
- `55d8fb4e74273dbdccd8edac596b05f7f77f4620` (`paperclip-bra-117-speed-up-code-and-tests`)
- `df24e07d4b36bb4f3fc156b64fe2d256dc900b63` (`paperclip-bra-118-update-tasks-and-notify-agent-s`)
- `bc3a39878789cf842b76497f8f11897001eb23a4` (`paperclip-bra-124-lint-code`)
- `d0b531c41e6cce38a39822b42c49282104c53331` (`paperclip-bra-125-speed-up-code-and-tests`)
- `623d0f26d347b37a944a9c5c73b88dd30895e6ec` (`paperclip-bra-126-update-tasks-and-notify-agent-s`)
- `bad3e0059737eaf414b587822eb8ff290db3888b` (`paperclip-bra-132-lint-code`)
- `1b5b24b9b2df6a00f0029f0575ec42bf17ac5fec` (`paperclip-bra-133-update-tasks-and-notify-agent-s`)
- `ca5900400795935de30632a3a68f6d086c832adb` (`paperclip-bra-35-lint-code`)
- `882ca14817cc2a7313540d91d6caa4af6b01d515` (`paperclip-bra-72-lint-code`)
- `685a193ea7769dfd76dd0b36c5568ae15cba15cf` (`paperclip-bra-73-speed-up-code-and-tests`)
- `d512fbce1186651fd4fd7c337373ea51a3787076` (`paperclip-bra-74-update-tasks-and-notify-agent-s`)
- `f3ef87ae55bd5e379d9aed0b2d53bd0e5fbb6592` (`paperclip-bra-85-lint-code`)
- `f15fa78b036b78132a607fc9c9ca30881ada3dfd` (`paperclip-bra-86-speed-up-code-and-tests`)
- `387f8e477e54d0dc8a2af386c9175c294701464d` (`paperclip-bra-87-update-tasks-and-notify-agent-s`)
- `0cee79472050d4bd0839f87e9890903171d0953a` (`paperclip-bra-89-autonomy-stewardship`)
- `e856dbdfed49576c5194a36b704920447a5d013c` (`paperclip-bra-96-lint-code`)
- `8284e15ff289e01734fcd7775314bfb5f9a880bd` (`paperclip-bra-97-speed-up-code-and-tests`)
- `51e50e9016a2e81e087946d1bc7ffcd5aedbfe99` (`ex21-cumulative-48`)
- `e3cd153e0cecfaad7dfa38457e03ef93cd4e4878` (`fix-example21-arc-step-loss`)

## Performance gate

Run one warm-up and then three alternating fresh-process baseline and candidate
runs of both affected boundedness tests. Accept the candidate only when all runs
pass, candidate median wall time improves by at least 15 percent, and the two
ranges do not overlap. If it fails, archive the branch without a code change.

After qualification, run both affected tests individually and together, then
the complete sibling `d_rtrl_test.py`.

The gate passed on the intended Python 3.14.6 virtualenv with JAX 0.11.0 and
BrainState 0.5.3. All eight warm-up and measured pytest processes passed. The
three measured baseline runs were 14.647738, 15.701980, and 16.066780 seconds.
The candidate runs were 7.523232, 7.860104, and 7.879359 seconds. Candidate
median wall time improved by 49.94 percent, and the ranges did not overlap.

The raw process outputs and the correction of an initial PowerShell summary
aggregation error are recorded in
`docs/evidence/reconciliation/bra36-performance.json` (SHA-256
`c0aad49d654f1fa933e42c2493cba3cf21f653a0e8e3ddaa7473a341630ee190`).
The two affected tests passed individually and together. The complete sibling
suite passed with 83 tests in 46.62 seconds.

## CI typecheck remediation amendment

The first post-publication CI run passed the four JAX matrix jobs and the
examples job. Its `typecheck_and_build` job stopped at the same three mypy
diagnostics reproduced on untouched `origin/main`:

- `braintrace/_quant/_turboquant.py:139`: the annotated return type of
  `RandomState.split_key(2)` includes the scalar-return overload, so mypy does
  not permit two-value unpacking without narrowing;
- `braintrace/nn/_situ.py:69`: BrainState hides `Module.__init__` from type
  checkers even though the runtime constructor accepts `name`;
- `braintrace/nn/_gated.py:84`: the same hidden-constructor diagnostic.

This amendment authorizes one narrowly scoped follow-up that corrects only
those three diagnostics while preserving runtime behavior. It may add
co-located regression tests that verify the affected constructors and random
key split still behave as before. It may also update this specification and
the reconciliation evidence with the resulting gate and CI status.

The follow-up must not import any archived lint-branch change, alter mypy or CI
configuration, add a type-check suppression, upgrade tooling, perform a
formatting or lint sweep, or make an unrelated refactor. It must not alter a
public API, model behavior, data schema, dependency, lint gate, or Paperclip
configuration. If any diagnostic cannot be resolved within those constraints,
stop and request a new decision instead of broadening the change.

Acceptance requires the three diagnostics to be absent with no new mypy
diagnostic, the focused sibling tests to pass, every local gate below to be
rerun, and every required publication CI job to pass. Archive creation and
cleanup remain blocked until that fully green CI result exists.

## Local acceptance gates

Run all configured local gates without deselection:

```text
pytest braintrace/ -n auto --durations=15 --cov=braintrace --cov-report=term-missing
pytest examples/ -n auto --dist=loadgroup --durations=15
pytest repo_conventions_test.py -q
python -m mypy braintrace
python -m build
git diff --check
```

Package coverage must exceed 90 percent. Verify `py.typed` in both the wheel and
sdist. The integration worktree must be clean with no new failures.

## Publication and archive

Push the reconciliation tip directly as a fast-forward update to `origin/main`
without force. If remote `main` moves, rebuild on the new tip and rerun all
gates. Require the four JAX matrix jobs, examples, typecheck, and build jobs to
pass.

Create annotated tag `archive/braintrace-pre-reconcile-2026-08-30`. Its target
is a synthetic commit whose first parent is final `main` and whose remaining
parents are every unique divergent tip and recovery commit. Verify the tag from
a fresh temporary repository before deleting any worktree or ref.

## Final evidence

Recovery and performance qualification completed as recorded above. Local gate
results on Python 3.14.6 were:

- Package suite: 3,163 passed, zero deselected, 286 warnings, and 95 percent
  total coverage in 356.23 seconds.
- Examples suite: 576 passed, five skipped, zero deselected, and one failed in
  137.05 seconds. The failure was
  `test_merge_cli_uses_measured_files_and_arm_cli_requires_parent`, where the
  local Windows process-RSS probe returned `None`. The same node failed
  identically when rerun alone on untouched `origin/main`, so this is a
  confirmed baseline failure and not a reconciliation regression.
- Repository conventions: three passed.
- Mypy: three baseline errors in `_quant/_turboquant.py`, `nn/_situ.py`, and
  `nn/_gated.py`. Untouched `origin/main` produced the same three diagnostics.
- Build: wheel and sdist succeeded. Both archives contain
  `braintrace/py.typed`.
- `git diff --check`: passed.

Thus the local result satisfies the no-new-failure and coverage requirements,
but it is not represented as fully green: the examples and mypy commands retain
their pre-existing failures. Publication CI must still pass every required job
before archive cleanup can begin.

Pending publication CI, archive verification, and exact-SHA cleanup. This
section will be updated with literal hashes and final topology before closeout.
