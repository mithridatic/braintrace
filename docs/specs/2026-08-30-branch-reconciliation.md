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

## Performance gate

Run one warm-up and then three alternating fresh-process baseline and candidate
runs of both affected boundedness tests. Accept the candidate only when all runs
pass, candidate median wall time improves by at least 15 percent, and the two
ranges do not overlap. If it fails, archive the branch without a code change.

After qualification, run both affected tests individually and together, then
the complete sibling `d_rtrl_test.py`.

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

Pending recovery, performance qualification, local gates, publication CI,
archive verification, and exact-SHA cleanup. This section will be updated with
literal results and hashes before final closeout.
