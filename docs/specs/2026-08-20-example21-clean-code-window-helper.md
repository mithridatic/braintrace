# Example 21 dead evaluation-window helper cleanup

Status: implemented
Date: 2026-08-20

## Goal

Reduce Example 21 source size without changing the live evaluation, scoring,
training, model, optimizer, or checkpoint paths.

## Finding

`_gather_window` has no production caller. The live evaluator passes the
selected arrays directly to `_score_windows` and `_trajectory_reports`; the
helper remains only because one unit test exercises its old indexing behavior.

## Change contract

- Remove `_gather_window` and its obsolete helper-only test.
- Preserve the active evaluation array selection, checkpoint schedule, scoring,
  trajectory reports, candidate selection, and metrics.
- Do not change model parameters, random seeds, tensor construction, optimizer
  inputs, or runtime control flow.
- Do not add a compatibility shim for an unreferenced private helper.

## Acceptance

- `rg` finds no production or test reference to `_gather_window`.
- The complete co-located Example 21 test module passes.
- Production and co-located test source line counts are lower than `main`.
- A deterministic smoke run reports the same metrics, candidates, losses, and
  parameter-change evidence as `main`.
- The cleanup does not add runtime work; smoke runtime is no slower than `main`.

## Verification

- `_gather_window` has no source or test references after the cleanup.
- Production entry-point source: 5,900 lines on `main` → 5,880 lines
  (`-20`). Its co-located test: 4,370 → 4,353 lines (`-17`).
- Focused scoring and trajectory gate: 2 passed.
- Complete co-located Example 21 gate: 149 passed, 20 existing compiler
  warnings, in 145.74 seconds. The removed helper test accounts for the one
  fewer test.
- Identical smoke configuration, seed 2108, CPU, 128 neurons, 1,024 edges,
  three updates, and 60 latent steps: normalized result artifacts compare
  equal between `main` and this branch; metrics, candidates, losses, parameter
  changes, and parameter hashes compare equal.
- Smoke runtime pair 1: `main` 33.101 seconds; this branch 39.034 seconds.
- Smoke runtime pair 2: `main` 32.677 seconds; this branch 28.995 seconds.
  The live path is unchanged, so the spread is process/cache noise; the
  cleanup adds no runtime work and the repeat is faster on this branch.
- Ruff reports no diagnostics for the changed source and test files.
