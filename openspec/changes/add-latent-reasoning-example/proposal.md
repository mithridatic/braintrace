## Why

Every pp_prop example in `examples/pp_prop/` supervises a network whose task rule
is fixed by training. None of them ask the network to acquire a *fresh* rule at
inference time from demonstrations, hold it in state, and then compute with it.
BDH-CQ (arXiv:2608.09888) defines a system-level interface for exactly that —
a recurrent contextual memory written by demonstrations with parameters frozen,
plus a latent workspace iterated at query time. Example 21 places the eligible
recurrent and readout parameters under terminal-only pp_prop supervision across
that full rollout. Its release memory-write projections are fixed random because
the feasibility spike did not establish a learned write path; the example tests
task behavior and geometry without claiming that pp_prop learned the write.

The change adds Example 21, which instantiates the published interface at
example scale and reports what the latent workspace actually looks like and
whether iterating it changes results.

## What Changes

- New example `examples/pp_prop/21-latent-reasoning-in-context.py` with three
  supporting modules and co-located tests.
- New in-context task generator: fresh symbol permutation per episode, drawn
  demonstrations, held-out query, deterministic oracle, and a supported-vs-short
  context split over byte-identical queries.
- New two-state model: a factored Hebbian contextual memory `(A, B)` written
  through fixed-random projections during demonstration ingestion as hidden
  state, read by a hidden-state contraction the compiler absorbs into the
  hidden-to-hidden transition, and a latent workspace iterated for `R` silent
  ticks before readout.
- New latent-geometry analysis: participation ratio, iteration-to-iteration
  trajectory norm, and linear probes for answer and rule decodability from the
  workspace and the query-conditioned memory read, with raw factors reported as
  a secondary diagnostic.
- `examples/pp_prop/README.md` gains a catalog row and two axis-map rows
  (in-context rule binding; latent iteration depth).
- Repository spec `docs/specs/2026-08-16-pp-prop-latent-reasoning.md`.
- No change to `braintrace/` library code. No breaking changes.

## Capabilities

### New Capabilities

- `pp-prop-latent-reasoning`: an example testing in-context rule acquisition and
  iterated latent computation under pp_prop, covering the task generator, the
  two-state model, the sweep arms and controls, the latent geometry report, and
  the claims the example is forbidden from making.

### Modified Capabilities

<!-- None. No existing braintrace requirement changes; this adds an example. -->

## Impact

- Affected code: `examples/pp_prop/` only. `braintrace/` is untouched — the
  example composes the existing public `matmul` ETP operator for
  parameter-times-input operations and the existing pp_prop configuration. The
  release optimizer excludes the fixed-random memory-write projections.
- Affected docs: `examples/pp_prop/README.md`, new `docs/specs/` entry.
- Dependencies: none added. Plotting uses the matplotlib Agg path already used
  by Examples 18 and 19.
- Runtime: the example must have a `--smoke` path that completes in the same
  order of time as the other examples' smoke paths, so the repository example
  gate stays usable.
- Risk carried into design: the task must degrade somewhere across the
  2→8 simultaneous-binding range, and the silent latent segment must not simply
  decay to zero. A throwaway spike resolves the initial sizing questions before
  the model is built; the corrected production LIF is then requalified
  separately, and a failed production gate blocks release rather than inheriting
  the spike's score.
