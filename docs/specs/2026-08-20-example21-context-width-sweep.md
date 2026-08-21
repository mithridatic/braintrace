# Example 21 — context_memory_width sweep with evaluation controls

Status: in progress
Date: 2026-08-20
Branch: `investigate/ex21-context-width-sweep`

## Question

Recommendation #2 of `2026-08-20-example21-bdh-cq-evaluation.md`: is the
binding-at-chance result (presence 0.99 / pairing 0.51) a **capacity** limit of
the 32-wide associative memory `S_K`, or a **coding** limit of the fixed random
Fourier keys / random value bases with only a 32×32 learned gain?

§4.3 of that evaluation makes the falsifiable prediction: a task writes up to
~600 outer products into ~`width` usable directions with `memory_decay = 1.0`,
so pairing should stay near chance at width 32 *for capacity reasons* even
though the `λI` architecture is correct.

## Decision rule

Run the already-written associative-capability diagnostics
(`--evaluation-controls`, including the derangement control `_derange_task`)
at `context_memory_width` ∈ {32, 128, 512}, everything else held at the
current best-known configuration.

- Pairing rises with width → memory is capacity-limited; architecture sound;
  next step is width/decay engineering.
- Pairing stays ≈ chance at width 512 → the write/read coding (fixed random
  features + 32×32 gain) is the constraint; next step is learning `U_θ`
  (recommendation #7).

## Code change required

`ExperimentConfig.__post_init__` rejects `context_memory_width > 128`. The cap
is a guardrail introduced with the original workspace integration
(`65fe456`), not structural: every width-dependent array
(`(width, width)` memory, `(neurons, width)` projections) scales generically,
and peak extra memory at width 512 is ~1 MB per example. The cap is raised to
512 (kept as a guardrail rather than removed); the boundary test moves from
129/"at most 128" to 513/"at most 512".

## Run matrix

Base configuration = the hyperparameter-sweep baseline
(`example21-mv-4096n-4096e-b32-u260-l60-lr1e-3-s2108`): 4096 neurons,
4096 recurrent edges, batch 32, 260 updates, chunk 5, latent steps 60,
lr 1e-3, seed 2108, decoder `row_refinement`, `memory_decay 1.0`.

| arm | context_memory_width | flags |
|---|---:|---|
| w32 | 32 | `--evaluation-controls` |
| w128 | 128 | `--evaluation-controls` |
| w512 | 512 | `--evaluation-controls` |

Single seed (2108) first pass; widths differ only in the one config value.
`--evaluation-controls` switches evaluation to the dense trajectory
(`latent_steps + 1` offsets) and runs five arms (intact, repeat, no-context,
shuffled-demonstrations, slot-ablation), so expect roughly 5× evaluation cost
over a plain run.

Image: rebuilt from this worktree as
`braintrace-gpu:0.11.0-py314-msgspec-arc-widthsweep` (own tag; the shared
`-arc` tag used by concurrent runs is not overwritten). ARC checkout:
`C:\tmp\braintrace-example21-data\arc-agi-1` at `aa922be`.

Outputs: `var/example21-ws-4096n-4096e-b32-u260-l60-lr1e-3-s2108-w{32,128,512}`
in this worktree.

## Report

Per width: pairing accuracy with Wilson 95% interval against
`pairing_chance`, presence, and the shuffled/no-context/ablation control
verdicts; pairing broken down against demonstration-row count where the
diagnostics expose it. Shape/pixel metrics recorded as secondary (widths may
shift the frozen-arm score, which is itself informative).
