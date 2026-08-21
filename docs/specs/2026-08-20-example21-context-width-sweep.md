# Example 21 — context_memory_width sweep with evaluation controls

Status: measured; width-capacity hypothesis refuted at tested loads
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

## Results (2026-08-20 evening; all runs measured, none over 6 minutes)

The original three-run full-split design was replaced after a runtime
objection: evaluation capped at 100 tasks for the ARC arms, and the
quantitative pairing measurement moved to its native instrument, the
`latent_workspace_binding_gate` runner, which takes `--context-memory-width`
directly and needs no ARC evaluation. Gate runs used
`--validation-episodes 1024` (off-preregistration on purpose, identical
across widths), so both artifacts are explicitly
`nonqualifying_abbreviated_no_capability_conclusion` — internally valid as a
width comparison, not as Gate A evidence.

### Binding gate (synthetic K=4 curriculum, 1024 held-out episodes)

| width | intact accuracy [Wilson 95%] | shuffled | no-context | wall |
|---:|---|---|---|---:|
| 32 | 1.000 [0.9963, 1.0] | 0.000 [0, 0.0037] | 0.1025 | 133 s |
| 512 | 1.000 [0.9963, 1.0] | 0.000 [0, 0.0037] | 0.0889 | 296 s |

Chance is 0.25. Binding is *perfect and width-invariant* at the K=4 load:
intact saturates, the derangement collapses to zero, and no-context sits
below chance. Width buys nothing because nothing is missing at this load.

### ARC pilot runs (100 evaluation tasks, evaluation controls on, seed 2108)

| width | shape | pixel | pairing-sensitive | no-context zero | wall |
|---:|---:|---:|---|---|---:|
| 32 | 0.5096 | 0.3620 | true | true | 233 s |
| 512 | 0.5385 | 0.3668 | true | true | 359 s |

A 16× width increase moves shape by +0.03 and pixel by +0.005 under the real
ARC demonstration load (~hundreds of outer-product writes per task).

### Verdict on §4.3's prediction

The capacity prediction is **refuted as stated**: pairing at width 32 is not
near chance for capacity reasons — the mechanism binds perfectly at low load,
and widening the memory 16× under ARC load produces no material gain. The
constraint is load-dependent: either interference at hundreds of writes that
width does not relieve (because the fixed random Fourier keys, not the store,
set the effective addressing resolution) or the write/read coding itself.
This moves recommendation #7 (learn `U_θ`) ahead of any width/decay
engineering.

### Decisive next measurement (not run; requires curriculum change)

The discriminating axis is **load**, not width: sweep the gate curriculum's
pair count (`SYMBOL_COUNT`, currently a preregistered constant of 4) at fixed
width and locate where intact accuracy departs from 1.0, then test whether
width shifts that knee. If the knee does not move with width, coding is
conclusively the binder. This modifies the preregistered gate and should be
proposed, not slipped into a sweep.

### Artifacts

- `var/binding-gate-width-sweep/gate-w{32,512}.json` (this worktree)
- `var/example21-wspilot-w{32,512}-t100/` (this worktree)
- ARC arms marked non-scientific by construction (task cap); gate arms marked
  nonqualifying by construction (off-preregistration validation count).
