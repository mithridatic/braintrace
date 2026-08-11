# Example 18 · Multi-task structural evolution under pp-prop

Date: 2026-08-10
Status: approved design, pre-implementation

## Question

If one spiking network learns two different temporal "tricks" continually
(pp-prop, no resets), and its recurrent synapses are pruned/regrown on a
fixed budget, does the wiring self-organize into task-specialized sections,
does it share parts, and can it keep doing both tricks at the same quality?

## Setup

- One LIF recurrent network, native CSR recurrence (example 09 wiring).
  Defaults: 1024 neurons, 1024 recurrent edges (irregular degree — no
  fixed-degree constraint), dense input and readout projections.
- Two delayed-cue temporal tasks, trials interleaved 1:1 in one stream:
  - "fetch": cue on input units 0–31 (ticks 0–4), respond ticks 10–14
  - "roll over": cue on input units 32–63 (ticks 0–4), respond ticks 30–34
  - 2-unit softmax readout, label = task identity, supervision only inside
    the active task's response window (same masking style as example 15).
- Continual learning: one pp-prop learner per round; per-trial
  `etrace_grad` + optimizer update; weights persist across everything.
- Structural evolution, once per round: prune the 5% of recurrent edges
  with smallest trained |w|; respawn the same count with endpoints sampled
  proportional to per-neuron mean spike rate (measured on probe trials)
  plus a small floor; no self-loops, no duplicate edges within a row.
  Edge budget stays exactly 1024. Surviving edge values and the readout
  weights carry over across the rebuild.
- Control arm: identical run with evolution off (topology frozen at the
  initial random graph, same budget, same trial sequence seeds).

## Reporting contract (user-facing)

No jargon metrics. The example ends with a plain-English section, e.g.:

```
=== What happened, in plain English ===
- Fetch works: NN% correct (chance 50%). Roll over works: NN%.
- Learning one trick did/did not wreck the other: ...
- The brain grew sections: X% of synapses care only about fetch,
  Y% only about roll over, Z% are shared.
- With evolution switched off, the split was flatter: ...
- Picture saved to <path>: red = fetch synapses, blue = roll over,
  grey = shared; right panel = accuracy per trick over rounds.
```

And one PNG (matplotlib Agg, `--plot-output`, default
`structural_evolution.png`): left = evolved adjacency scatter colored by
task attribution; right = per-task accuracy over rounds, solid = evolving,
dashed = control.

Task attribution per edge: accumulated per-task |gradient| mass on the CSR
values during training; edge is task-leaning when one task holds >60% of
its total mass, else shared.

## Scope limits

- Example + co-located test + README rows only. Example 17 and its
  manifest are untouched. No git commits.
- Lesion/ablation analysis is future work, not v1.
- Claims are illustrative example output, not benchmark evidence; the
  example carries no sealed-test or accuracy gates beyond finite losses
  and invariants.

## Adaptive budget (v2, 2026-08-10)

The evolving arm now runs an adaptive synapse-budget controller by default.
The v1 behavior (prune the weakest 5% by |w| and respawn the same count on
a constant 1024-edge budget) remains available via `--fixed-budget`. The
control arm is unchanged: frozen initial topology, 1024 edges.

Controller, applied after each round's evaluation:

- `min(fetch_acc, roll_over_acc) < --target-accuracy` (default 0.95) →
  **grow**: add activity-biased edges (same endpoint rule as v1 respawn —
  endpoints ∝ per-neuron mean spike rate on probe trials plus a floor, no
  self-loops, no within-row duplicates; values drawn from the initial value
  distribution; zero initial attribution mass). New budget =
  `min(round(current * 1.5), --max-edges)`, at least one edge; default cap
  1,048,576.
- Otherwise → **shrink**: prune the weakest 10% by |w| without replacement,
  never below `--min-edges` (default 64), and never growing through the
  floor when the current budget already sits under it.

The expected trace is a sawtooth — shrink until the weaker trick dips below
target, grow back — settling near the minimal sufficient budget. Edge count
per round becomes a first-class output: a grey twin axis on the right PNG
panel behind the accuracy curves, plus budget-journey lines in the
plain-English report (growth events naming the bottleneck trick and its
accuracy, the settling budget vs the frozen control, total seconds, and
per-round wall time at the end vs the start). Timing is reported honestly:
at this scale fixed per-step costs and the dense 64×1024 input projection
dominate the sparse recurrent edges, so edge count is not expected to move
round time much, and the report says so when it does not.

New tests: controller grow/shrink bound unit test (cap and floor
respected), and a 3-round smoke with forced growth (`target_accuracy=1.01`)
and forced shrink (`target_accuracy=0.0`) asserting exact budget sequences,
budget invariants, and CSR validity. Values outside [0, 1] are deliberately
legal for `target_accuracy` to enable those forced paths.

## Task style (v3, 2026-08-10)

`--task-style {simple,temporal-credit}` (default `simple`, unchanged from
v1/v2). The `temporal-credit` style replaces the easy detect-then-respond
trials with example 17's delayed-cue recall task, because the simple tasks
turned out to be solvable from passive damped membrane traces — accuracy
stayed at 100% down to 16 recurrent edges, so the adaptive controller's
GROW path could never fire organically.

The port reuses example 17's encoding module (`temporal_benchmark_data`:
channel layout, physical-rate spike probabilities) and its fixed time
constants (membrane 0.5 ms, feed-forward synapse 0.5 ms, recurrent synapse
3 ms, readout 0.5 ms), which make passive traces decay far below threshold
across the delay, so the task is solvable only by regenerating activity
through trained recurrent edges. No manifest, sealing, or runner machinery
is touched. The two interleaved tasks share one trial geometry: cue
ensemble A (units 0-7) or B (units 8-15) on ticks 0-3 at 200 Hz; the common
go channel (unit 16) fires identically in both response windows — fetch
answers at ticks 6-9 (17's short horizon), roll over at ticks 26-29 (17's
medium horizon, the longest delay pp-prop learns directly, with its
selected half-life-20 trace decay) — so response inputs are analytically
label-independent and only the active task's window is supervised. The long
horizon (silence 92) is deliberately not used: it sits beyond pp-prop's
eligibility horizon without a curriculum and would fail at every budget,
which is a degenerate result rather than a growth signal. Recurrent init
and respawn values use example 17's `gain / sqrt(degree)` scale (gain 0.8,
degree = initial average degree). Feed-forward/readout initializations and
the no-bias policy also follow example 17; the optimizer adopts example
17's per-group learning rates as the default (see v5).

Tests: a temporal-credit smoke test (1 round, 32 neurons/32 edges) checks
finite losses, accuracies in [0, 1], budget invariants, CSR validity, and
the ported trial geometry.

## More tricks (v4, 2026-08-10)

`--num-tricks` (default 2, back-compatible) scales the temporal-credit
style to N tricks competing for the same neurons, to find the true budget
floor, force repeated organic growth, and see whether synapse sharing
emerges under capacity pressure. Trick `k` gets cue channels `8k..8k+7`
(same 200 Hz ticks 0-3 encoding), the go channel sits at index `8N`, and an
N-unit softmax readout (label = trick id, chance = 1/N) is supervised only
in the active trick's window. Two tricks keep the v3 geometry (respond at
6-9 and 26-29 — example 17's short and medium horizons); three or more
space response windows every 7 ticks (6-9, 13-16, 20-23, 27-30 for four),
so the latest onset stays within one tick of the validated medium horizon
(config validation rejects anything later). More than two tricks requires
the temporal-credit style; the simple style stays two-trick. Trick display
names are fetch / roll over / sit / stay. Trials interleave uniformly.

Per-edge attribution generalizes to N tasks: an edge is trick-leaning when
its top task holds more than 60% of its total accumulated |gradient| mass,
else shared (the shared label equals N; for two tasks this is exactly the
v1-v3 rule). The report lists every trick's percentage plus shared, and the
wiring panel gives each trick its own color with grey for shared.

Tests: a four-trick temporal-credit smoke (finite losses, per-trick
accuracies in [0, 1], budget invariants, non-overlapping windows inside the
medium horizon) and an N-task attribution classification test on synthetic
masses.

## Per-group learning rates (v5, 2026-08-10)

The optimizer now mirrors example 17's per-parameter-group policy directly,
replacing the v1-v4 single Adam group (lr 3e-3, global clip 1.0): one Adam
instance per group — readout 3e-3, feed-forward 1e-3, recurrent 3e-4 —
with per-group clipping at 1.0, stepped in readout / feedforward /
recurrent order. Adopted as the default (no mode, no flag) per user
directive for direct alignment. Example 17's learning-rate schedule,
weight-decay search, and optimizer telemetry remain benchmark governance
and are not ported.

## Paced growth and consolidation (v6, 2026-08-10)

Two defaults changed after the v5 reruns: the growth factor is now 1.1
(was 1.5, also exposed as `--growth-factor`) and a round trains 800 trials
(was 400). Rationale: with example 17's per-group rates the recurrent group
learns at 3e-4, and the v5 4-trick runs showed that 1.5x growth spurts
inject fresh random edges faster than 400 trials per round can consolidate
them (each rebuild also resets the Adam moments) — per-trick accuracies
churned and the growing arm fell behind its own frozen control. Growth is
now paced against consolidation capacity, which also makes grow and shrink
steps symmetric (10% each way), so a shrink→dip→regrow sawtooth around the
sufficient budget becomes reachable. Tests that encode budget sequences
were updated to the 1.1x arithmetic.

## Context-dependent tricks (v7, 2026-08-10)

`--task-style context` (third style; `simple` and `temporal-credit`
unchanged) makes the tricks context-dependent so passive detection cannot
work and input channels cannot be privately partitioned per trick. Two cue
ensembles (A = units 0-7, B = units 8-15) and one context ensemble (X =
units 16-23) share the channel budget; the go channel is unit 24
(n_in = 25). Context X fires at ticks 0-3 on half the trials, the cue at
ticks 5-8, and the go fires in all four response windows
(label-independent response inputs). The four conditions — A alone
(respond 12-15), A with context (16-19), B alone (20-23), B with context
(24-27) — each supervise only their own window with label = condition
(chance 25%, uniform interleaving). Every response onset is at or before
tick 27, inside the validated medium-horizon bound.

Why this is harder than private-channel tricks: cue-detection wiring is
forced to be shared (both conditions of a cue read the same channels), so
only context-keeping circuitry can specialize; and X must be remembered
~20 ticks past its extinction — at the edge of pp-prop's medium horizon.
Encoding, time constants, trace decay (half-life 20), and gain scaling
follow the temporal-credit style. `--task-style context` implies four
conditions (`--num-tricks 4`); constructing the config directly requires
`num_tricks=4`.

Tests: layout validity (windows non-overlapping, onsets ≤ 27, n_in = 25),
a rate-template test proving context conditions differ from their
context-free counterparts only in the X channels (and that go inputs are
identical across conditions), and a tiny smoke run (finite loss,
accuracies in [0, 1], budget invariants).

## Tests (co-located `18-structural-evolution_test.py`, tiny fast config)

- Prune removes exactly the lowest-|w| edges; budget invariant holds.
- Respawn yields a valid CSR (monotone indptr, in-range indices, no
  self-loops, no within-row duplicates), endpoints respect the activity
  bias floor.
- Attribution classification (60% rule) correct on synthetic values.
- Smoke: 1 round, 32 neurons/32 edges, 8 trials completes with finite
  loss and accuracies in [0, 1].


## v8 (2026-08-10): gradient-guided growth

`--grow-rule gradient` replaces activity-biased sprouting in the adaptive
growth path. pp-prop only differentiates edges that exist, so the dense
candidate gradient is unavailable; instead each neuron's gradient
marginals come from the accumulated per-edge mass: post-synaptic *demand*
(mass summed over incoming edges) times pre-synaptic *supply* (mass summed
over outgoing edges) scores each free candidate — a rank-one estimate of
where a dense gradient would concentrate. The activity floor keeps every
free off-diagonal position reachable; zero mass everywhere falls back to
uniform. Fixed-budget mode keeps the activity rule.

Motivation: on the context task (v7), activity-biased growth fired every
round yet never fixed the weakest condition — the bottleneck was credit
assignment, not capacity, and undirected growth only paid the fresh-edge +
Adam-reset tax. Aiming growth at the error signal is the cheapest
structural rule that can in principle point at a credit bottleneck.

Tests: `_gradient_endpoints` falls back to a valid uniform draw under zero
mass, concentrates on the free neighbors of the hot row/column under
concentrated mass, and never emits self-loops or existing edges.


## v9 (2026-08-10): carry Adam moments across rebuilds

Every topology rebuild used to construct fresh optimizers, discarding all
Adam `mu`/`nu` and the bias-correction step counts. Both arms paid the
restart tax symmetrically, but the growing arm — rebuilding into a new
shape every round — paid it every round, which the v5/v6 churn analysis
identified as a growth penalty. Rebuilds now carry optimizer state by
default (`carry_optimizer_state=True`): readout/feed-forward moments copy
unchanged (shapes never change), and recurrent per-edge moments are
remapped by `(row, col)` edge identity — surviving edges keep their
moments, newborn edges cold-start at zero. Step counts carry, so bias
correction stays calibrated.

Tests: `_remap_edge_array` re-expresses per-edge arrays on a shrunk/grown
edge list by pair identity; an integration test proves recurrent `mu`
values land on their original edges after a shrink rebuild, dense-group
moments copy elementwise, and Adam counts carry.
