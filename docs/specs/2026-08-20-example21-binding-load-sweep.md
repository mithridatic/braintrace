# Example 21 — binding-gate load sweep (SYMBOL_COUNT at fixed width)

Status: measured; no knee found — binding is load-invariant through K=8
Date: 2026-08-20
Branch: `investigate/ex21-context-width-sweep`

## Question

Follow-up to `2026-08-20-example21-context-width-sweep.md`, which refuted the
width-capacity hypothesis: binding is perfect (intact 1.000, shuffled 0.000)
at the K=4 curriculum load at both width 32 and width 512, while ARC-load
performance is width-flat. The discriminating axis is therefore **load**: how
many demonstration pairs can the associative memory bind before intact
accuracy departs from 1.0, and does width move that knee?

## Design

Sweep the binding-gate curriculum's pair count K (`SYMBOL_COUNT`) at fixed
`context_memory_width = 32`:

| K | width | source |
|---:|---:|---|
| 4 | 32 | existing artifact `var/binding-gate-width-sweep/gate-w32.json` |
| 6 | 32 | new run |
| 8 | 32 | new run |

All other settings identical to the existing K=4 run: gate defaults plus the
shared off-default `--validation-episodes 1024` (keeps every artifact in the
`nonqualifying_abbreviated` regime and Wilson intervals tight). Pairing chance
scales as 1/K automatically (`pairing_chance = 1.0 / SYMBOL_COUNT` throughout
control and gate), so accuracies are compared against 0.25 / 0.1667 / 0.125
respectively.

Contingency (within approved scope, only if triggered): if intact accuracy
departs from 1.0 at K=6 or K=8, rerun that K at width 512 to test whether
width moves the knee. Width-invariant knee → the fixed random-feature coding
is conclusively the binder; knee relieved by width → capacity re-enters at
higher load.

## Code change

`SYMBOL_COUNT` in `latent_workspace_binding_control.py` is a module constant
that seeds every derived quantity (mapping catalog `_INPUT_SETS` /
`_OUTPUT_ASSIGNMENTS`, sequence length K+1+gap, row-event layout, derangement
rotation, chance levels) in both the control and the gate (which imports it).
Threading a config field through the preregistered gate would be a large,
risky diff; instead the constant becomes environment-parameterized at import:

- `SYMBOL_COUNT = _symbol_count_from_environment(os.environ)` with default 4,
  accepting integers in [2, COLOR_COUNT]; anything else raises at import.
- `qualification_regime` (control) additionally requires `symbol_count == 4`,
  so a non-default K can never masquerade as `preregistered_full` Gate A
  evidence — every K≠4 artifact is `nonqualifying_abbreviated` by
  construction.
- The K=4 mapping-catalog error message becomes K-generic.
- `_data_report` records `symbol_count` explicitly.

Unset env → byte-identical behavior to today (K=4, preregistration intact).

Feasibility at K=8: `_OUTPUT_ASSIGNMENTS` = P(10,8) = 1,814,400 tuples built
at import (~hundreds of MB transiently, seconds of CPU); catalog size
45 × 1,814,400 = 81.6M mappings comfortably exceeds the 640k + 1024 episode
demand. Sequence length grows 6 → 10.

## Decision rule

- Intact stays ≈1.0 through K=8 at width 32 → no knee within the reachable
  curriculum (K ≤ 10 colors); load alone does not break binding at these
  scales, pointing the interference explanation at the ARC regime's
  hundreds-of-writes load rather than pair count per se.
- Intact departs from 1.0 at some K → knee located; run the width-512
  contingency to attribute it to coding (knee immobile) vs capacity (knee
  moves).

## Results (2026-08-20 evening; both new runs measured)

All at width 32, 1024 held-out episodes, gate defaults +
`--validation-episodes 1024`, commit `e565872`.

| K | chance | intact accuracy [Wilson 95%] | shuffled | no-context | wall |
|---:|---:|---|---|---|---:|
| 4 | 0.250 | 1.0000 [0.9963, 1.0] | 0.0000 [0, 0.0037] | 0.1025 | 133 s |
| 6 | 0.167 | 0.9912 [0.9834, 0.9954] | 0.0029 [0.0010, 0.0086] | 0.1055 | 156 s |
| 8 | 0.125 | 1.0000 [0.9963, 1.0] | 0.0000 [0, 0.0037] | 0.1084 | 202 s |

### Verdict

**No knee within the reachable curriculum.** Intact accuracy is perfect at
K=4 and K=8 and 0.9912 at K=6 (9/1024 errors); the K=6 dip is
non-monotonic — heavier load at K=8 returns to exactly 1.000 — so it reads
as per-run training variation, not a capacity onset. The derangement control
collapses to ≈0 and no-context sits below chance at every K. The width-512
contingency was not triggered: with no load-dependent trend there is no knee
whose mobility width could attribute.

Combined with the width sweep (`2026-08-20-example21-context-width-sweep.md`):
the associative memory binds perfectly at every pair count the 10-color
curriculum can express, at its narrowest preregistered width. Neither width
nor pair-count load explains the ARC-regime binding deficit. The remaining
suspects are the ARC encoding regime itself — hundreds of heterogeneous
outer-product writes through *fixed random Fourier keys*, versus ≤8 clean
one-cell writes here — which keeps recommendation #7 (learn `U_θ`) as the
lead experiment.

Diagnostics note: `evaluation_complete_and_finite` is false at K=4 and K=8
but true at K=6 — the known GPU repeat-intact nondeterminism (±0.002) seen
throughout the width sweep, orthogonal to the accuracy readouts above.

### Artifacts

- `var/binding-gate-width-sweep/gate-k{6,8}-w32.json` (this worktree);
  K=4 row is the width-sweep artifact `gate-w32.json`.
- All three artifacts are `nonqualifying_abbreviated` by construction
  (shared off-preregistration validation count; K≠4 additionally
  disqualified by the new symbol-count regime guard).
