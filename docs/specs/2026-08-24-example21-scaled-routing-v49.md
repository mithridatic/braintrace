# Example 21 scaled query-routing arm V49

Status: approved for implementation

Date: 2026-08-24

Branch: `feat/example21-direct-generation`

## Objective

Scale the single mechanism that V48 proved. V48 (artifact
`var/ex21-online-v48-query-routing-v1`) produced the first strict
routing-family task of the direct line (`synthetic-v4:upscale:000088`) and
12/120 strict on the fresh v4 holdout, with unconverged losses at 800
updates (about 4.5 episodes per training task). V49 changes exactly one
variable: the update count, from 800 to 2400. Architecture, model seed,
curricula, loss, trainer, batch size, learning rate, trace decay, scopes,
and evaluation are byte-identical to V48. No other change is permitted in
this arm.

## Experiment contract

One score-ineligible run, artifact `var/ex21-online-v49-scaled-routing-v1`:
`latent_workspace_query_routing_experiment` with
`--training-updates 2400 --training-chunk-size 20` and every other flag at
its V48 default. The same fresh 120-task v4 holdout (seed 44108) is the
decision scope; the real in-library and fold-zero scopes remain diagnostics
only, and the evaluation role is never opened.

## Gates

1. Mechanism gate: unchanged V48 definition.
2. Scaling gate: holdout strict strictly above V48's 12/120 with at least
   one routing-family (crop, dihedral, mirror_concat, project_marker,
   upscale) strict task retained. A pass justifies the next separately
   specified mechanism arm and a matched longer training budget for any
   future nomination run. A fail — strict count at or below 12, or the
   routing family lost — closes the more-updates direction: duration is
   then ruled out as the binding constraint, and the next arm must change a
   mechanism, not a schedule.
3. Anti-collapse gate: unchanged V47/V48 thresholds.

## Non-goals

No architecture, program-count, shift-range, curriculum, loss, seed, or
real-task fitting change. No complete-manifest evaluation regardless of
outcome.
