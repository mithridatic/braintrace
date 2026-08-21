# Example 21: cosine learning-rate schedule for shared training

Date: 2026-08-21
Status: implemented

## Problem

Example 21 shared training ran every optimizer update at a flat
`learning_rate` (default `1e-4`). The optimizer benchmark work moved the
default optimizer to Muon; a decaying rate with a higher base is the standard
companion regime, and the flat `1e-4` base was previously implicated in the
answer-head displacement starvation recorded in
`adam_parameter_travel_budget`.

## Decision

- New config knob `lr_schedule: {"constant", "cosine"}`, default `"cosine"`,
  with validation, CLI flag (`--lr-schedule`), smoke-config passthrough, and
  inclusion in `to_dict` (automatic via `asdict`) and `_optimizer_policy`.
- `learning_rate` default raised `1e-4` -> `1e-3` in both the config field and
  the CLI default (kept in sync). `smoke_config` keeps its own explicit
  `5e-4` smoke value.
- The schedule is `optax.cosine_decay_schedule(init_value=learning_rate,
  decay_steps=max(training_updates, 1))` (default `alpha=0`): base rate at
  update 0, ~0 at the final update. No warmup.
- All three optimizer paths in `_make_training_optimizer` accept the schedule.
  braintools `Adam`/`AdamW` take only scalar-or-braintools-scheduler rates, so
  the adam and adamw branches now build the equivalent optax transformations
  (`optax.adam`, `optax.adamw` — same b1/b2/eps defaults, same decoupled decay
  placement for adamw) wrapped in `braintools.optim.OptaxOptimizer`, exactly
  like the muon branch. Muon receives the schedule as both `learning_rate`
  and `adam_learning_rate`.
- The schedule applies only to shared training. The per-tick adaptation path
  (`adaptation_learning_rate`, its own `braintools.optim.Adam`) is untouched.
- `_parameter_travel_budget` multiplies the rate by the schedule integral
  factor (1.0 constant, 0.5 cosine-to-zero) before applying the
  `rate * updates` Adam displacement bound, and reports the schedule and
  factor, so the operating point no longer overstates travel under cosine.

## Non-goals

- No warmup phase.
- No schedule for adaptation.
- No change to the resolved weight-decay policy (previous spec).

## Tests

Co-located in `examples/pp_prop/21-latent-reasoning-in-context_test.py`:

- defaults pinned across constructor, `smoke_config`, and CLI
  (cosine + `1e-3`); `"constant"` still selectable; invalid value rejected.
- schedule arithmetic pinned at update 0 (base rate), midpoint (half rate),
  and horizon (~0); constant path returns the flat base rate.
- `_optimizer_policy` and `to_dict` carry `lr_schedule`; CLI round-trips it.
- travel budget halves under cosine relative to constant.
