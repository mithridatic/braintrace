# Example 21 — restoring the demonstration-verified rule channel to the submission

## Problem

The submitted ARC score for Example 21 is produced by a single channel: two
candidates decoded from the spiking model's output heads
(`primary_candidate_mode = "model_only"`). At full scale on the strongest
retained base (4,096 neurons, 4,194,304 recurrent edges, `--copy-residual-gain
2.0 --row-head-carrier-scale 0.0`, seed 31337) that channel scores, at the
submission checkpoint (effort 60) over 400 evaluation tasks / 419 queries:

| metric | value |
| --- | ---: |
| query_pass_at_1 | 1 |
| query_pass_at_2 | 2 |
| strict_task_pass_at_1 | 0 |
| strict_task_pass_at_2 | 1 |

## Measurements that motivated the change

An eval-only replay (37 s, restored parameters, no training) dumped the raw
height / width / colour logits for all 419 queries. Replaying alternative decode
policies offline against that dump reproduces the harness metrics exactly and
gives the following ceilings at effort 60:

| decode policy | q@1 | q@2 | task@1 | task@2 | shape |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped (argmax heads, min-margin runner-up) | 1 | 2 | 0 | 1 | 0.5752 |
| demonstration-derived shape, model colours | 2 | 2 | 1 | 1 | 0.8783 |
| **oracle shape**, model colours | 2 | — | — | — | 1.0000 |

Shape is not the binding constraint on the exact score: with the *true* output
shape handed to the decoder the model still produces only two exact grids. The
colour head is the ceiling. Of the near misses under oracle shape, none is
recoverable inside a two-candidate budget — the true colour sits at rank 9, 2
and 4 respectively for `1a2e2828`, `642d658d` and `27a77e38`.

The repository already contains a demonstration-verified rule channel
(`latent_workspace_rules.py` and the `latent_workspace_rule_*` families). It
admits a rule only when that rule reproduces **every** demonstration pair
exactly, and it never inspects a query target. Scored standalone over the same
400 tasks it proposes on 27 of 419 queries and is correct on all 27:

| channel | q@1 | q@2 | task@1 | task@2 | admitted | wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| verified rules, standalone | 27 | 27 | 26 | 26 | 27 | 0 |

The channel was wired into the entry point in `dfe2412` and removed again in
`961df5e` ("enforce model-only LOO evaluation"). Nothing about the channel was
found to be wrong; the removal was a scoping decision.

## Change

Add a second primary candidate mode, `rule_then_model`, selected by
`--primary-candidate-mode`. `model_only` stays the default and keeps its exact
present behaviour, policy string, and acceptance checks.

Merge policy under `rule_then_model`, per query:

- candidate 1 — the cheapest admitted demonstration-verified rule when one
  exists, otherwise the model's first decoded candidate;
- candidate 2 — the model's first decoded candidate when a rule occupies
  slot 1, otherwise the model's second decoded candidate.

A rule that is admitted but wrong therefore costs the run at most the model's
runner-up slot, and never displaces the model's own best grid.

Each arm is fitted on the demonstrations *that arm actually has*: `no_context`
receives none and so admits no rule, and `shuffled_demonstrations` is fitted on
the deranged demonstrations. Without this the proposals would be arm-invariant
and every control would report the same solves as `intact`.

## Nothing here was fitted to the evaluation split

The merge policy is not new and was not chosen by looking at eval scores. Rule
in slot one, the model's own best grid in slot two, is `dfe2412`'s design,
written before any of these measurements. The rule engine and its families are
byte-identical to `4e9080b` — this change adds no rule, tunes no rule, and
removes no rule. The decode policies that *were* measured against the evaluation
split above (demonstration-derived shape, mean-normalised shape scoring,
alternative runner-ups) were all **rejected**; none of them ships.

## Reporting

The run must never describe a rule-assisted score as model-only.

- `submission_policy.name` becomes
  `latest_checkpoint_demonstration_verified_rule_then_model_v1` and
  `submission_policy.rule_channel_enabled` becomes `true`.
- Every candidate carries `provenance` of `"model"` or `"rule"`, and a rule
  candidate carries the `rule_name` that produced it.
- `channel_attribution` reports, per effort: queries that admitted a rule,
  admitted rules that were not exact, exact solves credited to the rule
  channel, and exact solves credited to the model's candidates.
- The model-only metrics are computed and retained in the same artifact, so the
  network's own score stays readable next to the merged score.

## Expected result

Offline replay of the merge against the retained logits dump, effort 60:

| channel | q@1 | q@2 | task@1 | task@2 |
| --- | ---: | ---: | ---: | ---: |
| model only | 1 | 2 | 0 | 1 |
| rule_then_model | 28 | 29 | 26 | 27 |

## Not in scope

The colour-head ceiling. The measured route to a better *model-only* score is a
better colour head, not a better decoder or a better shape prior; the oracle
shape probe above is the evidence. Feeding demonstration output shapes to the
shape head as a model input (keeping `model_only` true) is the natural next
experiment and is not attempted here.

## Measured result

Run `var/example21-rtm-final`, 2026-08-22 (reproduced at the branch tip; the
earlier `var/example21-rtm-s31337` returned the same four scores). Pre-protocol-v2 code
(`bde13ba` + this change), full scale (4,096 neurons / 4,194,304 recurrent
edges), Muon, cosine, weight decay 0.1, 260 updates, chunk 5, batch 32, seed
31337, `--copy-residual-gain 2.0 --row-head-carrier-scale 0.0`,
`--primary-candidate-mode rule_then_model`. Wall clock 523.3 s. 400 tasks,
419 queries. `full_structural_qualification` is `true`.

Effort 60, the submitted checkpoint:

| channel | q@1 | q@2 | task@1 | task@2 | shape | pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| submitted (`rule_then_model`) | 28 | 29 | 26 | 27 | 0.6134 | 0.5873 |
| model only (same run, not submitted) | 1 | 2 | 0 | 1 | 0.5752 | 0.5482 |

27 of 419 queries admitted a rule and all 27 were exact; no admitted rule was
wrong, so the channel cost the run nothing. The model's own candidates
contributed the remaining two exact queries (`e872b94a` q0 at slot 2, and
`bbb1b8b6` q1). Seventeen distinct completions did the solving, the most
frequent being `id|tile2x2` and `id|per0` at four each.

The model-only channel reproduces the retained baseline
(`example21-oldfull-cr2cs0-s31337`) to within GPU non-determinism: identical
shape (0.5752) and identical solved tasks, pixel 0.54820 against 0.54839.

`associative_capability_gates_complete` remains `false` because the run does not
enable `--evaluation-controls`; that is unchanged from the baseline and is not a
consequence of this change.

## On main

The change is on `main`. `model_only` remains the default everywhere, so doing
nothing keeps the previous behaviour, policy string, and completion gate exactly
as they were.

Protocol v2 runs the control arms by default, and `_control_summary` recomputes
the intact metrics and cross-checks them against the primary. Carrying the rule
channel on the primary alone made that recomputation disagree and the run failed
closed. Each arm is therefore fitted on its own demonstrations, which is
`dfe2412`'s discipline: `no_context` admits nothing, `shuffled_demonstrations`
is fitted on the deranged pairs, and the arms that ablate neurons rather than
demonstrations keep the intact proposals.

Measured on `main` with the same recipe and seed, at effort 60:

| run | q@1 | q@2 | task@1 | task@2 | model-only pixel | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `var/example21-rtm-main`, `--no-evaluation-controls` | 27 | 27 | 26 | 26 | 0.0160 | 525 s |
| `var/example21-rtm-main-full`, controls on | 27 | 27 | 26 | 26 | 0.0160 | 720 s |

The rule channel behaves identically to the pre-v2 tree — 27 admitted, 27 exact,
none wrong. The one-task difference against the pre-v2 result is entirely the
model channel, which the protocol-v2 regression takes from pixel 0.548 to 0.016
and from two exact queries to zero. Reproducing 28/29/26/27 therefore still
needs `feat/ex21-shape-decode`; that branch is retained.

Neither `main` run reaches `full_structural_qualification`. With controls off
the only failure is `required_controls_executed`, by construction. With controls
on the failures are `associative_diagnostics_complete`,
`repeat_intact_deterministic` and `slot_ablation_pre_intervention_matched` —
all three predate this change and reproduce on `example21-full-muon-cr2g` and
`example21-full-default-u390`, which additionally fail two checks this run
passes.
