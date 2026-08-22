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
