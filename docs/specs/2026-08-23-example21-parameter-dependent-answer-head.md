# Example 21 parameter-dependent answer head

Status: approved for implementation

Date: 2026-08-23

Branch: `feat/example21-parameter-dependence`

## Objective

Raise the complete ARC-AGI-1 evaluation result to a cumulative exact score of
at least 16 while proving that every counted answer flows through the trained
Example 21 network. A useful answer head is not merely called a model head: its
candidate grids and exact score must change when its checkpoint parameters are
changed under matched evaluation.

The local cumulative score is

```text
cumulative = query_pass_at_1_count
           + query_pass_at_2_count
           + strict_task_pass_at_1_count
           + strict_task_pass_at_2_count
```

The four terms are integer membership counts on one fixed, complete evaluation
manifest. Adding query and strict-task counts is an explicit engineering target,
not an official ARC metric. The individual counts remain the authoritative ARC
results and must always be reported beside the cumulative value.

## Current evidence and correction

The retained parameter-consuming `carrier_row` answer head scores
`2 + 2 + 1 + 1 = 6`. The newer demonstration-fitted forest scores
`9 + 9 + 7 + 7 = 32`, but it reads raw demonstration grids and never consumes
the trained checkpoint, recurrent trajectory, carrier, or memory. Its flat
answer under parameter perturbation disqualifies it from the model score.

The forest remains valuable evidence about the inductive bias needed for ARC.
It is henceforth a `demonstration_only_diagnostic`: it may be evaluated and
reported under a diagnostic namespace, but its raw ordering may not determine a
primary candidate slot or contribute to the cumulative qualification score.
Its target-free grids become eligible proposals only after every submitted rank
is assigned by the checkpoint-likelihood path below. Changing a provenance
label is not evidence of model ownership.

A read-only full-split replay then tested the narrowest model-owned composition:
the forest generates target-free proposals and trained-network likelihood
orders them using

```text
combined(candidate) = forest_log_probability(candidate)
                    + 1.0 * trained_network_candidate_log_probability(candidate)
```

The factorized network likelihood uses the recurrent model's predicted height,
width, and cell-color probabilities for the complete candidate grid. With the
same proposals, the ordered exact counts moved as follows:

| network factor | query@1 | query@2 | strict@1 | strict@2 | cumulative |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed seed-31337 checkpoint | 9 | 9 | 7 | 7 | 32 |
| same logits scaled by 0.5 | 8 | 9 | 6 | 7 | 30 |
| same-schema update-700 checkpoint | 8 | 9 | 7 | 7 | 31 |
| deterministic reseeded logits | 7 | 9 | 5 | 7 | 28 |

This establishes the implementation direction and shows candidate-rank and
exact-rank-membership sensitivity. It is not yet qualification: the production
path, provenance, repeat arm, hashes, tests, and complete artifacts still have
to satisfy this specification.

## 0. Normative full-matrix profile

The accepted baseline and all four matched control arms use this fixed
evaluation profile:

| Field | Required value |
| --- | --- |
| physical LIF neurons | 4,096 |
| directed recurrent edges | 4,194,304 |
| latent steps | 60 |
| retained effort checkpoints | 0, 30, 60 |
| nominated submission effort | 60 |
| evaluation seed | 31337 |
| answer head | `checkpoint_conditioned` |

Baseline, reload/repeat, scale, same-schema trained-checkpoint swap, and
deterministic reseed must match every field in this table. The independently
trained swap checkpoint may carry its own recorded training seed, but evaluation
still uses seed 31337. The earlier 2,048-neuron, 16,384-edge, 32-step,
0/8/16/32 profile is superseded historical evidence and cannot satisfy this
acceptance contract.

## 1. Target-free model-owned answer path

Every primary candidate must be the deterministic decode of logits or routing
scores that consume all of the following:

1. target-free demonstrations and the test input;
2. state or features produced by an executed BrainTrace recurrent-network path;
3. model-owned trainable parameter leaves restored from the nominated trained
   checkpoint.

The test output is moved into a scorer-only structure before candidate
construction. The answer path receives no target grid, target shape, exactness
bit, task/source identity, transformation label, or target-derived selector.
Replacing every held-out output while preserving demonstrations and test inputs
must leave candidate bytes unchanged.

Task-local fitting or adaptation on demonstrations is allowed only when it is
target-free and continues to pass the perturbation gates below. It must not
overwrite a checkpoint parameter with a value derived solely from
demonstrations. A bounded forest may generate target-free proposal grids, but
the checkpoint-conditioned score must determine their submitted rank and slot
eligibility. Static formatting may serialize a predicted shape and row-major
colors; it must not bypass the network-owned ordering with the raw forest order.

The intended implementation retains the demonstration forest's conjunctive
inductive bias as the proposal generator, then reranks every proposal with the
fixed-weight sum of forest log probability and trained-network factorized
candidate log probability measured above. The network term is computed from
recurrent carrier or memory outputs and checkpoint-owned ETP parameters.
Formally, a counted candidate and its rank must come from a function of the form

```text
decode(f(target_free_context, recurrent_state(theta), theta))
```

and not from `decode(g(demonstrations, query))` with an unused model value
attached for provenance. The rerank coefficient is fixed at 1.0 and is not
tuned against evaluation labels.

Every emitted candidate records one of these dependency classes:

- `model_checkpoint`: eligible for primary exact scoring only after all gates;
- `demonstration_only_diagnostic`: never eligible for the primary score;
- `rule_diagnostic`: never eligible for the primary score.

Each `model_checkpoint` candidate also records the answer-head implementation
version, proposal source, ranking source, combined score components, and the
ordered parameter-leaf paths that participate in its executed path. Every
candidate counted at pass@1 or pass@2 must individually have
`model_checkpoint` submission provenance, even when its proposal source is the
target-free forest. Movement by a separately appended model candidate in slot
two cannot launder an unranked demonstration-only candidate in slot one.

## 2. Exact score acceptance

One effort checkpoint and answer-head configuration is nominated before the
complete evaluation targets are scored. The same configuration, task order,
candidate budget, and decoder are used for the baseline and all controls.

A baseline is eligible only when:

- all 419 queries from all 400 ARC-AGI-1 evaluation tasks are present under the
  retained manifest, or a future manifest explicitly records and justifies a
  different complete split;
- at most two deterministic `model_checkpoint` candidates are emitted per
  query;
- query pass@1/pass@2 and strict-task pass@1/pass@2 use exact shape and exact
  cell equality;
- `cumulative >= 16` using the four integer counts above;
- held-out targets were unavailable to training, checkpoint selection, answer
  construction, perturbation selection, and hyperparameter selection.

Shape accuracy, valid-cell pixel accuracy, logits, state movement, diagnostic
forest scores, and rule scores remain diagnostics. None can compensate for a
cumulative score below 16.

## 3. Repeat and checkpoint-perturbation matrix

The existing eval-only path is run five ways on byte-identical target-free
inputs. All arms retain the same task order, initial-state policy, effort,
candidate decoder, backend, and scorer.

### 3.1 Baseline

Load the nominated trained checkpoint with exact schema validation. No missing,
extra, reshaped, or dtype-coerced leaf may be silently skipped.

### 3.2 Repeat intact

Reload the same checkpoint and repeat evaluation. The canonical candidate-byte
digest, exact-membership digest, all four counts, and cumulative score must be
identical to baseline. Existing float32 state-RMS tolerances remain separately
reported; they do not relax candidate or exact-score repeatability.

### 3.3 Checkpoint scale

Apply a predeclared finite non-unit scale factor to every floating trainable
leaf on the recorded answer-dependency path, without retraining or changing the
checkpoint schema. The factor is fixed before evaluation-label scoring and is
recorded in the artifact. Qualification requires all of:

- a changed answer-parameter digest;
- a changed canonical candidate-byte digest;
- a changed exact-membership digest; and
- a cumulative integer score different from baseline.

NaN/Inf output, a schema mismatch, or a flat score fails this gate rather than
being interpreted as dependence.

### 3.4 Same-schema trained-checkpoint swap

Load an independently seeded trained checkpoint from another run with exactly
the same ordered leaf paths, shapes, and dtypes. Qualification requires a
changed parameter digest, changed candidate-byte digest, changed exact-rank-
membership digest, and cumulative score different from baseline.

Partial restoration, architecture drift, a different evaluation manifest, or
metadata-only checkpoint changes fail closed.

### 3.5 Deterministic parameter reseed

Deterministically reseed every trainable leaf on the recorded answer-dependency
path under the exact baseline schema using `brainstate.random`. Record that this
is an untrained causal control. Qualification independently requires a changed
parameter digest, changed candidate-byte digest, changed exact-rank-membership
digest, and cumulative score different from baseline. A passing trained swap
does not waive this gate.

### 3.6 Canonical movement definitions

Candidate bytes contain, in manifest order, each candidate's rank, predicted
height, predicted width, and row-major color bytes. The digest is explicitly
order-sensitive: reranking identical proposal sets changes it. It excludes
provenance strings, hashes, timestamps, score values, and other metadata, so a
metadata or floating-logit change cannot satisfy the gate by itself.

Exact rank membership contains the ordered per-query `(pass@1, pass@2)`
booleans and the ordered per-task `(strict@1, strict@2)` booleans. The artifact
records the changed query/task identifiers and candidate ranks in addition to
the digest. Offset swaps that change memberships but leave the cumulative total
flat still fail the explicit score-movement requirement.

## 4. Hashes and provenance

Every matrix arm records:

- source revision and dirty-state summary;
- evaluation-manifest digest and ordered task/query identifiers;
- checkpoint path or run identifier, training seed, and training configuration;
- full checkpoint SHA-256;
- an answer-parameter SHA-256 over ordered leaf path, shape, dtype, and bytes;
- individual participating-leaf hashes;
- topology/edge digest and, when used, neuron-type/Dale-mask digest;
- candidate-byte and exact-membership SHA-256 values;
- all four exact counts, cumulative score, and diagnostic metrics;
- candidate-level dependency class and answer-path version;
- perturbation kind, scale factor or alternate seed/run, and schema comparison.

The report distinguishes base checkpoint bytes, parameters after perturbation,
and any target-free task-local adapted state. Hash equality or inequality is
checked by the evaluator and not asserted from filenames.

## 5. Dale-sign evidence

Dale constraints are optional supporting evidence and do not replace checkpoint
perturbation. If an EI/Dale arm is used in the qualifying result, the report
must prove zero effective recurrent-weight sign violations, record excitatory
and inhibitory counts and assignment digest, and run a predeclared matched sign
control. A claimed Dale-dependent answer requires candidate bytes, exact
memberships, and cumulative score to move under that control. A flat sign
control is reported as null.

## 6. BrainState execution constraints

Repeated network evaluation, task-local model routing, and repeated training
steps must use `brainstate.transform.jit`, `for_loop`, `scan`, or the appropriate
checkpointed variant. A bare Python `for` or `while` loop may prepare static
host metadata or score already-produced candidate bytes, but it must never
drive repeated neuron, synapse, answer-head, or trainable-state updates.

All model, topology, augmentation, reseed, and perturbation randomness uses
`brainstate.random`. Direct `jax.random` use is forbidden. Evaluation candidate
construction itself remains deterministic after the seed and checkpoint are
fixed.

## 7. Test and evidence gates

Before a full run, co-located tests must reproduce the provenance bug and then
prove:

1. a raw demonstration-only forest ordering is classified diagnostic and
   cannot enter primary metrics, while target-free forest proposals become
   eligible only after checkpoint-likelihood reranking;
2. changing held-out targets cannot change candidate bytes;
3. every counted candidate has a nonempty, executed checkpoint-parameter
   dependency set;
4. exact-schema restoration rejects missing, extra, reshaped, or dtype-changed
   leaves;
5. candidate and membership serialization is canonical and excludes metadata;
6. repeat-intact is byte- and score-stable;
7. a flat scale, trained-checkpoint swap, or deterministic reseed independently
   fails qualification;
8. moving controls pass only when candidate bytes, memberships, and cumulative
   score all move;
9. score 15 fails and score 16 passes the threshold when every other gate is
   satisfied;
10. the changed production modules have more than 90 percent meaningful test
    coverage, including failure paths and edge cases.

The final artifact must contain one complete baseline plus repeat, scale,
same-schema trained-checkpoint swap, and deterministic reseed results. Until the
baseline reaches 16 and every gate passes, the valid parameter-dependent score
remains the last verified value; diagnostic forest performance is not promoted.

## 8. Measured full-matrix qualification

The completed 2026-08-23 matrix used clean source revision
`ccc20cf7bd13c5c86dbd274319c0ccb840a13952`, Docker image ID
`sha256:cfbb91d3195d320335779919050f36622921674a67dc3f9f44671c892b1daa9b`,
manifest SHA-256
`b9ab482f3f4f03193cf5ebd73433ab1899685797079e39fb9cf8ee88d6ad7d2f`,
and topology SHA-256
`1a32975bc76b50e459d0d3b623750ba53ca7e060555de809600fcd1a681d0460`.
Every arm ran on the GPU with 4,096 physical LIF neurons, exactly 4,194,304
directed recurrent edges, checkpoints 0/30/60, submission effort 60, and
evaluation seed 31337.

| Arm | Query @1 | Query @2 | Strict task @1 | Strict task @2 | Cumulative |
| --- | ---: | ---: | ---: | ---: | ---: |
| trained baseline | 8 | 9 | 6 | 7 | **30** |
| intact reload/repeat | 8 | 9 | 6 | 7 | **30** |
| exact 0.5x scale | 7 | 9 | 5 | 7 | **28** |
| independently trained swap | 7 | 9 | 5 | 7 | **28** |
| deterministic BrainState seed-73 reseed | 7 | 9 | 5 | 7 | **28** |

The baseline exceeds the predeclared cumulative threshold by 14. Repeat is
exact across checkpoint, parameter, topology, manifest, query-manifest,
candidate-byte, exact-membership, count, and cumulative-score evidence. Each
perturbation independently differs from baseline in parameter bytes, ordered
candidate bytes, exact membership, and cumulative score.

| Arm | Checkpoint SHA-256 | Parameter SHA-256 | Candidate SHA-256 | Membership SHA-256 |
| --- | --- | --- | --- | --- |
| baseline | `d8d97f50b62ba71e9c9b31029e3084e508c1666ca12dc7c44a6dcca4e75f93f5` | `dc9d3e2eaca3fc29e93f55e4b8f3e56d1688d094459af4246093565e479fe3ce` | `104fe7fe94f4c4bbbcf44a16ae677ceb333111037217a601254acce76fbdf08c` | `e32632118c8f99bcbb535fce9bf32a8f61399243304a1ce5852594c89e5a1949` |
| repeat | `d8d97f50b62ba71e9c9b31029e3084e508c1666ca12dc7c44a6dcca4e75f93f5` | `dc9d3e2eaca3fc29e93f55e4b8f3e56d1688d094459af4246093565e479fe3ce` | `104fe7fe94f4c4bbbcf44a16ae677ceb333111037217a601254acce76fbdf08c` | `e32632118c8f99bcbb535fce9bf32a8f61399243304a1ce5852594c89e5a1949` |
| scale | `e924008d5efe8dfda8e531d4e709adfcf66151b29e6f819172521d70254f8869` | `935cfe5735eb1be831b82e8238ea591d3de5d263d998d6b13c4cbbeaa1f5b94a` | `f0aaa0e62c1660947a5c417ef6b8ec91e8bb904dd7007b33b7c5b443de0ed4fc` | `89d97d2e70161a43e06577c6221e4ba17cdca64177ca19fe56ce1c2deafd95cf` |
| swap | `78d3e772161c8725ebdf89e8e473568c0527b782b09e72cb814a25aa6cd16607` | `104e7c338b4a0d2c9ca485fddcd33f9384c4ebc14ddf6c42d8467f10ae4dd783` | `72acefa20e0783778b0c44b500bdd39045d99101baf1930ec6b4d39cde80c28f` | `89d97d2e70161a43e06577c6221e4ba17cdca64177ca19fe56ce1c2deafd95cf` |
| reseed | `46305ce013502b68923381eff4e61a1d7744daef65d1c546f2e885458ca4bd43` | `ea0cf14772c6b51e51dd3ea5e5115160f6264f220a58e89592134a948f8e6488` | `3e800f52cb4d6b501279869d04a35dea9db14fab896f58526fa2b9fb2b6b1e3d` | `89d97d2e70161a43e06577c6221e4ba17cdca64177ca19fe56ce1c2deafd95cf` |

The authoritative artifact is `var/ex21-paramdep-v1-matrix.json`, SHA-256
`6ef287bf41fd3cd88ac00aee33c5b69088d8c3549ed0ce1f271b39a82f301581`.
It reports every qualification check true and no rejection reason. The
qualifying model uses neuron-typing mode `none`; Dale dependence is explicitly
`not_claimed`, so the conditional EI/Dale sign-control requirement is not
applicable and no Dale result is promoted.

## Non-claims

Passing this contract proves exact ARC performance and measured dependence on
the nominated checkpoint under the specified perturbations. It does not prove
that every neuron is causally necessary, that pp-prop equals BPTT, that the
private BDH-CQ architecture was reproduced, or that the mechanism generalizes
beyond the fixed evaluation manifest.
