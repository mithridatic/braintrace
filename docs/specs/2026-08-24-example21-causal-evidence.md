# Example 21 causal explanation evidence

## Purpose

This document holds the qualification rules, measurements, and technical
support for the short causal explanation in
`2026-08-24-example21-causal-phase-map.md`. The short explanation describes
observed system behavior. It does not contain an experiment history, design
diary, or list of future versions.

## Documentation requirements

The short explanation must:

1. start with the strict ARC acceptance target;
2. describe the direct path from task input to first prediction;
3. distinguish observed behavior from an inferred cause;
4. identify the point where the direct path currently fails;
5. state that synthetic, pixel, shape, forest, and pass@2 results do not count;
6. avoid version-by-version narration, repeated conclusions, and proposed-arm
   details.

## Acceptance rule

The only acceptance score is:

```text
strict_task_pass_at_1_count >= 16
```

The fixed evaluation manifest has 400 ARC tasks and 419 test queries. A task
counts only when the first model-generated grid is exactly correct for every
test query in that task. Exact means that the height, width, and every cell are
correct.

An eligible grid must come directly from logits produced by one executed,
trained BrainTrace recurrent checkpoint. A forest, rule engine, retrieval
system, template, task-local fitter, repair step, or reranker cannot create or
select a counted grid.

Query counts, pass@2 counts, shape accuracy, cell accuracy, and sums of several
metrics are diagnostics. They cannot replace the strict task pass@1 count.

## Perfect-prediction scorer check

This check answers one narrow question: if the decoder receives a uniquely
correct height score, width score, and color score for every evaluation query,
does the production decoder and strict scorer report every task as correct?

The check loads the fixed evaluation manifest. It uses each scorer-only target
to construct perfect hierarchical background, nonzero-color, height, and width
logits. It then calls the production `decode_hierarchical_online_outputs` and
`strict_task_pass_at_1` functions. It records the task count, query count,
decoded-grid mismatch count, strict score, and wall time. The run has a hard
limit of three minutes.

This is a target-fed scorer oracle. It tests the output representation, greedy
decoder, and strict scorer. It is not a model evaluation and must never be
reported as model-generated ARC performance.

The completed check used image
`braintrace-gpu:0.11.0-py314-msgspec-arc` and the fixed manifest at
`/datasets/arc/example21-sources.json`. It exited successfully with:

```json
{
  "decoded_grid_mismatch_count": 0,
  "decoder": "decode_hierarchical_online_outputs",
  "evaluation_query_count": 419,
  "evaluation_task_count": 400,
  "failed_task_count": 0,
  "oracle_kind": "target_fed_perfect_hierarchical_logits",
  "strict_task_pass_at_1_count": 400,
  "wall_seconds": 6.261481215013191
}
```

The complete Docker command took 18.547 seconds, including container startup.
Thus, uniquely correct height, width, and color logits are sufficient for the
implemented decoder and scorer to mark all 400 tasks correct. This check says
nothing about whether a trained model can produce those logits.

## Current direct model facts

The latest completed direct artifact uses architecture
`query_routing_gated_memory_v48`. It has two independent dense MiniLSTM
memories. Each memory has 128 hidden units. One memory updates during the
demonstration phase. The other updates during the query and decode phases.

The model joins the two 128-value states and their elementwise product into a
384-value relation state. Twelve color experts and 16 learned routing programs
use this state. Separate learned heads produce height, width, and cell scores.

The recorded checkpoint contains 19 trainable leaves and 12,533,828 trainable
numeric values. Most values are in dense connection matrices. This model does
not store a sparse recurrent graph, so it has no declared neuron-to-neuron edge
or synapse count.

The MiniLSTM hidden units are mathematical state variables, not physical LIF
neurons. Their dense weights have no excitatory/inhibitory type and no Dale-sign
constraint. Older Example 21 spiking architectures support explicit recurrent
edges and optional EI/Dale typing. That fact does not apply to this direct
MiniLSTM result.

## Current direct-system measurements

The latest completed direct-system artifact is
`var/ex21-online-v48-query-routing-v1/result.json`.

| Scope | Tasks | Queries | Strict tasks at pass@1 | Meaning |
|---|---:|---:|---:|---|
| Synthetic holdout | 120 | 120 | 12 | Development evidence only |
| Real in-library development | 51 | 53 | 0 | Real-task evidence |
| Real fold-zero development | 80 | 85 | 0 | Real-task evidence |
| Fixed ARC evaluation manifest | 400 | 419 | Not run | Required acceptance scope |

The 12 synthetic solves comprise eight copy tasks, one count task, one pattern
label task, one marked-region selection task, and one upscale task. The upscale
task is the first exact synthetic routing solve recorded for this direct line.
It proves that the added query-routing path can affect an exact output. It does
not prove useful coverage of spatial ARC transformations.

The artifact records `source_dirty=true`. Treat it as development evidence,
not as a reproducible qualification artifact.

The direct system predicted the correct shape for 86/120 synthetic holdout
queries. It predicted the correct shape for 22/53 real in-library queries and
51/85 real fold-zero queries. None of the real development tasks was strict
exact. These results show why shape and cell diagnostics cannot stand in for
the acceptance score.

## Evidence for the failure location

The observed output pattern separates three parts of the system:

- The encoder carries useful task information. A frozen-state probe can read
  several synthetic operator labels above chance.
- The output path can produce exact copy and small-label answers in some
  synthetic tasks.
- The output path rarely moves the correct query content to a different output
  position. The query-routing system produced only one exact synthetic upscale
  result and no strict real development task.

This supports a limited conclusion: the current break is between useful task
state and reliable exact-grid generation. It does not prove that every failed
task was recognized correctly. Recognition and rendering can both fail.

The matched full reverse-mode training artifact,
`var/ex21-online-v47-bptt-matched-arm-v1/result.json`, solved 11/120 synthetic
holdout tasks. Its solved-family pattern matched the PP-Prop system: copy and
small global-label tasks passed, while spatial-routing families did not. This
result makes the gradient method an unlikely explanation for the current
family-level failure. It does not prove that PP-Prop and BPTT are equivalent in
general.

## Qualification boundary

The older checkpoint-conditioned system scored 6/400 strict tasks at pass@1,
but a demonstration forest generated its candidate grids. The checkpoint only
changed their order. That result is a hybrid diagnostic and is not a
model-generated ARC result.

No completed direct-system artifact has passed the fixed 400-task evaluation
and the required checkpoint-dependence controls. Therefore Example 21 does not
currently satisfy the acceptance rule.
