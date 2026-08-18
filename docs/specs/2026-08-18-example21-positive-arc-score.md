# Example 21 positive exact ARC score

Status: approved performance-first implementation

Date: 2026-08-18

Branch: `feat/example21-latent-reasoning`

## Objective

Produce at least one exact ARC-AGI-1 evaluation query through the intact
Example 21 model while retaining 2,048 LIF neurons, 16,384 recurrent synapses,
pp-prop training, contextual fast-weight memory `S_K`, and recurrent workspace
`H_r`. Exact query pass@1 or pass@2 greater than zero is the success condition;
shape or pixel diagnostics alone are not success.

## Evidence motivating the change

The retained 10,000-update run used `context_memory_width=0` and therefore
tested the legacy reservoir rather than the associative architecture. Its
highest pixel diagnostic still had zero exact queries. The closest reported
query predicted the correct 15 by 15 shape but an all-background grid, missing
all five foreground cells. The current color objective averages valid cells,
so a frequent background color dominates sparse targets. Candidate two changes
only one lowest-margin decision and cannot repair a multi-cell error.

The existing width-32 architecture has already passed controlled binding and
demonstrated-depth gates. This experiment therefore tests the shortest
performance path rather than repeating the invalid Gate C3 artifact. A
positive score from this lane is a performance result only; it does not
retroactively qualify C3 or support a causal-mechanism claim.

## Change

Add an opt-in class-balanced valid-cell color loss. Within each target grid,
every color present inside the valid shape contributes equal total weight,
while the height and width objectives are unchanged. The legacy loss remains
the default for compatibility. The experiment enables balancing explicitly
and enables `context_memory_width=32`, `memory_decay=1.0`.

No evaluation target may enter training, candidate construction, checkpoint
selection, or hyperparameter selection. ARC evaluation labels are used only by
the frozen scorer after training.

## Focused gates

1. Co-located loss tests prove that balanced and legacy losses agree for a
   one-color target and that a rare-color error receives more relative weight
   only when balancing is enabled.
2. Existing memory-model and entry-point tests covering configuration,
   compilation, and packed training remain green in the focused lane.
3. A tiny GPU smoke must compile pp-prop with width-32 memory and produce finite
   loss before the bounded full-data run.
4. The retained run reports exact pass@1/pass@2, query identities, configuration,
   source revision/dirty state, and parameter movement. A zero score remains a
   valid negative result and must not be relabeled from pixel accuracy.

## Deferred work

Fresh Gate C4 causal qualification, structured multi-hypothesis candidate
generation, public synthetic curricula, slow ALIF state, horizon-matched trace
decay, persistent test-time learning, consciousness claims, human-neuron YAML,
and Example 20 pruning are outside this bounded run. If exact score remains
zero, structured candidate construction and foreground/object decoding are the
next performance changes; neuron/edge scaling is not.
