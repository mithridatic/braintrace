# Example 20 · Batched probe evaluation

Date: 2026-08-16
Status: implementation specification
Amends: `2026-08-16-post-training-neuron-pruning.md`. The question, the causal
lesion semantics, the contribution ranking, the coarse-to-fine frontier, the
joint neuron-and-edge fixed point, the edge accounting, the physical compaction
and the reporting contract all stand unchanged. Only how one probe evaluation
is executed changes, plus one recorded number and one caveat.

## Question

Example 20 answers how small the trained network can get. This document asks
how quickly a single causal evaluation can be measured, without altering which
network the search converges on.

## What was slow

`_probe_logit_evaluator` ran the fixed probes one trial at a time at
`batch_size=1`, inside a `for_loop` over trials, inside a `for_loop` over time
steps. At Example 18's four-task configuration that is 64 trials of 31 steps, so
**one causal evaluation is about 1,984 sequential model steps** on a
2,048-neuron network. The greedy fixed point performs thousands of them.

The probe trials are mutually independent. Sixty-three of those sixty-four
rollouts are avoidable serialization of work the device can do at once.

## What changes

The trials move onto the model's batch axis. One rollout of `n_step` steps at
batch `n_trial` replaces `n_trial` rollouts of `n_step` steps at batch one.

Three facts make this available, and each is verified by a co-located test
rather than assumed:

1. `braintrace.nn.SparseLinear` accepts a leading batch axis in the forward
   direction. The absent `brainunit.sparse` batching rules recorded in
   `examples/drtrl/README.md` block `vmap`, not a batch dimension — which is
   why no `vmap`, no `State` axis handling and no framework work is needed, and
   why the state-aliasing failure mode that batching usually invites does not
   arise here.
2. Every other stage — the dense feed-forward projection, both exponential
   synapses, the CUBA output, the LIF population and the leaky rate readout —
   already carries a batch axis through
   `brainstate.nn.reset_all_states(model, batch_size=B)`.
3. The alive mask keeps both of its applications, on the spike vector entering
   the recurrent projection and on the spike vector entering the readout, and
   the per-task rate stays the mean of the second. Contribution scores are
   therefore computed from exactly the same quantity as before.

Spike rates accumulate in a `brainstate.transform.scan` carry, because stacking
them would cost one `(n_trial, n_rec)` array per time step. Readout outputs are
stacked and reduced in one pass instead, which preserves the logit summation
shape the unbatched evaluator used.

**Priming.** A compiled `for_loop` or `while_loop` carries model state, so the
state shape must match the rollout's batch *before* the loop is traced; the
reset inside the rollout then reproduces that shape. The evaluator therefore
exposes a `prime` hook that the mask sweep and the fixed-point search call once
before entering their loops. An evaluator that drives no model — a synthetic
oracle in a test — exposes no hook and needs none.

**The search is untouched.** It compares fixed-probe accuracy, not logits, and
every existing test passes unchanged. This is deliberate: an earlier attempt to
also batch *candidates* is recorded below as rejected.

## Where batching stops being exact

On CPU, batching is exact: per-task spike rates are bitwise identical to the
unbatched rollout, so no threshold crossing moves.

On GPU it is not. Batch size changes which kernels XLA selects, and that
rounding is occasionally enough to move a marginal membrane potential across the
spike threshold. One flipped spike shifts a logit by about 1e-4. Measured
without training on untrained networks at 32, 256 and 2,048 neurons, a batched
masked model differs from its compacted rebuild by up to `1.4e-04` where the
unbatched rollout differs by `6e-08` — while predictions and per-task accuracies
stayed identical in every case.

Consequently **the physical-compaction equivalence check keeps the unbatched
rollout**, and so does its timing benchmark. That check compares raw logits at
`rtol=1e-5` / `atol=1e-6`, which a flipped spike blows straight through, and it
is a published fail-closed guarantee that predates this change. It runs a
handful of times per analysis, so there is nothing to gain by batching it and a
real check to lose. The search batches because its decisions are argmax
comparisons, invariant to perturbations of this size except at an exact tie.

When compaction does fail closed, the error now names which of the three
conditions failed and reports the measured logit error, rather than stating only
that something did.

## Rejected: batching the candidates

A screen-and-accept search was implemented and measured, in which each round
ablated every retained coordinate individually against the current mask in
batched calls, then removed the largest verified-safe prefix of whatever
screened safe. The design is sound on its own terms — the terminal screen
doubles as the 1-minimality certificate at no extra cost, it certified
correctly, and it produced a *smaller* network at 512 neurons (18 neurons and 16
active edges against the greedy search's 20 and 19).

It is rejected because it converges on a materially worse network at scale. On
one 2,048-neuron, two-task checkpoint with 18,022 stored edges, the greedy
search retained **19 neurons and 6 active edges** while screen-and-accept
retained **74 neurons and 21 active edges**, at 10,284 causal evaluations
against roughly 4,700, for an analysis phase of 12.4 s against 14.5 s — a
difference inside the run-to-run variance of the same GPU.

The reason is structural, not a matter of tuning. Sequential-within-pass greedy
tests each candidate against the *cumulative* mask, so by the time it reaches
candidate `i` the earlier removals have landed and coordinates that only become
removable after others are gone get caught in the same pass. A screen against a
fixed mask cannot see those cascades; repeated rounds recover them only
partially. Two 1-minimal sets can differ by a factor of four, and the greedy
path finds a much better one.

That coupling is the whole finding: candidate batching pays only if the search
tests independent candidates, which is precisely what costs it the cascade
discovery. Trial batching has no such coupling, which is why it is what shipped.

If the idea is revisited, the targeted repair is to rescreen only the graph
neighbours of each removed neuron after an acceptance, since those are the only
coordinates whose removability can have changed.

## Recorded measurement

The `2026-08-16` record reports 718.0 s for the round-4 wall time including
pruning, compaction, compilation and timing. That figure carries a correction:
`_benchmark_compaction` already divides by its repetition count, so the
`32.764 ms` recorded alongside it is the cost of **one** full probe evaluation,
not ten.

Independently measured on the same hardware at 2,048 neurons, 16,384 initial
edges, two tasks and a 100 percent target, the whole post-training analysis —
frontier sweep, joint fixed point, compaction, verification and benchmark —
takes about 14.5 s unbatched. The pruning phase was never the several hundred
seconds the record implies at the configurations that can be measured directly;
whatever dominated that run, it was not the number of causal evaluations alone.

## Required tests

Beyond the existing coverage, which stands unchanged:

- the batched evaluator matches the unbatched rollout for the same masks, with
  per-task rates compared bitwise and predictions compared exactly (these run on
  CPU, where the equality is exact);
- a dead mask silences every trial in the batch, not one, and a live mask does
  not — the check that would fail if state were shared across the batch;
- priming sizes every model state to the probe batch, and an evaluator that
  drives no model is left alone;
- the probe trials are memoized per configuration object.
