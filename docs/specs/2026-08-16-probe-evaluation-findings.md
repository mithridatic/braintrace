# Example 20 · Probe-evaluation findings

Date: 2026-08-16
Status: findings, with two rejected optimizations
Amends: `2026-08-16-post-training-neuron-pruning.md`, whose recorded timing is
corrected below. Every other part of that document stands.

## Question

Example 20's post-training analysis reads as expensive: it evaluates the fixed
probes once per candidate coordinate, and the record reports 718.0 s. Can a
better algorithm reach the same minimal network in far fewer, or far cheaper,
causal evaluations?

Measured answer: **no, and it does not need to.** The analysis takes about
13 seconds at the recorded configuration. Both optimizations that looked
promising were implemented, measured, and rejected because each traded a
materially worse minimal network for no measurable time.

## The recorded 718.0 s is training, not pruning

Two corrections to the record.

`_benchmark_compaction` already divides its elapsed time by the repetition
count, so the `32.764 ms` recorded there is the cost of **one** full probe
evaluation, not ten.

More importantly, the same code was re-run at the recorded configuration —
2,048 neurons, 16,384 initial edges, 12 rounds, four temporal-credit tasks,
100 percent target — and reproduced the published result exactly: the round-4
checkpoint with 21,806 stored edges, 174 retained neurons, 631 original
retained-to-retained edges, 397 causally removed, 234 active, and persistent
storage falling 485,748 → 28,324 bytes. The split was:

| phase | wall time |
|---|---|
| Example 18 training, 24 rounds across both arms | 1,369.9 s |
| **whole post-training analysis** | **13.4 s** |

The 13.4 s covers the frontier sweep, the joint neuron-and-edge fixed point,
physical compaction, its equivalence verification, and the warmed timing
benchmark. The 718.0 s in the record is a training round with the analysis
folded into it, not the cost of pruning. There is no evaluation-count problem
to solve.

## Rejected: batching the candidate masks

A screen-and-accept search was implemented in full. Each round ablated every
retained coordinate individually against the current mask in batched calls,
then removed the largest verified-safe prefix of whatever screened safe, found
by sweeping candidate prefix lengths in parallel. It rests on a correct
observation: the published guarantee is 1-minimality, a property of the **final
mask, not the search path** — the output contract of delta debugging (Zeller &
Hildebrandt 2002) — so the terminal screen doubles as the certificate at no
extra cost, and the path may be as speculative as we like.

It worked. It certified correctly, and at 512 neurons it found a *smaller*
network than the greedy search: 18 neurons and 16 active edges against 20 and
19.

It is rejected because it converges on a materially worse network at scale. On
one 2,048-neuron, two-task checkpoint with 18,022 stored edges:

| | greedy | screen-and-accept |
|---|---|---|
| retained neurons | **19** | 74 |
| active recurrent edges | **6** | 21 |
| causal evaluations | ~4,700 | 10,284 |
| analysis phase | 14.5 s | 12.4 s |

The time difference is inside the run-to-run variance of the same GPU, measured
at about 10 percent on identical training work.

The cause is structural, not a matter of tuning. Sequential-within-pass greedy
tests each candidate against the **cumulative** mask, so by the time it reaches
candidate `i` the earlier removals have landed, and coordinates that only become
removable once others are gone are caught in the same pass. A screen against a
fixed mask cannot see those cascades; repeated rounds recover them only
partially. Two 1-minimal sets can legitimately differ by a factor of four, and
the greedy path finds a much better one.

That coupling is the whole finding: **candidate batching pays only if the search
tests independent candidates, which is exactly what costs it the cascade
discovery.** If the idea is revisited, the targeted repair is to rescreen only
the graph neighbours of each removed neuron after an acceptance, since those are
the only coordinates whose removability can have changed.

## Rejected: batching the probe trials

The 64 fixed probe trials are mutually independent but run one at a time at
`batch_size=1`, so one causal evaluation is about 1,984 sequential model steps
rather than 31. Putting them on the model's batch axis needs no `vmap` and no
framework work: the absent `brainunit.sparse` batching rules recorded in
`examples/drtrl/README.md` block `vmap`, not a leading batch dimension, and
`braintrace.nn.SparseLinear` accepts one natively. This was implemented, and
every existing test passed unchanged.

It is rejected for a different reason, and the reason is the more interesting
result of the two.

**On GPU, batch size changes the answer.** Different batch sizes select
different XLA kernels, and that rounding is occasionally enough to move a
marginal membrane potential across the spike threshold. One flipped spike shifts
a logit by about 1e-4. Measured without training on untrained networks at 32,
256 and 2,048 neurons, a batched masked model differs from its compacted rebuild
by up to `1.4e-04` where the unbatched rollout differs by `6e-08`. On CPU the
same comparison is exact — per-task spike rates are bitwise identical at every
batch size — so this is a property of GPU kernel selection, not of the batching
itself.

Two consequences, both measured:

1. The physical-compaction equivalence check compares raw logits at `rtol=1e-5`
   / `atol=1e-6`. A flipped spike blows straight through that, and the check
   failed closed — correctly — on the first 2,048-neuron run.
2. More seriously, the perturbation reorders the greedy search. At the same
   2,048-neuron checkpoint where the unbatched search deterministically retains
   **19** neurons across repeated runs, the trial-batched search retained **41**,
   while the analysis phase moved only 14.5 s → 13.5 s.

So this network's greedy path is numerically fragile: any change that perturbs
per-evaluation numerics costs result quality on GPU. That is worth knowing
before anyone else optimizes this evaluator.

## What was kept

Only changes that touch no numerics:

- the probe trials are memoized per configuration object, since one analysis
  hands the same configuration to five evaluators and rebuilding them is a
  Python loop over every trial;
- when physical compaction fails closed it now names which of its three
  conditions failed and reports the measured logit error, rather than stating
  only that something did — this is what made the batching failure diagnosable;
- `_probe_logit_evaluator` documents why it stays unbatched, so the next reader
  does not repeat the experiment.

`np.add.at` in `_contribution_scores` was deliberately left alone. `np.bincount`
is the faster idiom, but it changes a summation order in the quantity that
orders greedy candidates, which is precisely the class of perturbation that cost
19 → 41 above.

## If Example 20 does need to get faster

The evaluation count is not the lever; 13.4 s of analysis is not worth
restructuring. The levers, in order of expected value:

1. **Example 18 training**, which is 99 percent of the recorded run.
2. **XLA compilation.** A run at this configuration logs about 110 compilations
   totalling ~16 s, and the fixed point's four-deep `while_loop` nest is the
   largest single program. Shrinking or caching it touches no numerics and no
   search path — the one optimization here that cannot cost result quality.
