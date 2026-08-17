# Example 20 · Batched topology minimization

Date: 2026-08-16
Status: implementation specification
Supersedes: the "Coarse-to-fine pruning frontier" and "Joint neuron-and-edge
fixed point" sections of `2026-08-16-post-training-neuron-pruning.md`. Every
other section of that document — functional removal, edge accounting, physical
compaction, symmetry alignment, reporting — stands unchanged.

## Question

Example 20 already answers *how small* the trained network can get. This
document answers *how fast that answer can be reached*, without weakening the
guarantee it certifies.

## What the guarantee actually constrains

The published result is a property of the **final mask**, not of the search
path:

> removing any one retained neuron or any one retained recurrent edge lowers at
> least one task below target.

That is 1-minimality, the output contract of delta debugging (Zeller &
Hildebrandt 2002). Nothing in it requires that candidates were tested one at a
time, in score order, or in alternating neuron and edge phases. The superseded
specification nevertheless prescribed exactly that:

> Candidates in one pass are therefore tested sequentially, not independently
> against a stale mask and not removed as an unsafe simultaneous batch.

That sentence is withdrawn. It described one admissible path to a 1-minimal
mask and mistook it for the requirement. The requirement is the terminal
certificate, and the certificate is *n independent single-ablation tests against
one fixed mask* — a shape that batches exactly.

A consequence to state plainly: a different search path reaches a **different**
1-minimal mask. The recorded 174 neurons / 234 edges will move. That is not a
regression provided the certificate holds; only a broken certificate is.

## Why the current implementation is slow

Three costs, of which only two were previously identified.

1. **Trials.** One probe evaluation runs each of the 64 fixed probe trials
   through its own rollout at `batch_size=1`. The trials are mutually
   independent, so 63 of those 64 rollouts are avoidable serialization of work
   the device could do at once.
2. **Candidates.** `_joint_fixed_point_prune` nests four loops — cycles, phases,
   passes, candidates — and the innermost one spends a full probe evaluation per
   candidate coordinate. The recorded run executed roughly 6,000-7,000 probe
   evaluations, each ~1,984 sequential model steps, to travel 1934 → 323 → 174
   neurons and 1604 → 242 → 234 edges. Screening the first pass alone is about
   3.8 million sequential batch-1 model steps.
3. **Repeated compilation.** `_evaluate_structural_masks` builds a *fresh*
   evaluator and a fresh `brainstate.transform.jit` on every call, at whatever
   leading mask count it was handed. `_analyze_pruning` calls it three times
   with counts 1, ~21 and ~103; `_analyze_compaction` adds two
   `_evaluate_probe_logits` calls and two `_benchmark_compaction` programs; the
   search adds its own. That is roughly eight independent XLA compilations of a
   nested 64-trial rollout in one run.

The arithmetic on the recorded run: `_benchmark_compaction` already divides by
its repetition count, so the recorded 32.764 ms is the cost of **one** full
probe evaluation, not ten. Approximately 6,500 evaluations therefore account for
about 213 s of the 718.0 s round. The remainder is training plus that
compilation tax — which is why evaluator caching is a first-class part of this
change and not an afterthought.

## Batched probe evaluation

The probe path is inference-only: no learner, no eligibility traces, no
gradients. It may therefore be evaluated with a leading batch axis that carries
**both** trials and candidate masks, flattened into one dimension of size
`n_candidates × n_trials`.

Four facts make this faithful rather than approximate, and each is verified by a
co-located test rather than assumed:

1. `braintrace.nn.SparseLinear` accepts a leading batch axis in the forward
   direction. The absent `brainunit.sparse` batching rules noted in
   `examples/drtrl/README.md` block `vmap`, not a batch dimension, so no
   `vmap` and no framework work is required.
2. A per-candidate edge mask cannot go through that op, because it takes a
   single `(n_edge,)` weight vector. It is replaced, for the probe path only, by
   a gather / scatter-add product

   ```
   current[b, cols[e]] += spikes[b, rows[e]] * values[e] * edge_mask[b, e]
   ```

   which is exactly equal to the sparse operator when the mask is all ones, and
   which scatters strictly within a batch row.
3. Every other stage of the network — the dense feed-forward projection, both
   exponential synapses, the CUBA output, the LIF population, and the leaky rate
   readout — already carries a batch axis through
   `brainstate.nn.reset_all_states(model, batch_size=B)`.
4. **Neuron ablation needs no per-candidate edge mask at all.** A silenced
   neuron emits no spike into the recurrent projection and none into the
   readout, so its incident edges already transmit nothing; the induced edge
   mask is bookkeeping, not dynamics. Neuron screening therefore keeps the
   native sparse operator with one edge mask shared across the whole candidate
   batch, and the `(batch, n_edge)` scatter intermediate never materializes.
   Only *edge* screening needs the per-row masked product.

**Numerical fidelity, stated precisely.** Trial batching is bitwise identical to
the serial rollout. Candidate batching leaves the spiking dynamics bitwise
identical — per-task spike rates match at every batch size, so no threshold
crossing moves — while the dense readout's reduction order changes, shifting
logits by up to ~1e-7. The reported quantity is an argmax over logits, which is
invariant to a perturbation that small except at an exact tie. Accuracy, and
therefore every acceptance decision and the certificate itself, is unchanged.
The tests assert bitwise equality on rates and exact equality on predictions,
and a tolerance only on raw logits.

**Where the speedup comes from.** This is a kernel-launch-latency effect, not a
FLOP effect: the recorded GPU run spends ~17 µs per model step on 2,048 neurons,
which is slower per step than a CPU at a quarter of the width. Batching pays on
GPU and does not pay on CPU. The co-located tests therefore assert *evaluation
counts*, which are device-independent, and never wall time.

The alive mask keeps both of its current applications: once on the spike vector
entering the recurrent projection, once on the spike vector entering the
readout. The reported per-task rate stays the mean of the second, masked spike
train, so contribution scores are unchanged.

The masked projection is installed on the existing model and removed afterwards,
so the trained recurrent values are restored exactly as before and post-hoc
analysis still cannot perturb the qualifying checkpoint.

**Accumulate in a carry, do not stack.** The current rollout stacks every step's
spike vector only to take its mean. At 2,048 model batch rows that intermediate
is over half a gigabyte. The batched rollout uses
`brainstate.transform.scan` with a running rate and logit sum instead, which is
the primitive AGENTS.md rule 10 names for a loop with an explicit carry and
which bounds the rollout's memory at one `(batch, n_rec)` array.

**Compile once per mask mode, not once per call site.** The probe trials are
memoized per configuration, and one runner holds the compiled programs for the
whole search: a shared-edge-mask rollout, a per-row rollout, and one screen
driver for each. Every call is padded to a fixed candidate batch with copies of
the current accepted mask, whose outputs are discarded, so the baseline
evaluation, the frontier sweep, every screen and every commit test reuse those
same programs instead of compiling a fresh one at each leading mask count. The
compaction verification and benchmark still compile their own single-mask
programs, as they must — they run against a different, narrower model.

`--eval-batch` bounds the candidate batch (default 32). Peak memory is roughly
57 KB per model batch row, so the default is about 120 MB at the recorded
configuration; the value is reported alongside the timing so a run states what
it used.

## Screen-and-accept search

The cycle / phase / pass / candidate nest is replaced by a single loop over one
**unified coordinate set**: the retained neurons and the retained
retained-to-retained edges together. Separate neuron and edge phases existed to
let each rerun after the other changed; one unified set gets that for free.

Each round:

1. **Screen.** Ablate every retained neuron individually against the current
   accepted mask, in `ceil(n_retained / eval_batch)` batched evaluations. Fall
   through to the active recurrent edges only when no neuron is removable.
   Removing a neuron disables its incident edges for free, so screening every
   active edge first would spend thousands of causal trials on edges a later
   neuron removal deletes anyway — measured at roughly five times the total
   evaluation count on a 512-neuron run. Because the terminal round finds no
   removable neuron by definition, it always screens both.
2. **Certify or continue.** If nothing is individually safe, the mask is
   1-minimal and the search is done. That screen *is* the terminal
   single-ablation certificate; it is not a separate pass.
3. **Accept.** Order the individually-safe set ascending by contribution score,
   then index. Try removing the whole prefix at once. If every task stays at or
   above target, accept all of it; otherwise binary-search the largest prefix
   length whose simultaneous removal is verified safe. Then retry the leftover
   against the newly reduced mask, and keep going until the safe set is drained.
   A screen costs one evaluation per retained coordinate, so it is worth several
   cheap prefix probes to postpone the next one.
4. Removing a neuron disables its incident edges in the same step, as before: a
   structural consequence, not a causal claim, requiring no separate trial.

Removing a set is *not* implied by each member being individually safe —
interactions exist, and the prefix search is what handles them. A coordinate the
screen called safe may also fail once earlier removals have landed; that is
measured, recorded in place of the screen's optimistic number, and the
coordinate is dropped. Acceptance is decided by measured probe accuracy at every
step; contribution scores only order the prefix and never decide a removal.

**Termination, without any monotonicity assumption.** A prefix of length one is
known safe: its single coordinate was screened safe against this very mask. So
every non-terminal round removes at least one coordinate, the retained set
strictly shrinks, and the search terminates in at most `n_rec + n_edges` rounds.
The existing iteration cap is retained as a backstop. The empty screen is the
stopping condition and simultaneously the certificate.

**Complexity caveat.** Delta debugging's efficiency bound assumes the safety
predicate is monotone in the removal set, and the superseded specification
correctly notes that accuracy under lesions is not monotone. Without
monotonicity the binary search finds *a* safe prefix rather than provably the
longest, which costs an extra round and nothing else. Termination and
1-minimality are unaffected; only the speedup is empirical rather than proven.

## Where the compiled boundary sits

AGENTS.md rule 10 forbids driving a model with a bare Python loop that runs
repeatedly. The screen — the dominant cost, one evaluation per retained
coordinate — runs entirely inside compiled `brainstate.transform` control flow:
a `scan` over time steps inside a `for_loop` over candidate chunks. Python
drives only the round loop and the commit prefix search, which together issue on
the order of `rounds × log(n)` calls to that one compiled program, all at the
same shapes so nothing recompiles. No Python loop performs a per-candidate model
call.

## Structurally free removals

Two classes of coordinate cannot influence any output, by construction rather
than by measurement, and are removed without a rollout:

- a retained neuron whose live outgoing edges and readout weights are all zero;
- a retained edge whose presynaptic rate is identically zero on every probe.

These are structural consequences in the same sense the superseded document
already grants to edges incident to a removed neuron. They are still subject to
the terminal screen, so admitting them cannot weaken the certificate.

## Progressive compaction

The physical compactor is no longer used only at the end. When the retained
count falls below half the current recurrent width, the search rebuilds a
compact model, verifies its logits against the masked model at the existing
`rtol=1e-5` / `atol=1e-6` tolerance with identical predictions, and continues in
the smaller coordinate space. Retained indices are carried back to original
neuron indices for every reported quantity. A failed verification fails closed;
the search does not proceed on an unverified model.

This changes only where evaluations happen, never which coordinates are safe.

## Reporting

The plain-text report and the three-panel figure keep their contracts. Field
changes in the `neuron_pruning` mapping:

- unchanged in meaning: `converged`, `final_alive_mask`, `final_edge_alive_mask`,
  `retained_indices`, `retained_edge_indices`, `final_accuracies`, every
  `final_*_scores` and `*_owners` entry, and the four edge-accounting counts;
- re-interpreted as per-round quantities on the unified coordinate set:
  `cycle_count`, `neuron_pass_count`, `edge_pass_count`,
  `neuron_accepted_per_pass`, `edge_accepted_per_pass`;
- `accepted_neurons` and `accepted_edges` keep their meaning — acceptance order —
  now ordered by round and then by coordinate index within the round;
- added: `evaluation_count`, `screen_evaluations`, `commit_evaluations`,
  `round_count` and `eval_batch`, so the cost of a run is reported rather than
  asserted, and so a test can assert it device-independently.

`accepted_accuracies` needs one explicit note. It used to be the accuracy of the
mask immediately after each single accepted removal, making a cumulative curve
with one point per removed coordinate. It is now each coordinate's *screen*
accuracy — its single ablation against that round's accepted mask. Same shape,
same units, same honest answer to "what did removing this cost", but the
accuracy-versus-retained-count curve behind the second figure panel becomes one
point per round rather than one per coordinate. It is coarser by design.

New flags: `--eval-batch` (default 32) and `--prune-profile`, which prints the
phase timing split and the compilation count.

The coarse-to-fine sweep no longer seeds the search; the first round reaches the
same place directly. It is retained only as the accuracy-versus-retained-count
series behind the second figure panel, evaluated in one batched sweep.

## Required tests

Beyond the existing coverage, which stands:

- batched evaluation equals the current serial evaluator element-wise on both
  logits and per-task rates, for the same masks on a small configuration;
- the masked gather/scatter projection equals the sparse operator under an
  all-ones mask, and scatters strictly within a batch row;
- a candidate batch whose members are genuinely unsafe is still rejected, so
  shared state across the batch cannot masquerade as success;
- the terminal screen certifies 1-minimality: every retained neuron and every
  retained edge, ablated singly, drops at least one task below target;
- batch acceptance falls back to bisection when the individually-safe set is not
  jointly safe, and the accepted subset is verified rather than assumed;
- structurally free removals are identified without a rollout and survive the
  terminal screen;
- progressive compaction preserves logits within tolerance and fails closed on
  mismatch;
- `--eval-batch` rejects invalid values and chunking does not change results.

Focused tests must exceed 90 percent statement coverage for Example 20.

## Recorded substantive result

To be filled by the GPU run, against these baseline figures from the superseded
document: 718.0 s round-4 wall time, approximately 6,500 probe evaluations, 174
retained neurons, 234 active edges, four tasks at 100 percent, maximum absolute
compaction logit error `1.91e-06`, persistent storage 485,748 → 28,324 bytes.

## Release boundary

Complete when this specification, the implementation, the co-located tests, and
the README entry are committed on a worktree branch; the focused tests and the
normal example gate pass; and one substantive GPU run has recorded its wall
time, evaluation count, retained counts, and certificate against the baseline
above.
