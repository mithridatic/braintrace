# Example 20 · Post-training neuron-and-edge pruning

Date: 2026-08-16
Status: implementation specification

## Question

Once Example 18's structurally evolving recurrent network first reaches a
requested accuracy on every task, how many recurrent neurons can be
functionally removed without lowering any task below that accuracy, and do the
removable neurons align with Example 19's exact structural twins or
task-specific wiring?

This is a post-training causal lesion experiment. It does not introduce a new
learning rule, retrain after pruning, prove that the selected subnetwork is
globally minimum, or claim that fixed-probe accuracy generalizes to unseen
data. It establishes coordinate-wise local minimality for both recurrent
neurons and recurrent edges on the fixed probes once the joint fixed-point
phase converges.

## Inputs and execution

Example 20 defaults to Example 18's four-task temporal-credit configuration;
explicit Example 18 task arguments override that default. It forwards the
remaining training arguments unchanged and adds:

- `--prune-target`, the minimum fixed-probe accuracy required for every task
  after each lesion (default `1.0`);
- `--prune-step-fraction`, the coarse fraction of neurons removed between
  pruning checkpoints (default `0.05`); and
- `--pruning-plot-output`, the pruning figure path (default
  `neuron_pruning.png`); and
- `--compact-model-output`, the physically compacted inference bundle path
  (default `compacted_network.npz`);
- `--compact-benchmark-repetitions`, the number of compiled full-probe repeats
  used for each timing measurement (default `10`); and
- `--device`, one of `gpu`, `cpu`, or `auto` (default `gpu`).

An explicit device request is fail-closed. In particular, the default GPU run
must raise a clear error if JAX cannot bind a GPU; it must never silently emit
CPU timings or results under a GPU-default command. `auto` deliberately accepts
whatever backend JAX selects.

Example 18 also exposes `--neurons`, `--initial-edges`, `--n-rounds`, and
`--trials-per-round` so width, starting sparse density, and training budget can
be varied without changing source constants.

Example 18 exposes two narrow optional callbacks. A checkpoint callback runs
after each evolving-arm accuracy evaluation and before any topology rebuild;
Example 20 uses the first checkpoint where every task meets `prune_target`.
The existing post-training callback provides a fail-closed final evaluation if
no checkpoint qualified. With no callbacks, Example 18's behavior and return
schema remain unchanged. This distinction is load-bearing: Example 18 records
round accuracy before its structural controller rebuilds the graph, and a
growth or pruning rebuild can change the model that would otherwise be handed
to post-hoc analysis.

Repeated trials, time steps, and lesion masks must execute through
`brainstate.transform` primitives. Python may construct deterministic probe
data and masks, but must not drive repeated model calls.

## Functional removal

A binary alive mask is applied at every recurrent step. A silenced neuron may
still receive current internally, but its spike is masked before both the
recurrent projection and readout. It therefore cannot influence any surviving
neuron or output. Keeping the original tensor shapes avoids recompiling a new
model dimension after each removal and is functionally equivalent to deleting
the neuron for inference in this architecture.

A second binary mask multiplies the frozen recurrent CSR values during each
probe evaluation. An edge with mask value zero therefore transmits no current.
The evaluator restores the original trained values before returning so the
post-hoc analysis cannot alter the qualifying checkpoint or Example 18's
subsequent training. The returned masks describe a compactable subnetwork; the
experiment does not rebuild the checkpoint merely to shrink its stored CSR
arrays.

The fixed probe seeds, response windows, and task labels are exactly Example
18's evaluation probes. Each candidate subnetwork resets all model state before
each trial.

## Contribution ranking

The pruning order is a deterministic diagnostic ranking, not a proof of causal
independence. For every neuron and task, Example 20 combines three normalized
signals measured on the trained evolving arm:

1. task-specific mean probe spike rate times absolute readout weight;
2. task-specific mean probe spike rate times total absolute outgoing recurrent
   weight; and
3. accumulated task-specific gradient mass on incident recurrent edges.

The three components are averaged per task. A neuron's scalar score is the
maximum task score, protecting neurons important to any one task. Ties are
broken by neuron index. Cumulative masked evaluation, rather than the ranking
score alone, decides whether a proposed removal is safe.

## Coarse-to-fine pruning frontier

The all-alive network is evaluated at the first qualifying pre-rebuild
checkpoint. If no checkpoint reaches `prune_target`, the final all-alive model
is measured, no neuron is removed, and the report states that the requested
frontier was never reached.

Otherwise, neurons are removed from lowest to highest contribution score. A
coarse sweep evaluates zero removals, fixed-size fraction steps, and the
one-neuron-retained endpoint. The first coarse checkpoint below target brackets
the conservative pruning frontier. Every individual retained count inside that
crossing interval is then evaluated. The selected point is the largest
contiguous number of removals from the baseline for which every evaluated task
stays at or above target.

Accuracy can be non-monotonic under lesions. Later recovery points are reported
but are not called safe or optimal because an intervening failure broke the
contiguous frontier. This frontier supplies the initial mask for the
fixed-point phase; it is not the final result.

## Joint neuron-and-edge fixed point

Starting from the coarse-to-fine safe neuron mask, Example 20 first disables
every recurrent edge incident to a removed neuron. Those endpoint removals are
structural consequences and require no separate causal trial. It then
alternates complete neuron and edge phases.

Each neuron phase repeatedly performs a greedy pass over every currently
retained neuron:

1. Re-evaluate the current mask and recompute task-aware contribution scores.
2. Visit retained neurons from lowest to highest current score, with neuron
   index as the deterministic tie-breaker.
3. Functionally ablate exactly one candidate from the latest accepted mask.
4. Permanently accept that removal only when every task remains at or above
   `prune_target`; otherwise restore the candidate before testing the next
   neuron.
5. After the pass, recompute scores from the resulting mask and repeat.

Candidates in one pass are therefore tested sequentially, not independently
against a stale mask and not removed as an unsafe simultaneous batch. Repeated
passes and candidate trials execute inside compiled `brainstate.transform`
control flow. After the neuron phase reaches a zero-removal pass, newly dead
neurons' incident edges are disabled automatically.

Each edge phase ranks the currently active retained-to-retained edges by the
maximum across tasks of an equally weighted combination of normalized
presynaptic-rate-times-absolute-weight and accumulated absolute-gradient mass.
It then individually disables edges from lowest to highest score, accepting a
deletion only when every task remains at target, and repeats passes with fresh
scores until a complete pass accepts zero. The ranking selects a deterministic
greedy path; causal probe accuracy, not the score, decides every deletion.

After an edge phase, the neuron phase runs again because a connectivity change
can make another neuron dispensable. Alternation stops only when one complete
neuron phase and one complete edge phase both accept zero removals. The final
terminal passes therefore test every retained coordinate against the same
network. The result is a *coordinate-wise locally minimal network for the fixed
probes*: removing any one retained neuron or any one retained recurrent edge
lowers at least one task below target.

This certificate does not exclude a safe swap, a multi-coordinate change that
crosses an intermediate failure, a different greedy path with fewer elements,
or changed behavior on unseen probes. Minimum equivalent-subnetwork search is
combinatorial; this backward-elimination result is not a global minimum proof.

## Edge accounting

The report distinguishes four quantities so stored shape is not confused with
functional connectivity:

1. stored checkpoint edges in the original CSR;
2. edges incident to at least one finally removed neuron;
3. original retained-to-retained edges among the final neurons; and
4. active edges left after causal edge pruning.

The causally removed retained-to-retained count is the difference between the
third and fourth quantities. The final edge mask must be a subset of the final
neuron-induced edge mask. A final active count of zero is permitted if the
feed-forward and readout paths alone preserve every task.

## Physical compaction

After the joint fixed point converges, Example 20 materializes a new inference
model whose recurrent dimension equals the retained-neuron count and whose CSR
contains only final active edges. Original neuron indices are mapped to compact
indices in ascending order. The compactor subsets the trained feed-forward
projection's output columns and readout's input rows, remaps recurrent
endpoints, preserves recurrent values and units, and initializes fresh runtime
state without compiling a learner or optimizer.

The compact model is verified against the final masked checkpoint on the same
fixed probe trials. Per-trial logits must agree within `rtol=1e-5` and
`atol=1e-6`, predictions must be identical, and every compact-model task must
remain at `prune_target`. A mismatch fails closed and no bundle is published.
The portable compressed NumPy bundle contains the compact configuration,
original retained-neuron indices, feed-forward parameters, remapped CSR
topology and values, and readout weights. A co-located loader reconstructs the
inference model from the bundle without needing the original 2,048-neuron
checkpoint.

Storage reporting counts trained parameter arrays plus CSR `indices` and
`indptr`; it does not count transient neuron/synapse state or compiler buffers.
Timing excludes compilation: each masked and compact runner is warmed once,
then one synchronized compiled `brainstate.transform.for_loop` executes the
requested number of complete fixed-probe evaluations. The report labels this
as a probe-batch timing for the observed device, not a universal latency claim.
If no neuron survives (possible only under a permissive target), physical
compaction is skipped with an explicit status because `_Net` requires a
positive recurrent dimension. A retained network with zero recurrent edges is
valid and must still be compactable.

## Symmetry and task alignment

Example 20 reuses Example 19's sparse exact-twin partition on the final evolved
topology. At the converged fixed point it reports:

- removed and retained neuron counts;
- per-task ownership counts, using the initial ranking for removed neurons and
  the recomputed final ranking for retained neurons;
- removed neurons belonging to non-singleton twin classes; and
- fully removed and partially pruned non-singleton twin classes.

Structural twins need not have identical learned weights, activity, or lesion
effects. The alignment is descriptive and must not label topology-only twins
as functionally interchangeable.

## Reporting and plot contract

The plain-text report includes baseline per-task accuracy, target, coarse and
refined checkpoint counts, the initial frontier, alternating-cycle counts,
neuron and edge removals accepted per greedy pass, final retained neuron and
edge counts, the joint single-ablation certificate, task-ownership alignment,
and twin-class alignment. It also includes masked-versus-compacted logit error,
compact accuracy, parameter/CSR storage bytes, compression ratios, synchronized
probe timing, and the compact bundle path.

The pruning PNG contains:

1. contribution score in initial removal order, colored by strongest task;
2. per-task fixed-probe accuracy versus retained-neuron count, with the target,
   initial frontier, and converged fixed point marked; and
3. stored, final-neuron-induced, and final active recurrent-edge counts.

Example 20 returns Example 18's result with a new top-level `neuron_pruning`
mapping. Development plots are not release artifacts.

## Required tests

Co-located tests cover deterministic neuron and edge ranking and ties,
normalized zero rows,
candidate masks and endpoints, contiguous-frontier selection, refinement after
the first failure, non-monotonic recovery reporting, a baseline below target,
sequential greedy acceptance, reranking, alternating neuron/edge phases,
automatic incident-edge removal, zero-removal terminal passes, both final
single-ablation certificates, task and twin alignment, functional masking of
recurrent and readout paths, restoration of trained recurrent weights,
parameter subsetting and endpoint remapping, bundle save/load round trips,
masked-versus-compacted logit equivalence, compiled timing, callback forwarding,
report output, plot creation, and a smoke entry-point run. Invalid target,
fraction, score shapes, masks, task dimensions, bundles, and benchmark counts
must raise clear exceptions.

Focused tests must exceed 90 percent statement coverage for Example 20 while
prioritizing the causal mask, frontier, and reporting paths.

## Recorded substantive result

The 2026-08-16 GPU run used four temporal-credit tasks, 2,048 neurons, 16,384
initial edges, 12 training rounds, a 100 percent prune target, and the
persistent JAX compilation cache. The first qualifying checkpoint was round 4,
before its topology rebuild, with 21,806 stored recurrent edges.

The fixed-ranking frontier retained 1,934 neurons. The first joint neuron phase
reproduced the neuron-only result of 323 retained neurons; their induced graph
had 1,604 active edges. The first edge phase reduced that to 242 edges. A second
neuron phase then reduced the network to 174 neurons, automatically disabling
incident edges, and the second edge phase retained 234. The third alternating
cycle accepted zero neuron removals and zero edge removals. All four fixed
probes remained at 100 percent.

Among the final 174 neurons, the original checkpoint contained 631
retained-to-retained edges. The final partition was therefore 21,175 edges
incident to removed neurons, 397 causally removed retained-to-retained edges,
and 234 active edges.

The physical compactor produced a 174-neuron, 234-edge inference model whose
fixed-probe predictions were identical to the masked checkpoint, whose maximum
absolute logit error was `1.91e-06`, and whose four task accuracies remained at
100 percent. Persistent parameter-plus-CSR storage fell from 485,748 to 28,324
bytes, a 94.2 percent reduction; the compressed NPZ bundle occupied 28,495
bytes. Ten warmed compiled full-probe repeats measured 32.764 ms for the masked
model and 35.186 ms for the compact model on this GPU (`0.93x`), so this run
shows a large storage reduction but no latency improvement at this small,
kernel-launch-dominated size. The round-4 wall time including joint pruning,
compaction, compilation, and timing was 718.0 seconds. These values are evidence
for this deterministic trained checkpoint, device, and probe set, not a general
minimum-size or speedup claim.

## Release boundary

The example is complete when the specification, Example 18 callback seam,
Example 20 implementation, co-located tests, and README catalog entries are
committed on a worktree branch; focused tests and the normal example gate pass;
and smoke plus at least one substantive pruning run have recorded results.
