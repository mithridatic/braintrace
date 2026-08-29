# Example 21 bounded structural adaptation

## Optimizer clarification (2026-08-26)

The board selected Optax Muon for the model-integrated structural arms. The
active environment provides it as `optax.contrib.muon` through the declared
`optax>=0.2.8` dependency; the unrelated PyPI package named `muon` is not used.
Rank-two parameters use Muon and non-matrix parameters use Muon's built-in
AdamW fallback. The existing input, recurrent, and readout learning rates are
preserved, and both paths use decoupled weight decay 0.1. Topology rebuilds
remap the optimizer leaves that correspond to surviving items and initialize
new item state to zero. References below to Adam remapping mean this optimizer
state remapping contract and are superseded by Muon for execution.

## Baseline and success metrics

The baseline is `BrainCellArcModel`: 2,048 Hodgkin-Huxley neurons, 441 sparse
inputs, eight recurrent edges per neuron, and a 360-value voltage readout. The
fixed task order is the eight `TRAINING_TASK_IDS` followed by the four
`VALIDATION_TASK_IDS` declared by Example 21. Seed 21 initializes input edges,
seed 22 recurrent edges, and seed 23 the readout. Structural selection itself
is deterministic and consumes no random numbers.

Learning quality is the ordered vector of direct `strict_task_pass_at_1`
Booleans for all fixed tasks. Adaptation speed is the number of PP-Prop updates
to the first strict false-to-true transition. Addition arms must use exactly 64
updates even if the transition occurs earlier. Engineering metrics are complete
arm wall time (limit 300 seconds), peak process resident memory, mutated item
count, and maximum resident connection-candidate tile size (limit 65,536
pairs). The evidence artifact records Python, JAX, backend, device, platform,
seeds, starting commit, command, and all metrics. A claim is valid only when its
artifact path and SHA-256 digest are reported.

The reproducible baseline command is:

```text
python examples/pp_prop/example21_structural.py baseline --data-root <repo-root> --output <artifact.json>
```

The arm command adds exactly one of `neuron-prune`, `connection-prune`,
`neuron-add`, or `connection-add`. One invocation rejects multiple arms.

## Scope

This specification implements OpenSpec tasks 7.1 through 7.6 as pure sparse
topology operations. A topology stores CSR row pointers, target indices, and
one-dimensional edge values. No operation materializes a neuron-by-neuron pair
array.

## Evidence and ranking

Task rows are normalized independently by their nonzero maximum. Neuron scores
average normalized direct readout effect, source-row recurrent transmission,
and incident pre-clip PP-Prop gradient mass. Connection scores average
normalized source-spike transmission and pre-clip PP-Prop gradient mass. Final
scores are task maxima. Stable index order breaks score ties. Owners contain all
tasks tied at a neuron's positive maximum; zero-score neurons are unowned.

Structural twins have equal input-source, recurrent incoming-source, recurrent
outgoing-target, Dale-label, and active-mechanism sets. Values do not affect the
twin class.

## Bounded mutation

Each pruning or addition changes `ceil(0.05 * active_count)` items. Pruning is
blocked while every validation strict Boolean is false. Neuron pruning removes
all incident sparse edges. Connection pruning changes recurrent edges only.

Compaction remaps every surviving sparse endpoint, neuron row, Dale label, and
Adam row or edge value. It preserves surviving moments and step counts, gives
new items zero moments, and reports that eligibility must reset. A caller must
compare prediction bytes and strict Booleans before promotion.
Example 21's grouped Adam implementation and the canonical task identifiers in
`arc_contracts.py` are authoritative for structural measurement; structural
runs use the same eight training and four validation tasks.

Muon remapping for either addition arm is keyed by sparse identity, not by a
prefix position. The map is a target-order permutation of `(source, target)`
edge keys; surviving keys point to their source state row and new keys use a
zero sentinel. This is required because canonical CSR sorting can insert a
new edge before an existing edge. The optimizer proof checks the mapped state
against these source and target keys and fails when values are preserved at a
different sparse pair.

Neuron addition selects distinct high-score donors for the first failing
training task. Selected donors may not connect to one another. Each twin copies
input and recurrent incoming edges, duplicates outgoing edges while splitting
their values with the donor, and splits the donor readout row. It inherits owner,
Dale label, and mechanism. The complete sparse edge count must remain at or
below 1,024 times the new neuron count.

Connection addition ranks sources by mean spikes and targets by incident
gradient mass plus wrong-output readout evidence. It evaluates stable 256 by
256 source-target tiles, holds at most 65,536 resident candidate pairs, excludes
self and existing pairs, and retains the global best requested set. Untyped
edges start at raw zero; typed edges start at the inverse-softplus value whose
effective magnitude is `1e-6`.

## Promotion and execution

One process evaluates one arm. Addition arms run exactly 64 PP-Prop updates.
Every arm must finish within 300 seconds. Promotion requires at least one direct
strict false-to-true change and no true-to-false change. Pruning also requires
no regression. A failed candidate leaves the parent unchanged.

The model integration boundary snapshots pre-clip gradients directly from
`learner.etrace_grad` before `clip_gradient` is called. It accumulates absolute
mass per task and sparse parameter item. Spike evidence is accumulated from
`model.previous_spikes`; direct readout evidence is measured from the model's
voltage feature and readout rows. Physical compaction rebuilds model and learner
objects, remaps surviving Adam moments, and calls `reset_episode(learner)` so no
eligibility state crosses a topology change.
Compiled advance and padding branches must derive their output width from the
rebuilt model state, not from the 2,048-neuron baseline constant.

Mask-versus-compaction identity is evaluated on the same episode snapshot. The
masked parent zeros removed neurons and incident edges. The compact child uses
the remapped sparse topology. Promotion requires byte-identical decoded
prediction arrays and identical strict vectors before any post-compaction
training.

The measured runner requires `--data-root` (or `EXAMPLE21_DATA_ROOT`) and loads
the declared practice tasks through `load_task`. It must collect strict task
Booleans by decoding the model's 31 request readouts, and must use the same
episode snapshot for evidence and validation. Addition updates use the real
`PPPropEpisodeTrainer.update_episode` callback inside the compiled
`brainstate.transform.for_loop`; an identity callback is allowed only in
isolated unit-test doubles.

For addition arms, the compiled update loop also returns a direct strict screen
for the first failing training task after each update. The runner computes the
first update whose screen changes that task from false to true, while still
completing all 64 updates. A full fixed-task screen runs after the loop and
must show no strict regression. The transition value is measured from the
per-update screens; it must not be inferred from the final screen or replaced
with the fixed update count. The artifact records the transition probe task,
the full final fixed-task vector, and the aggregate control requires every
addition arm to be promoted with a non-null measured transition update.

## Risk register

1. **High — compilation cost:** rebuilding a learner after a shape change can
   exceed the arm budget. Measure compile and execution time separately.
2. **High — semantic drift:** incorrect sparse endpoint or Adam remapping can
   change predictions. Require byte identity and hand-calculated remap tests.
3. **Medium — invalid evidence:** post-clip gradients hide contribution mass.
   Capture gradients at the `etrace_grad` return boundary and test values larger
   than the clip norm.
4. **Medium — memory growth:** dense pair ranking would exceed memory. Tile at
   256 by 256 and report the observed maximum resident pair count.
5. **Low — unstable ties:** backend ordering can change selected items. Resolve
   every tie by source then target index and test exact selections.

## Verification

Co-located tests use hand-calculated sparse fixtures for normalization,
contribution, ownership, twins, exact ceiling counts, pruning, compaction and
Adam remapping, connected twin addition, tiled connection selection, absence of
dense pair allocation, and strict promotion rules. Integration tests must check
prediction-byte equality across accepted mask compaction and the fixed 64-update
driver.
### Gate 5 remediation contract

The real-model rebuild must carry the remapped structural Adam moments and step
into the PP-Prop trainer. Compaction identity must receive the exact pruning
alive mask and evaluate both candidates from one decoded episode snapshot. The
focused evidence command is `python -m coverage run --branch --source=. -m
pytest examples/pp_prop/example21_structural_test.py -q`, followed by
`python -m coverage report -m`.

### Gate 5 evidence clarifications (2026-08-28)

The measured runner must collect one row for every fixed task, in the declared
eight-training then four-validation order. Rows contain direct
`model.previous_spikes` means, direct voltage-readout effects, and pre-clip
`etrace_grad` mass. Validation rows are forward-only; their gradient-mass rows
are zero when no learning update is permitted. Structural scores retain their
task dimension: neuron evidence is reduced by task maximum, recurrent-edge
evidence is reduced by task maximum, and owners retain all tied task maxima.

Neuron additions select non-connected donors from the first failing training
task. Connection additions use mean spike evidence for sources and the sum of
incident pre-clip gradient mass and wrong-output readout evidence for targets.
These arrays are measured inputs to selection; synthetic fallback scores are
not valid measured evidence.

Each addition update is a real target/readout-loss PP-Prop update from the
ordered eight-task training schedule. The 64 updates are executed inside one
`brainstate.transform.for_loop`; repeating one episode or an identity callback
does not satisfy this contract.

The update loss follows the strict request schedule in `arc_contracts.py`.
Only kind-4 shape-request events contribute the sum of two categorical
cross-entropies for target height and width. Only kind-6 row-request events
contribute the mean categorical cross-entropy for every valid target cell in
the requested row. Kind-0 through kind-3 input/demonstration events, kind-5
separators, invalid rows, and padding contribute exactly zero. The loss is
computed from the live 360-value readout with JAX operations so every target
column and valid row cell can affect the PP-Prop gradient; targets must not be
encoded by selecting only column zero. Co-located regressions must prove that
a non-first-column target change changes the loss or gradient, that non-request
events remain zero, and that the compiled 64-update state/order contract is
unchanged.

#### Gate 5 row-loss correction (2026-08-29)

For a kind-6 row request, reshape the final 300 readout values to `(30, 10)`
and use the complete matrix. Each valid target column `c` is scored against
`row_logits[c]` and `target_grid[row, c]`; selecting `row_logits[row]` would
reuse one column's ten logits and is incorrect. The structural JAX loss must
match `arc_contracts.request_loss` for the same matrix, labels, and width mask.
The co-located regression uses distinct logits per column and checks a
non-first-column gradient slice.

The parent optimizer state is captured after a real warm-up update. Structural
rebuilds preserve nonzero state for surviving items and zero state for new
items, and the artifact records these checks. Complete arm time starts before
model construction and ends after strict evaluation and mask/compaction
identity. It records Python, JAX, backend, device, peak RSS, the first strict
transition update, and the PID for each isolated arm process. The final commit
must contain exactly `Co-Authored-By: Paperclip <noreply@paperclip.ing>`.

The runner always derives the evidence-ranked neuron alive mask before the
validation gate. A closed validation gate leaves the parent candidate
unchanged and reports zero mutations, but mask-versus-compaction identity uses
that measured mask rather than an index-generated substitute. Tiled connection
selection may stop only when a tile's nonnegative score upper bound is strictly
below the current retained threshold; equal-score tiles remain eligible for
stable index tie-breaking. Fixed-task forward and spike collection uses one
compiled `brainstate.transform.for_loop` over the episode snapshot.

#### Gate 5 strict-gain correction (2026-08-29)

An addition arm is eligible only when its canonical-loss updates mutate the
candidate model parameters used by the subsequent fixed-task screen. The
runner must preserve the candidate learner and trainer state created for the
structural topology through all 64 transformed updates; rebuilding or
evaluating a separate model instance invalidates the measurement. The artifact
must contain at least one direct strict false-to-true transition across the
fixed task vector, no true-to-false transition, and `promoted: true`. A
canonical-loss run with zero strict gain is a failed Gate 5 result even when
timing, sparse-memory, exact-count, and identity controls pass.

The measured candidate-only Muon rates for the real model are input `0.01`,
recurrent `0.003`, and readout `0.03`. Parent warm-up state keeps the model's
declared rates; only the 64-update addition candidate uses these rates. This
override is part of the arm artifact and must be repeated for both addition
arms.
