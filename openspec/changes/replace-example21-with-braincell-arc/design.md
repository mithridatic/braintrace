## Context

See `proposal.md` for the reason for this replacement. The binding behavior is
in `specs/pp-prop-braincell-arc/spec.md`. The approved background and measured
facts are in
`docs/specs/2026-08-24-example21-architecture-recommendations.md`.

The replacement must join four systems: raw ARC task files, BrainCell
Hodgkin-Huxley dynamics, BrainTrace PP-Prop, and exact ARC decoding. BrainTrace
PP-Prop is an approximate online learning rule. A one-step local derivative can
have an independent finite-difference check. A general temporal PP-Prop
gradient must not be described as a BPTT gradient.

BrainCell 0.1.0 has passed a basic import and Hodgkin-Huxley execution check
with the current BrainState and JAX stack. Example 20 supplies useful structural
methods. Its CSR contribution code uses the wrong source endpoint for an
`input @ weight` operation. This design keeps its task normalization, stable
ranking, ownership, twin labels, and physical compaction, but uses CSR rows as
sources and CSR columns as targets.

## Goals / Non-Goals

**Goals:**

- Make one direct model path that a person can inspect from ARC input to integer
  prediction.
- Keep static shapes, sparse biological connections, compiled temporal work,
  and bounded run times.
- Make each learning, state, structural, and Dale claim depend on observed
  direct behavior.
- Keep implementation and focused tests small enough for fast iteration.

**Non-Goals:**

- Do not preserve the old Example 21 command or result schema.
- Do not solve output generation with a rule engine, copy path, candidate set,
  retrieval system, forest, reranker, or target input.
- Do not use BPTT, synthetic tasks, partial ARC scores, or score averages.
- Do not include chemical synapses, neuromodulation, morphology, persistent
  memory, or more than one recurrent layer in the first model.
- Do not reproduce Example 20's coordinate-by-coordinate fixed-point search.
  A later compiled experiment can test that search only if the complete arm
  stays inside five minutes.

## Decisions

### 1. Keep one executable module and one sibling test module

The replacement will use `examples/pp_prop/21-braincell-arc.py` and
`examples/pp_prop/21-braincell-arc_test.py`. The executable will provide
`proof`, `run`, `structure`, `dale`, `plot`, and private backend-probe command
modes. ARC-specific parser, encoder, model, training, scoring, and result code
will stay in this module. A helper will move to a shared module only after a
second non-ARC caller exists.

This choice favors direct inspection over a new framework. The alternative was
to split the example into many components. That structure made the current
Example 21 difficult to change and test as one system.

The image, development requirements, and applicable project extras will pin
`braincell==0.1.0`. A clean development install and the image must both import
BrainCell. A later dependency update must repeat the compatibility proof before
it changes the pin.

### 2. Load only named raw ARC files

The command will accept `--arc-root`, with `/datasets/arc/raw` as the image
default. A practice task path will be
`<arc-root>/data/training/<task-id>.json`. An explicitly approved evaluation
run can use `<arc-root>/data/evaluation/<task-id>.json`. Ordinary commands will
reject the evaluation role.

The loader will validate JSON lists directly. It will reject Boolean cells,
non-integer cells, ragged rows, empty grids, dimensions outside 1 through 30,
colors outside 0 through 9, and more than ten demonstration pairs. The last
error will state that the task has too many demonstrations and that the maximum
is ten. The loader will return `uint8` grid arrays. It will not make or read an
index, fingerprint, hash, or source manifest.

The eight training tasks, four validation tasks, and their order are fixed in
the capability spec. Each task test item becomes one query episode. Training
uses the test output only as the supervised target. The output never enters an
event.

### 3. Use one fixed 705 by 441 event array

The 441 values use these half-open slices:

- event type, `[0:7]`: task start, grid start, grid row, grid end, input end,
  shape request, or row request;
- role, `[7:11]`: demonstration input, demonstration output, query input, or
  request;
- demonstration slot, `[11:21]`;
- height, `[21:51]`;
- width, `[51:81]`;
- row, `[81:111]`;
- valid-cell mask, `[111:141]`; and
- position-specific color, `[141:441]`, with index `column * 10 + color`.

Height and width class zero mean value one. Row and column positions use zero
as their first position. The schedule has one task-start event, ten 64-event
demonstration blocks, one 32-event query-input block, one input-end event, one
shape request, and 30 row requests. A grid block has one start event, 30 row
slots, and one end event.

Unused demonstration blocks and absent grid rows contain zero values and a
false advance mask. All other declared events have a true advance mask. All 30
row requests advance because output height is not available to inference. The
target controls only which request losses are valid.

The event type always has one active value. A grid event has its role, height,
and width values. A demonstration grid also has its demonstration slot. A grid
row also has its row value, valid-column bits, and one color value for each
valid column. A query grid has no demonstration slot. Task start and input end
have no other field. Shape request has the request role. Row request has the
request role and its row value. Every inapplicable field is all zero.

This representation is lossless and has a static compiled shape. A cell-event
design was rejected because it produces about 18,900 steps at maximum size.

### 4. Use deterministic sparse topology and bounded current density

The first model has 2,048 neurons. Input feature `f` has targets
`(131 * f + 61 * k) mod 2048` for `k` from 0 through 31. This gives 14,112
input connections. Recurrent source `s` has targets
`(s + offset) mod 2048`, where `offset` is one of
`1, 2, 4, 8, 16, 32, 64, 128`. This gives 16,384 recurrent connections with no
self-connection or duplicate.

In both CSR arrays, a row is the source and a column is the target. The model
will calculate `event @ input_weight` and
`previous_spikes @ recurrent_weight` with BrainTrace sparse matrix operations.

`brainstate.random.RandomState(21)` will initialize dimensionless `float32`
input weights with normal standard deviation `1 / sqrt(32)`.
`brainstate.random.RandomState(22)` will initialize dimensionless `float32`
recurrent weights with normal standard deviation `1 / sqrt(8)`. The current
density is:

`0.02 * tanh(input_drive) + 0.01 * tanh(recurrent_drive) mA/cm²`.

The bounded transform prevents one large weight from making unbounded current.
The current boundary will reject a total-current unit. Direct unbounded current
and dense input or recurrent matrices were rejected because they add stability
and memory risk without evidence.

Each neuron is one exact BrainCell 0.1.0 Hodgkin-Huxley model. It subclasses
`braincell.SingleCompartment` with length `10 um`, radius `5 um`, capacitance
`1 uF/cm²`, threshold `0 mV`, constant initial voltage `-65 mV`, BrainCell's
`ReluGrad(alpha=0.3, width=1.0)` spike surrogate, and solver
`ind_exp_euler`. It contains:

- `braincell.ion.SodiumFixed` with reversal potential `50 mV` and
  `braincell.channel.Na_HH1952` with `g_max=120 mS/cm²`, temperature
  `309.15 K`, `q10=3`, reference temperature `309.15 K`, and shift `-45 mV`;
- `braincell.ion.PotassiumFixed` with reversal potential `-77 mV` and
  `braincell.channel.K_HH1952` with `g_max=10 mS/cm²`, temperature `309.15 K`,
  `q10=3`, reference temperature `309.15 K`, and shift `-45 mV`; and
- `braincell.channel.IL` with `g_max=0.03 mS/cm²` and reversal potential
  `-54.387 mV`.

`init_state` followed by `reset_state` initializes the sodium `p` and `q` gates
and potassium `p` gate at their BrainCell steady-state values for `-65 mV`.
Spike state starts false. The baseline has no assigned Dale type and no chemical
synaptic mechanism.

### 5. Map one event to one fixed biological interval

For an advancing event, the step will read previous spikes, calculate both
sparse drives, calculate current density, update the BrainCell population, and
save voltage, gate, and spike state. The default uses one independent
exponential-Euler integration step of `0.1 ms`.

The stability check will run the same fixture with two compiled `0.05 ms`
substeps per event. This keeps the biological event interval at `0.1 ms`. The
comparison records the maximum absolute voltage difference and whether any
spike event differs. The default remains `0.1 ms` only when both paths are
finite, maximum voltage difference is at most `1 mV`, spike events are
identical, and prediction bytes and strict Booleans are identical. Otherwise,
the model will use two `0.05 ms` substeps.

A false advance mask will use `brainstate.transform.cond` and return every
biological and PP-Prop eligibility state unchanged with zero loss and gradient.
The complete sequence will use
`brainstate.transform.for_loop`. The outer model driver will use
`brainstate.transform.jit`. No repeated neural operation will use a bare Python
loop.

Each query episode resets voltage, channel gates, spikes, and eligibility
state. It retains trainable parameters and optimizer state during training.
This prevents state from one ARC task from entering another ARC task.

### 6. Use one direct 360-value readout

The readout feature is
`tanh((voltage + 65 mV) / 20 mV)`. A direct 2,048 by 360 weight and a 360-value
bias produce 30 height scores, 30 width scores, and 300 row-color scores. The
readout has no hidden decoder layer. Its parameters are not biological
connections.

`brainstate.random.RandomState(23)` initializes the `float32` readout weight
from a normal distribution with standard deviation `1 / sqrt(2048)`. The bias
starts at zero.

The shape request reads the first 60 values. Each row request reads color value
`60 + 10 * column + color`. The row-request event makes the biological state
specific to that row. Height and width use one argmax each. Each requested cell
uses one argmax over ten colors. The decoder stacks the 30 requested rows,
slices the 30 by 30 color grid to the selected shape, and returns an integer
array.

This design makes the output shape a direct model prediction. Fixed output
shape, copied input shape, and target-conditioned shape were rejected because
ARC output dimensions can differ from input dimensions.

### 7. Train with PP-Prop and one optimizer update per episode

The implementation will compile the state-changing input and recurrent
operations with `braintrace.compile(..., braintrace.pp_prop,
vjp_method="single-step")`. It will use `etrace_evolve`, `etrace_grad`, and
`reset_state` through the compiled path. The compiler report will be a proof
gate. A state-changing trainable weight cannot be missing its hidden-state
relation or be classified as non-temporal. A readout-only weight can be
non-temporal and must have a finite direct gradient.

One training update means one complete query episode. Eligibility evolves on
each advancing event. The implementation sums the shape-request and valid
row-request gradients, clips their combined norm to 1.0, and applies one Adam
update after the episode. Input, recurrent, and readout parameters use learning
rates 0.001, 0.0003, and 0.003. Trace decay is 0.95. Batch size is one. The
proof performs eight updates. An ordinary arm performs 64 updates. Both cycle
through tasks and queries in fixed order and never increase the count after a
failed gate.

All eight proof updates use only training task `d631b094`. Validation task
`46f33fce` is forward-only. Its target can enter loss inspection and strict
scoring, but no validation gradient, eligibility update, optimizer update, or
structural evidence can affect the model.

The shape-request loss is height cross-entropy plus width cross-entropy. A
valid row loss is the sum of its cell cross-entropies divided by valid cell
count. Every other event has zero loss. This row normalization controls gradient
scale. It is not an ARC score.

A four-neuron one-step fixture will use one custom 1 by 4 input CSR relation.
Its indices are `[0, 1, 2, 3]`, its row pointer is `[0, 4]`, and its raw values
are `[0.1, 0, 0, 0]`. It has no effective recurrent drive. It starts all cells
at `-65 mV`, sets previous spikes to zero, activates its one input feature, and
uses
`mean(tanh((voltage + 65 mV) / 20 mV))` as its smooth objective. The centered
perturbation is `1e-3`. Agreement means
`absolute_error <= 1e-5 + 1e-2 * max(abs(pp_prop), abs(finite_difference))`.
Each plus and minus arm starts from a separate reset-state copy. The fixture
uses the production current transform, `0.1 ms`, PP-Prop `single-step`, and
trace decay `0.95`. It requires finite, nonzero selected gradient, finite
voltage and gates, and zero spikes. It does not use a hard spike value as its
objective.

A separate deterministic spike fixture resets the same cell, then sets voltage
to `-0.001 mV`, retains the reset gates, sets previous spikes to zero, and uses
an input drive of `20` through the production bounded-current path. It first
asserts that one `0.1 ms` step crosses `0 mV`. It then requires finite
spike-path gradients and at least one nonzero spike-path gradient. The readout
fixture requires finite direct gradients and at least one nonzero gradient in
each output head. BPTT is not an oracle or an experiment arm.

### 8. Make exact data the complete ARC result

The decoder returns `uint8`. The scorer accepts any integer prediction dtype and
rejects a Boolean or floating prediction. It compares shape and integer cell
values with zero tolerance. A task passes only when every query passes.
`result.json` has the exact schema in the
capability spec and no other fields. It must encode to at most 256 KiB before
the final path replaces an old result.

The runner will write to a temporary sibling file and use an atomic replace
after size and schema checks. This prevents a timeout or write failure from
damaging the last complete result. Loss, state, gradients, package versions,
and hashes will not enter this file.

### 9. Use one compact binary checkpoint contract

The `proof` and `run` commands accept `--checkpoint-out`. The `structure`,
`dale`, and `plot` commands accept `--checkpoint`. A child checkpoint path must
differ from its parent path. A stage never overwrites its parent.

The checkpoint is a compressed NumPy archive with format value `1`. The loader
uses `allow_pickle=False`. It stores arrays only:

- original neuron identifiers, Dale codes, task-owner codes, and active
  mechanism codes;
- input CSR endpoints, `float32` values, and Adam first and second moments;
- recurrent CSR endpoints, raw `float32` values, and Adam first and second
  moments;
- readout weight, readout bias, and their Adam first and second moments;
- three Adam step counts, with readout weight and bias sharing one count; and
- neuron count and integration substep count.

Dale codes are `-1` for inhibitory, `0` for untyped, and `1` for excitatory.
Owner codes are `-1` for unowned, `-2` for shared, and a nonnegative fixed-task
index for one owner. Mechanism code `0` means the baseline direct-current path.

The loader validates every dtype, shape, endpoint, count, Dale sign, and
biological-connection limit. It resets voltage, channel gates, spikes, and
eligibility state. A checkpoint does not store targets, predictions, losses,
gradients, eligibility traces, or other runtime state. The writer uses a
temporary sibling and atomic replace. The encoded checkpoint must be at most
32 MiB. This binary artifact is separate from the small `result.json`.

### 10. Select CPU or GPU with one matched preflight

The parent command will start separate child processes for CPU and GPU when
both are available. Each child will compile the first query episode of training
task `d631b094` with the 2,048-neuron baseline, all 705 events, the production
PP-Prop forward and `etrace_grad` paths, and every applicable request loss. It
will not apply an optimizer update or write a file. It will run one warm call
and time three synchronized calls with `time.perf_counter()`. The comparison
value is the median. The timed return values are the prediction bytes and a
finite-gradient Boolean. A child is invalid when state or gradient is not
finite or prediction bytes differ from its repeat.

The valid backend with the lower median wins. An exact timing tie selects CPU.
The parent prints one line with both medians and the selected backend. It
freezes that backend for all compared arms. It does not put timing data in
`result.json`.

The 100 ms decoder gate consumes the 31 readout features saved at the executed
shape and row request states. It includes the direct readout matrix, argmax
operations, row stacking, and grid slicing. The runner first warms this path.
It then records five synchronized calls for each fixed validation query. Every
recorded call must be at or below 100 ms. The neural request steps and complete
705-event context-to-grid time are separate direct measurements and remain
inside the five-minute experiment limit. Proof, experiment, and pytest timers
include process startup and compilation where the capability spec requires
them.

### 11. Adapt Example 20 as five-percent block stages

Structural evidence will be collected on the eight training tasks. The four
validation tasks remain unseen by gradient and contribution calculations. A
candidate gate compares direct strict booleans on all 12 fixed tasks. These are
development tasks, not an independent final evaluation.

PP-Prop gradient mass means the per-task sum of the absolute, pre-clip
parameter gradients returned by `etrace_grad` across supervised request
events. It does not mean a raw eligibility value. The implementation records
this evidence only for the structural diagnostic and does not put it in the
ordinary result.

For each training task, neuron contribution has three components:

1. mean absolute readout feature times the L1 norm of that neuron's 360
   readout weights;
2. mean spike activity times the sum of absolute recurrent weights
   whose CSR row is that neuron; and
3. PP-Prop gradient mass on input connections that target the neuron and
   recurrent connections incident to the neuron.

Each component row is divided by its maximum when that maximum is nonzero. The
three normalized rows are averaged. A neuron's final score is its maximum task
score. A neuron with all-zero task scores is unowned. Equal nonzero maximum task
scores give a shared owner. Otherwise, the one maximum task is the owner. This
maximum protects a neuron that is important to one task.

For each recurrent connection, transmission is the source neuron's mean spike
activity times absolute weight. The second component is PP-Prop gradient mass
for that connection. The two task rows are normalized and averaged. The final
score is the maximum task score. Equal final scores use original index order.

One pruning arm removes `ceil(0.05 * active_count)` lowest items. Neuron pruning
also removes incident input and recurrent connections. Connection pruning
changes recurrent connections only. No optimizer update occurs between the
mask and the strict causal gate. Pruning is disabled while all four validation
task booleans are false. A passed mask is physically compacted and checked for
identical prediction bytes before it becomes the parent checkpoint.

Structural twins have the same input-source set, recurrent incoming-source
set, recurrent outgoing-target set, Dale label, and active cell mechanism set.
Weights do not define a twin. The report and plot group neurons by task owner,
twin class, and Dale label.

One neuron-addition arm adds `ceil(0.05 * active_neurons)` structural twins.
Donors are the highest task-score neurons for the first failing training task,
with stable index ties. Donors must be distinct and must not have a recurrent
connection to another selected donor. If the required donor set does not
exist, the stage fails clearly.

A twin copies its donor's input and recurrent incoming connections and their
weights. Each donor recurrent outgoing connection is duplicated for the twin,
and the donor and twin outgoing values each receive half the old value. The
donor and twin readout rows each receive half the old row. The twin inherits the
donor Dale label and task owner. New optimizer moments are zero. Existing
moments and step counts remain. The complete structure must satisfy the
biological-connection limit before compilation.

One recurrent-connection addition arm adds
`ceil(0.05 * active_recurrent_connections)` absent, non-self source-target
pairs. For the first failing training task, source evidence is normalized source
mean spike activity. Target evidence is the normalized sum of target-incident
PP-Prop gradient mass and L1 readout weight for currently wrong output data.
Pair score is source evidence times target evidence. Stable descending score,
source index, and target index select the pairs.

The selector sorts source and target evidence once and evaluates candidates in
256 by 256 tiles. It excludes self-connections and existing connections, keeps
only the global best required count, and stops when the next tile's score upper
bound cannot enter that set. It expands through later tiles in stable order if
the first tile has too few valid pairs. It holds at most 65,536 candidate pairs
at one time and never creates an `active_neurons` by `active_neurons` array.
New optimizer moments start at zero. A new connection from an untyped source
has raw weight zero. A new connection from a typed source has raw value
`inverse_softplus(1e-6)` and effective magnitude `1e-6`. This makes the first
change depend on PP-Prop, not random regrowth.

An addition arm performs the fixed 64 training updates after the structural
operation. It becomes the parent only when at least one direct strict Boolean
changes from false to true and none changes from true to false. One process
tests one block arm. The implementation does not run a combinatorial search or
an automatic unbounded cycle.

Physical compaction remaps input target indices, recurrent source and target
indices, readout rows, Dale labels, and Adam arrays. It preserves surviving
optimizer values, gives new values zero moments, and resets biological and
eligibility state. It causes one intentional compilation. Prediction bytes and
strict booleans must match the accepted masked checkpoint.

### 12. Add Dale types only as separate measured candidates

Excitatory and inhibitory candidates start from the same accepted untyped
checkpoint. Each candidate selects `ceil(0.05 * untyped_neurons)` neurons.
Excitatory evidence combines positive outgoing-weight coherence, activity,
PP-Prop gradient mass, task ownership, and the block lesion effect. Inhibitory
evidence uses negative outgoing-weight coherence with the same other
components. Each component is task-normalized, and stable index order resolves
ties.

The recurrent operation will call `braintrace.sparse_matmul` directly with a
differentiable `weight_fn`; it will not assume that `SparseLinear` accepts this
argument. Excitatory sources use `softplus(raw_weight)`. Inhibitory sources use
`-softplus(raw_weight)`. Untyped sources use the raw signed weight. Type
assignment uses
`inverse_softplus(max(abs(old_effective_weight), 1e-6))` for each selected raw
value. A new connection from a typed source starts with effective magnitude
`1e-6`. Addition and compaction preserve the selected label and effective sign.
Each candidate performs the fixed 64 updates and uses the same false-to-true,
no-regression strict gate as structural addition.

AMPA or GABAa is a later arm from an accepted typed checkpoint. It cannot enter
the first type-assignment arm. This separation shows whether the type constraint
or the chemical mechanism changed behavior. A chemical-synapse arm represents
conductance as a nonnegative magnitude. Its reversal potential makes GABAa
inhibitory; it does not use a negative conductance. The arm defines and tests
this representation change before it replaces direct signed recurrent current.

### 13. Keep optional biology and plots outside the baseline

Later arms can test one of AMPA, GABAa, HCN, calcium-dependent adaptation,
NMDA, electrical junctions, multiple compartments, morphology,
neuromodulation, or persistent episodic memory. Each arm changes one feature,
uses the fixed tasks, and must meet the same strict and time gates.

The first plot path uses Matplotlib in two dimensions because it has lower setup
cost than a three-dimensional renderer. It reads an accepted compact checkpoint
and groups nodes by task owner and Dale label. Plotting is explicit and cannot
run during normal training.

## Risks / Trade-offs

- [A 705-event Hodgkin-Huxley rollout can consume the five-minute run budget] ->
  Measure CPU and GPU first, keep all temporal work compiled, report full
  rollout time separately from decoder time, and stop before a slow arm.
- [The bounded current gain can make neurons silent] -> Require finite distinct
  states, a nonzero spike-path gradient, and direct prediction change in the
  proof. Change gain only in a separately specified arm.
- [A direct voltage readout can have many parameters] -> Keep it as one matrix,
  exclude it from biological-connection claims, and measure decoder time.
- [Eight updates cannot establish ARC skill] -> Treat the proof only as a
  mechanism check. Report strict data and make no ability claim from loss.
- [The 12-task screen is used for promotion] -> Call it development evidence
  and keep the 400 evaluation tasks outside ordinary iteration.
- [Physical compaction can change sparse reduction order] -> Require identical
  prediction bytes and strict booleans before accepting the compact checkpoint.
- [Neuron twins can exceed the connection limit] -> Calculate the complete new
  input and recurrent count before model construction and reject the arm.
- [Five-percent block pruning is less minimal than coordinate search] -> Prefer
  fast iteration now. Test a compiled coordinate method later only as a bounded
  optional experiment.
- [A Dale transform can reverse many effective outgoing signs at once] ->
  Compare excitatory and inhibitory candidates from the same untyped parent,
  preserve effective magnitudes, and require strict gain.

## Migration Plan

1. Add BrainCell 0.1.0 to the Example 21 image and pass the isolated import,
   integration, and PP-Prop compatibility fixtures.
2. Add the replacement module and sibling tests without importing an old
   `latent_workspace` module.
3. Pass the event, no-target, sparse topology, loss, scorer, result, compiler,
   timing, and temporary proof gates.
4. Write the causal explanation and executed system-model documents in
   ASD-STE100 Simplified Technical English.
5. Remove the old Example 21 entry point, `latent_workspace*` modules and tests,
   index builder and test, Docker index command, and obsolete README command.
6. Implement the bounded structural and Dale commands and their tiny fixtures.
   Do not execute or promote a real structural or Dale arm until the baseline
   proof passes.
7. Run the focused test and OpenSpec validation.

Before an accepted replacement checkpoint exists, rollback keeps the previous
Git revision. After structural or Dale work starts, each arm keeps its immutable
parent checkpoint and promotes only an accepted child. A failed or timed-out
arm leaves the parent active.
