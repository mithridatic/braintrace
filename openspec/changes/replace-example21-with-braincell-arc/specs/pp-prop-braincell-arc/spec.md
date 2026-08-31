## Purpose

Defines a small, direct, and time-bounded ARC model that uses BrainCell
Hodgkin-Huxley dynamics and BrainTrace PP-Prop without answer shortcuts or
partial-score qualification.

The requirements before "Iterative full-corpus evolution" define the bounded
`proof` and legacy `run` modes. The later iterative requirements supersede
their corpus size, promotion, result, plot, and module layout clauses only for
the `evolve` command. Their Muon and 128-update requirements also apply to the
legacy `run` command.

## ADDED Requirements

### Requirement: Real ARC task boundary

The example SHALL read the existing public ARC practice and evaluation data.
Each grid SHALL have a height and width from 1 through 30 and integer cell
colors from 0 through 9. Generated tasks SHALL NOT train, select, or qualify the
model. The 400 evaluation tasks SHALL NOT enter ordinary iteration.

The ordinary training screen SHALL use `d631b094`, `dc433765`, `b782dc8a`,
`d06dbe63`, `aedd82e4`, `0b148d64`, `b2862040`, and `150deff5` in that order.
The ordinary validation screen SHALL use `46f33fce`, `3428a4f5`, `d8c310e9`,
and `09629e4f` in that order. The runtime SHALL load these task files directly
and SHALL NOT build a task fingerprint or provenance index.

#### Scenario: Practice and evaluation tasks use the same grid contract

- **WHEN** the loader reads a valid task from either data role
- **THEN** it returns the same demonstration-pair and held-out-query structure
- **AND** role changes do not change grid validation or prediction shape rules

#### Scenario: Invalid ARC datum fails closed

- **WHEN** a grid is empty, nonrectangular, larger than 30 by 30, or contains a
  value outside the integer range 0 through 9
- **THEN** loading stops with one concise error that names the task and the
  value to correct

#### Scenario: Demonstration capacity fails clearly

- **WHEN** a task contains more than ten demonstration pairs
- **THEN** loading stops with one concise error that names the task, observed
  pair count, and maximum count of ten

#### Scenario: Routine work excludes evaluation answers

- **WHEN** an ordinary training, validation, structural, or timing run starts
- **THEN** it loads no evaluation-role target and scores only the fixed practice
  screen

### Requirement: Lossless temporal ARC events

The example SHALL encode one query episode in this order: task start, each
demonstration input grid, its output grid, the remaining demonstration pairs,
the query input grid, input end, one shape request, and 30 row requests. Each
grid SHALL contain a start event, 30 fixed row events, and an end event.

An input event SHALL contain only event type, grid role, demonstration number,
height, width, row, a 30-value valid-cell mask, and 30 position-specific cell
colors. Each applicable event type, grid role, demonstration slot, height,
width, row, and position-specific color field SHALL use one-hot values. An
inapplicable field SHALL be all zero. The valid-cell mask SHALL use independent
binary values. These binary values MAY enter neural computation as `float32`
without loss. The target query output, task
identifier, source identifier, and engineered spatial relation SHALL NOT enter
inference input.

The fixed capacity SHALL contain ten demonstration blocks and exactly 705 event
slots. The default event width SHALL be 441: seven event-type bits, four role
bits, ten demonstration-slot bits, 30 height bits, 30 width bits, 30 row bits,
30 valid-cell bits, and 300 position-specific color bits. A separate Boolean
advance mask SHALL contain one value per event. An absent row or unused
demonstration block SHALL have false advance and SHALL leave all biological and
PP-Prop eligibility state unchanged.

#### Scenario: Event round trip is exact

- **WHEN** a valid episode is encoded and decoded without the neural model
- **THEN** every demonstration input, demonstration output, and query input is
  reconstructed with the original shape, integer dtype, and cell values

#### Scenario: Query target cannot affect inference

- **WHEN** two evaluation copies differ only in the held-out query target
- **THEN** their inference event arrays are byte-identical

#### Scenario: Event order is deterministic

- **WHEN** the same episode is encoded twice with the same configuration
- **THEN** both event arrays are byte-identical and have the same request steps

#### Scenario: Maximum episode stays bounded

- **WHEN** an episode uses ten demonstrations and 30 by 30 grids
- **THEN** its complete input and output schedule contains exactly 705 events
- **AND** no event encodes fewer than the 30 valid colors of a full row

### Requirement: Minimal untyped BrainCell baseline

The baseline SHALL contain one recurrent layer with 2,048 BrainCell
single-compartment Hodgkin-Huxley neurons. Each neuron SHALL use sodium,
potassium, and leak channels. The baseline SHALL use independent
exponential-Euler integration at `0.1 ms` unless the required matched `0.05 ms`
stability comparison rejects it. The matched comparison SHALL use two compiled
`0.05 ms` integration substeps per event so that both arms simulate `0.1 ms`
per event.

Each cell SHALL subclass `braincell.SingleCompartment`. It SHALL use length
`10 um`, radius `5 um`, capacitance `1 uF/cm²`, threshold `0 mV`, initial
voltage `-65 mV`, `ReluGrad(alpha=0.3, width=1.0)`, and `ind_exp_euler`. Its
sodium ion SHALL have reversal potential `50 mV` and `Na_HH1952` with
`g_max=120 mS/cm²`, temperature `309.15 K`, `q10=3`, reference temperature
`309.15 K`, and shift `-45 mV`. Its potassium ion SHALL have reversal potential
`-77 mV` and `K_HH1952` with `g_max=10 mS/cm²` and the same temperature, `q10`,
reference temperature, and shift. Its `IL` channel SHALL have
`g_max=0.03 mS/cm²` and reversal potential `-54.387 mV`. Reset SHALL initialize
the sodium `p` and `q` gates and potassium `p` gate at the BrainCell steady
state for `-65 mV`, and it SHALL initialize spike state to false.

The baseline SHALL contain exactly 14,112 sparse input connections and 16,384
sparse directed recurrent connections. Each of the 441 event values SHALL have
32 input targets. Each neuron SHALL have eight recurrent targets. Recurrent
connections SHALL have no duplicate and no self-connection. Baseline neurons
SHALL have no Dale type, no E/I ratio, and no BrainCell AMPA, GABAa, or NMDA
mechanism.

The baseline SHALL have exactly 30,496 trainable signed biological-connection
weights. It SHALL create no dense 441 by 2,048 input matrix or dense 2,048 by
2,048 recurrent matrix. Input plus recurrent connection count SHALL never
exceed 1,024 times the active neuron count. Decoder weights SHALL be readout
parameters and SHALL NOT enter the biological-connection count.

#### Scenario: Baseline structure is exact

- **WHEN** the default model is constructed
- **THEN** it reports 2,048 neurons, 14,112 input connections, 16,384 recurrent
  connections, 30,496 trainable biological-connection weights, zero recurrent
  self-connections, zero Dale-typed neurons, and zero chemical synaptic
  mechanisms

#### Scenario: Biological state is finite

- **WHEN** one event and one compiled event sequence drive the baseline
- **THEN** membrane voltage, sodium gates, potassium gates, and spike values are
  finite

#### Scenario: Reset state matches the declared cell

- **WHEN** `init_state` and then `reset_state` run at `-65 mV`
- **THEN** sodium `p` is `0.0529324853`, sodium `q` is `0.5961207535`, and
  potassium `p` is `0.3176769141`, each within absolute error `1e-6`
- **AND** spike state is false

#### Scenario: Matched timestep check is explicit

- **WHEN** the `0.1 ms` and matched `0.05 ms` paths are compared
- **THEN** the proof records maximum absolute voltage difference, finite-state
  status, spike-event difference status, prediction equality, and strict-result
  equality
- **AND** `0.1 ms` remains selected only when voltage difference is at most
  `1 mV`, all state is finite, spike events are identical, and prediction and
  strict result are identical

#### Scenario: Recurrent storage stays sparse

- **WHEN** model arrays and compiled equations are inspected
- **THEN** no input value has shape 441 by 2,048 and no recurrent value has
  shape 2,048 by 2,048
- **AND** each stored sparse value count equals its declared connection count

### Requirement: One biological event step

For an advancing event, one model step SHALL read the previous spike state,
calculate sparse input and recurrent current, update the BrainCell population
for one fixed `0.1 ms` biological interval, and retain the new voltage,
channel-gate, and spike state. The default SHALL use one `0.1 ms` integration
step. The stability fallback SHALL use two compiled `0.05 ms` integration
substeps. The recurrent current SHALL use the spikes from the prior event
through all substeps. Both current paths SHALL have explicit BrainUnit
current-density units and bounded finite magnitude.

For a false advance value, the step SHALL leave every biological and PP-Prop
eligibility state value bitwise unchanged and SHALL produce zero loss and zero
gradient. It SHALL read output logits only at a shape-request or row-request
event.

#### Scenario: Padding does not simulate time

- **WHEN** a false-advance padded event is applied
- **THEN** voltage, channel gates, spikes, and PP-Prop eligibility remain
  bitwise unchanged
- **AND** loss and gradient are zero

#### Scenario: Padding cannot change a prediction

- **WHEN** a false-advance all-zero event is inserted into a valid tiny fixture
- **THEN** final biological state, eligibility state, gradient, and direct
  prediction are identical to the fixture without that event

#### Scenario: Recurrent current is delayed by one step

- **WHEN** a neuron first spikes during event `t`
- **THEN** that spike can enter recurrent current at event `t + 1` and cannot
  enter recurrent current at event `t`

#### Scenario: Wrong current unit fails clearly

- **WHEN** an input boundary supplies total current instead of current density
- **THEN** the step stops with one concise error that requires `mA/cm²`

### Requirement: Compiled BrainTrace PP-Prop learning path

Every counted training arm SHALL use BrainTrace PP-Prop. Repeated neural steps
SHALL use a BrainState transform and SHALL NOT use a bare Python `for` or
`while` loop. Random model, topology, and structural operations SHALL use
BrainState random state and SHALL NOT call JAX random directly. BPTT SHALL NOT
train, compare, or qualify the model.

Every trainable recurrent or input parameter that changes biological state
SHALL have the required BrainTrace hidden-state relation. A missing relation or
an unintended non-temporal classification for such a parameter SHALL stop the
run. A readout-only parameter MAY be non-temporal, but it SHALL receive a
finite direct gradient. Each height, width, and color head SHALL have at least
one nonzero direct-gradient value on the compatibility fixture.

#### Scenario: Temporal parameter is connected

- **WHEN** BrainTrace compiles the default training step
- **THEN** every intended temporal recurrent and input parameter is connected
  to each biological hidden state that it influences

#### Scenario: Direct readout gradient is valid

- **WHEN** one supervised terminal update is evaluated
- **THEN** every direct readout gradient value is finite
- **AND** each height, width, and color head has at least one nonzero gradient
  value

#### Scenario: Local derivative has an independent check

- **WHEN** a four-neuron one-step fixture starts at `-65 mV`, uses zero previous
  spikes, activates one input feature, and uses a custom 1 by 4 input CSR with
  indices `[0, 1, 2, 3]`, row pointer `[0, 4]`, and raw values
  `[0.1, 0, 0, 0]`
- **THEN** its PP-Prop derivative of
  `mean(tanh((voltage + 65 mV) / 20 mV))` agrees with the centered
  finite-difference derivative at perturbation `1e-3`
- **AND** absolute error is at most
  `1e-5 + 1e-2 * max(abs(pp_prop), abs(finite_difference))`
- **AND** the plus and minus arms start from separate reset-state copies
- **AND** the check does not use BPTT

#### Scenario: Spike path can receive learning signal

- **WHEN** a deterministic fixture resets a cell, sets its voltage to
  `-0.001 mV`, keeps the reset gates, uses zero previous spikes, and supplies
  input drive `20` through the production bounded-current path for `0.1 ms`
- **THEN** the cell crosses `0 mV` before gradient acceptance
- **AND** every inspected spike-path gradient is finite and at least one is
  nonzero

#### Scenario: PP-Prop changes executed behavior

- **WHEN** the temporary proof performs its bounded training updates
- **THEN** at least one intended recurrent weight changes by a finite nonzero
  value
- **AND** at least one direct integer prediction changes from its pretraining
  value

### Requirement: Fixed bounded training schedule

The ordinary training screen SHALL use batch size one, BrainTrace PP-Prop
`single-step`, trace decay `0.95`, gradient clip norm `1.0`, and separate Muon
optimizer groups with AdamW fallback for non-matrix values. The input learning
rate SHALL be `0.001`, the recurrent learning rate SHALL be `0.0003`, and the
readout learning rate SHALL be `0.003`.

The ordinary screen SHALL perform 128 ordered updates. It SHALL cycle through
the labeled query episodes of the eight fixed training tasks in declared task
and query order. It SHALL not add updates automatically after a failed gate.
Before each query episode, it SHALL reset voltage, channel gates, spikes, and
PP-Prop eligibility state. It SHALL retain trainable parameters, active Muon or
AdamW-fallback state, and optimizer step counts between training episodes.
Evaluation SHALL not update any of these retained values.

#### Scenario: Episode reset does not erase learning

- **WHEN** two consecutive training query episodes run
- **THEN** the second episode starts from reset biological and eligibility
  state
- **AND** it starts with the parameter and optimizer state produced by the first
  episode

#### Scenario: Failed gate does not expand training

- **WHEN** the eight-update proof or 128-update ordinary screen fails an
  acceptance gate
- **THEN** the runner stops without increasing its update count or changing its
  fixed task set

### Requirement: Strict-aligned request loss

The shape-request step loss SHALL equal height cross-entropy plus width
cross-entropy. Each row-request step loss SHALL equal the sum of
cross-entropy for its valid target cells divided by that row's valid-cell count.
An empty or non-request step SHALL have zero loss. The target SHALL enter only
the step-loss function and SHALL never enter a model event. This internal loss
normalization SHALL NOT create an averaged ARC result or score.

The temporary proof SHALL test loss alignment directly. A correct-logit case
SHALL have lower loss than matched cases with one wrong height, one wrong
width, or one wrong cell. The maximum 30 by 30 target SHALL include all 900 cell
terms without truncation.

#### Scenario: Every strict datum affects loss

- **WHEN** one target height, width, or valid cell changes between two classes
  that have different logits while all other target data and logits stay fixed
- **THEN** the calculated terminal loss changes

#### Scenario: Loss appears only on request steps

- **WHEN** one complete episode loss is inspected
- **THEN** only the shape request and valid row requests have nonzero loss
- **AND** every input, padding, and invalid row-request step has zero loss

#### Scenario: Maximum target has complete supervision

- **WHEN** a 30 by 30 target is used
- **THEN** the loss contains two shape terms and exactly 900 included color
  terms across 30 row losses

#### Scenario: Real-data loss evidence stays direct

- **WHEN** the temporary proof compares each real proof query before and after
  training
- **THEN** it records the shape loss and each valid row loss separately with
  the actual prediction, target, query exact value, and task strict value
- **AND** it does not replace these data with an average or aggregate loss

#### Scenario: Lower loss is not called strict success

- **WHEN** loss decreases but one predicted dimension or cell remains wrong
- **THEN** the query exact value is false and the report makes no ARC-success
  claim from the loss change

### Requirement: Direct integer prediction

After the final input event, the model SHALL select one height and one width
from separate 30-value score vectors. It SHALL use one argmax operation for
each dimension and SHALL NOT average repeated dimension votes. At each of 30
row-request events, it SHALL select one color from 10 scores for each of 30
columns. It SHALL retain only cells inside the selected height and width.

The returned prediction grid SHALL use `uint8`. The direct prediction
path SHALL consume executed biological state and trained model parameters. It
SHALL NOT use the query grid as an output residual or use a rule, template,
retrieval result, forest, candidate generator, repair step, or reranker.

#### Scenario: Prediction has a direct shape and dtype

- **WHEN** one query is decoded
- **THEN** the result is one rectangular integer grid with height and width
  from 1 through 30 and colors from 0 through 9

#### Scenario: Target-free prediction is stable

- **WHEN** only a held-out target changes
- **THEN** the prediction bytes do not change

#### Scenario: State interventions test causal use

- **WHEN** the proof applies voltage-only, sodium-gates-only,
  potassium-gate-only, spikes-only, all-state, and null interventions at the
  decoder boundary while input, checkpoint, and decoder remain fixed
- **THEN** it records direct prediction, request loss, and strict result before
  and after each intervention
- **AND** the all-state reset changes at least one proof-query prediction byte
- **AND** at least one individual state intervention changes a prediction byte
  or request loss
- **AND** a null intervention changes none of those values
- **AND** the report does not claim that one state component caused strict
  success when its strict result is unchanged

### Requirement: Zero-tolerance strict task scoring

The routine ARC scorer SHALL calculate only query exactness and strict task
pass-at-1. A query SHALL be exact only when the prediction has a non-Boolean
integer dtype and its height, width, and every integer cell equal the target. A
task SHALL pass only when every query in that task is exact. The tolerance SHALL
be zero. Integer storage width SHALL NOT affect exactness.

The routine scorer SHALL NOT calculate pixel, cell, color, shape, partial,
pass-at-2, cumulative, mean, or average scores.

#### Scenario: One wrong datum fails the query

- **WHEN** one predicted dimension or cell differs from the target, or the
  prediction has a Boolean or non-integer dtype
- **THEN** query exactness is false

#### Scenario: One wrong query fails the task

- **WHEN** any query in a multi-query task is not exact
- **THEN** strict task pass-at-1 is false

### Requirement: Small direct result

An ordinary `result.json` SHALL be at most 256 KiB. It SHALL contain the actual
integer prediction, actual integer target, query index, query exact boolean,
task identifier, task strict boolean, and strict passed-task count. Prediction
and target cells SHALL serialize as JSON integers. The root object SHALL contain
only `tasks` and `strict_task_pass_at_1_count`. A task object SHALL contain only
`task_id`, `queries`, and `strict_task_pass_at_1`. A query object SHALL contain
only `query_index`, `prediction`, `target`, and `exact`.

The result SHALL NOT contain logits, probabilities, hidden-state traces,
channel traces, eligibility traces, gradients, loss histories, plotting arrays,
package inventories, schema hashes, or repeated configuration data.

#### Scenario: Result exposes direct data

- **WHEN** an ordinary validation run completes
- **THEN** a reader can reproduce every query exact boolean and task strict
  boolean from the stored prediction and target grids

#### Scenario: Oversized result fails clearly

- **WHEN** encoded routine output would exceed 256 KiB
- **THEN** writing stops with one concise error that names the oversized field
  or record

### Requirement: Compact immutable checkpoint

A checkpoint SHALL be a compressed NumPy archive with format value `1`, encoded
size at or below 32 MiB, and arrays only. Its loader SHALL use
`allow_pickle=False`. It SHALL store original neuron identifiers, Dale codes,
task-owner codes, active-mechanism codes, neuron count, integration substep
count, input and recurrent CSR endpoints and raw `float32` values, readout
weight and bias, optimizer first and second arrays for all trainable values,
and three optimizer step counts. The readout weight and bias SHALL share the readout
optimizer and step count.

Dale codes SHALL use `-1` for inhibitory, `0` for untyped, and `1` for
excitatory. Owner codes SHALL use `-1` for unowned, `-2` for shared, and a
nonnegative fixed-task index for one owner. Mechanism code `0` SHALL identify
the baseline direct-current path.

The loader SHALL validate dtype, shape, endpoint, count, Dale sign, and the
biological-connection limit. It SHALL reset voltage, gates, spikes, and
eligibility state. The checkpoint SHALL NOT contain a target, prediction, loss,
gradient, eligibility trace, or other runtime state. A writer SHALL use a
temporary sibling and atomic replace. A child stage SHALL write to a path that
differs from its parent and SHALL NOT overwrite the parent. This binary artifact
SHALL remain separate from `result.json`.

#### Scenario: Checkpoint round trip preserves learned state

- **WHEN** a valid checkpoint is saved and loaded
- **THEN** every stored topology, parameter, optimizer moment, step count,
  label, mechanism code, and configuration count is identical
- **AND** biological and eligibility state starts at its declared reset value

#### Scenario: Invalid checkpoint fails closed

- **WHEN** a checkpoint has an object array, wrong dtype, invalid endpoint,
  inconsistent count, illegal Dale sign, connection-limit violation, or size
  above 32 MiB
- **THEN** loading stops with one concise error that names the value to correct

#### Scenario: Failed child preserves its parent

- **WHEN** a structural or Dale child fails, times out, or cannot complete an
  atomic checkpoint write
- **THEN** the parent checkpoint bytes remain unchanged

### Requirement: Measured backend and runtime limits

The system SHALL run the same compiled preflight in separate CPU and GPU
processes when both backends are available. The workload SHALL use the first
query of training task `d631b094`, the full 705-event baseline episode, the
production PP-Prop forward and `etrace_grad` paths, every applicable request
loss, and no optimizer update or file write. It SHALL return prediction bytes
and a finite-gradient Boolean. Each child SHALL run one warm call and three
synchronized timed calls. It SHALL use the valid backend with the literal lower
median time. An exact tie SHALL select CPU. It SHALL record both medians and the
selected backend in one concise console line, outside `result.json`. It SHALL
freeze the selected backend for all comparable arms and SHALL NOT select it
again between those arms.

The temporary proof SHALL complete within 180 seconds. Each ordinary
experiment SHALL complete within 300 seconds. A warmed batch-one decoder call
SHALL consume the 31 readout features saved at the executed request states. It
SHALL include the direct readout, argmax operations, row stacking, and grid
slicing. Each fixed-validation query decoder call SHALL complete within 100
milliseconds. The runner SHALL record five synchronized calls for each fixed
validation query. Every recorded call SHALL pass. It SHALL record
the neural request steps and full 705-event context-to-grid time separately.
The complete focused Example 21 pytest selection SHALL complete within 60
seconds on the reference development environment.

#### Scenario: Lower valid backend time wins

- **WHEN** both backend probes are valid and have different median times
- **THEN** the backend with the lower median is selected

#### Scenario: CPU is selected when it is faster

- **WHEN** the valid CPU workload is faster than the matched valid GPU workload
- **THEN** the run selects CPU and records both direct times

#### Scenario: Runtime limit stops work

- **WHEN** a proof, experiment, decoder call, or pytest selection reaches its
  declared limit
- **THEN** that operation fails its speed gate and does not report acceptance

### Requirement: Temporary real-data proof

Before structural or biological stages run, one bounded proof SHALL use the
training task `d631b094` and validation task `46f33fce`. It SHALL perform eight
ordered batch-one updates on `d631b094` only and SHALL NOT increase the update
count after a failed gate. Task `46f33fce` SHALL be forward-only. Its target MAY
enter direct loss inspection and strict scoring, but its gradient, eligibility,
optimizer update, and structural evidence SHALL NOT affect the model. The proof
SHALL show
exact event round trips, distinct finite biological states for distinct real
inputs, complete temporal parameter relations, finite direct readout gradients,
finite PP-Prop weight movement, a changed direct prediction, loss alignment,
state-intervention results, and the small strict result. For each proof query,
it SHALL show the pretraining and post-training shape loss and each valid row
loss beside the actual prediction, target, query exact value, and task strict
value. It SHALL NOT replace these direct data with a loss average.

Passing this proof SHALL mean only that the mechanism executes and affects
direct behavior. It SHALL NOT mean that the model has useful ARC ability when
the strict task count is zero.

#### Scenario: Unchanged prediction fails the proof

- **WHEN** every direct prediction remains unchanged after training
- **THEN** the temporary proof fails even if loss or parameters changed

#### Scenario: Shortcut fails the proof

- **WHEN** a held-out target or prohibited answer path enters inference
- **THEN** the temporary proof fails before any result is accepted

#### Scenario: Validation cannot train the proof model

- **WHEN** the proof evaluates `46f33fce`
- **THEN** parameters, optimizer state, and PP-Prop eligibility are identical before
  and after that validation evaluation

### Requirement: Observed structural stages

Structural change SHALL start only after the temporary proof passes. A pruning
candidate SHALL remove the lowest measured 5% of active neurons or active
recurrent connections. An addition candidate SHALL add 5% of the current active
count, rounded up to one, for a measured deficit. Pruning SHALL remain disabled
until at least one fixed validation task passes strictly.

PP-Prop gradient mass SHALL mean the per-task sum of absolute pre-clip
parameter gradients returned by `etrace_grad` across supervised request events.
It SHALL NOT mean a raw eligibility value. Neuron contribution SHALL adapt
Example 20's task-specific calculation. For each task it SHALL combine
normalized direct voltage-readout effect, normalized mean-spike recurrent relay
effect, and normalized incident PP-Prop gradient mass. Connection contribution
SHALL combine normalized source-neuron mean-spike transmission and normalized
PP-Prop gradient mass. Each final score SHALL be the maximum task score. Equal
scores SHALL use stable original index order. CSR row SHALL mean source neuron
and CSR column SHALL mean target neuron.

A pruning stage SHALL remain only when no direct per-task strict result
regresses. An addition stage SHALL remain only when direct strict behavior
improves on at least one fixed-screen task and no fixed-screen task regresses.
Random regrowth SHALL NOT be the primary addition method. Input plus recurrent
connection count SHALL never exceed 1,024 times the current neuron count.

A neuron-removal stage SHALL also remove every incident input and recurrent
connection. A neuron-addition stage SHALL add its required input and recurrent
wiring and readout values as one declared structural operation. It SHALL not
leave a new neuron disconnected. A connection-addition stage SHALL add only
connections selected for the measured deficit. Connection candidates SHALL be
evaluated in stable 256 by 256 source-target tiles, with at most 65,536 pairs in
memory. The selector SHALL exclude existing and self-connections, keep the
global best required count, and stop only when the next tile cannot change that
set. It SHALL NOT create a dense active-neuron by active-neuron array.

#### Scenario: Five-percent stage is exact

- **WHEN** a structural stage starts with `n` active items
- **THEN** it changes `ceil(0.05 * n)` items selected by measured evidence

#### Scenario: Zero strict score blocks pruning

- **WHEN** every fixed validation task has strict value false
- **THEN** neuron and connection pruning do not start

#### Scenario: Regression rejects a stage

- **WHEN** an otherwise matched candidate changes a previously true task strict
  result to false
- **THEN** the candidate is rejected and the parent checkpoint remains active

#### Scenario: Accepted topology is physically compact

- **WHEN** a structural stage is accepted
- **THEN** its checkpoint stores only retained neurons and recurrent
  connections and reports direct neuron, input-connection, and
  recurrent-connection counts
- **AND** the compact model produces prediction bytes identical to the accepted
  masked model on the fixed screen

### Requirement: Observed Dale-type stages

The baseline SHALL assign no Dale type. After the baseline proof passes, one
candidate stage MAY assign an excitatory type to
`ceil(0.05 * remaining_untyped_neurons)` neurons. A separate matched candidate
MAY assign an inhibitory type to the same count from the same parent. Neuron
selection SHALL use observed
outgoing-weight signs, activity, PP-Prop gradient mass, task ownership, and
causal effects. It SHALL NOT use a random type ratio.

Every outgoing connection from an accepted typed neuron SHALL keep the assigned
sign after training, pruning, addition, and compaction. Untyped neurons MAY
remain. The implementation SHALL call `braintrace.sparse_matmul` with a local
differentiable Dale `weight_fn`; it SHALL NOT assume that `SparseLinear`
accepts this argument. Excitatory effective weights SHALL use
`softplus(raw_weight)`, inhibitory effective weights SHALL use its negative,
and untyped weights SHALL remain raw and signed. Type assignment SHALL use
`inverse_softplus(max(abs(old_effective_weight), 1e-6))`. A new typed-source
connection SHALL start with effective magnitude `1e-6`; a new untyped-source
connection SHALL start with raw value zero. New optimizer moments SHALL start
at zero in both cases.

AMPA or GABAa SHALL be a later separate stage and SHALL NOT be combined
with the first type assignment. A later chemical-synapse stage SHALL represent
conductance as a nonnegative magnitude. The reversal potential, not a negative
conductance, SHALL make a GABAa connection inhibitory. A Dale candidate SHALL
be accepted only when at
least one fixed-screen task changes from false to true and no fixed-screen task
changes from true to false.

#### Scenario: Baseline has no implied Dale law

- **WHEN** the baseline result is written
- **THEN** it reports all 2,048 neurons as untyped and makes no E/I or Dale-law
  claim

#### Scenario: Type candidates are separate

- **WHEN** the first excitatory and inhibitory candidates are compared
- **THEN** each begins from the same accepted untyped checkpoint and changes
  exactly `ceil(0.05 * remaining_untyped_neurons)` neurons to one type only

#### Scenario: Accepted signs remain valid

- **WHEN** one PP-Prop update and one structural operation complete after a
  Dale stage is accepted
- **THEN** every outgoing weight from each typed neuron has its required sign

### Requirement: Deferred biological detail is evidence gated

AMPA, GABAa, HCN, calcium-dependent adaptation, NMDA, electrical junctions,
multiple compartments, morphology, neuromodulation, and persistent episodic
memory SHALL NOT be part of the first baseline. A later experiment SHALL change
only one such feature from an accepted parent checkpoint and SHALL use the same
fixed real-ARC screen and runtime limits. It SHALL be promoted only when at
least one fixed-screen task changes from false to true and none changes from
true to false.

#### Scenario: Optional detail does not silently enter baseline

- **WHEN** the default baseline is inspected
- **THEN** none of the deferred mechanisms is active

#### Scenario: Slow detail is rejected

- **WHEN** an optional feature causes the ordinary experiment to exceed 300
  seconds without an accepted earlier result
- **THEN** the feature is not promoted

### Requirement: Optional structural visualization

An explicit plot command SHALL produce one compact two-dimensional Matplotlib
view of an accepted checkpoint. It SHALL show the executed sparse topology and
label neuron count, input-connection count, recurrent-connection count,
untyped count, accepted Dale-type counts, and task-ownership groups. Plotting
SHALL NOT change topology, model state, prediction bytes, or routine result
content. It SHALL NOT run during ordinary training unless explicitly requested.

#### Scenario: Plot reflects executed checkpoint

- **WHEN** a plot is requested for an accepted compact checkpoint
- **THEN** the plotted node and connection counts equal the checkpoint's direct
  counts
- **AND** prediction bytes before and after plotting are identical

### Requirement: Minimal implementation and clear documentation

The executable Example 21 implementation SHALL consist of
`examples/pp_prop/21-braincell-arc.py` and its co-located
`examples/pp_prop/21-braincell-arc_test.py`. ARC-specific code SHALL remain in
that module. A shared helper SHALL require a demonstrated non-ARC use.

The implementation SHALL maintain a causal explanation and a system-model
document under `docs/specs`. The system-model document SHALL describe executed
code only. The causal explanation SHALL separate observations from inferences
and SHALL show direct prediction and target data for each claim.

All new user-facing text, errors, help, docstrings, and result labels SHALL use
ASD-STE100 Simplified Technical English. Public callables SHALL use NumPy-style
docstrings. Errors SHALL use sentence case, state what failed, and name the
corrective value or action.

#### Scenario: Focused tests are co-located and meaningful

- **WHEN** implementation validation runs
- **THEN** the co-located test module provides more than 90% meaningful
  coverage of the new example and completes inside the pytest budget

#### Scenario: Documentation does not overclaim

- **WHEN** strict task pass-at-1 is zero
- **THEN** neither document calls state change, loss decrease, or parameter
  movement successful ARC reasoning

### Requirement: Obsolete Example 21 path is retired

The replacement SHALL remove the old Example 21 entry points
`21-arc-agi-latent-reasoning.py` and
`21-latent-reasoning-in-context.py`, their sibling test, the
`latent_workspace*` production and test modules, the ARC index builder and its
test, the `docs/diagnostics/example21_*.py` scripts, obsolete Docker build
arguments, labels, index or manifest environment values, and the active old
README command. The Example 21 image SHALL retain raw ARC files and SHALL pin
`braincell==0.1.0`.

Ordinary commands SHALL load only the 12 fixed practice-task files directly.
An explicitly approved evaluation command MAY load named evaluation files. It
SHALL be forward-only and SHALL NOT use evaluation targets for training,
selection, structure, or Dale evidence. Historical Git and OpenSpec records MAY
remain. The public `braintrace` package API SHALL not change.

#### Scenario: Only one executable Example 21 remains

- **WHEN** repository files and documented commands are inspected after the
  migration
- **THEN** the BrainCell module is the only active Example 21 implementation
- **AND** no active command imports a removed latent-workspace module

#### Scenario: Approved evaluation stays held out

- **WHEN** an explicitly approved evaluation command loads a named evaluation
  task
- **THEN** it produces direct predictions and strict results without any
  parameter, optimizer, topology, owner, or Dale-type change

### Requirement: Iterative full-corpus evolution boundary

The `evolve` command SHALL load a digest-bound manifest of exactly 400 sorted
ARC training tasks and every query. Every query SHALL have a target. A missing
target, changed source digest, changed task order, or changed query order SHALL
stop the run. The 400 ARC evaluation tasks SHALL remain unopened during
training, ranking, promotion, stopping, retry, and resume reconciliation. The
terminal accepted checkpoint SHALL receive exactly one forward-only evaluation
after success, stability, or round-budget exhaustion.

#### Scenario: Full training corpus drives evolution

- **WHEN** a new evolution run starts
- **THEN** its manifest contains all 400 sorted training task identifiers,
  source digests, and target-bearing queries
- **AND** no evaluation identifier, byte, label, prediction, or aggregate
  enters an optimization decision

#### Scenario: Terminal evaluation cannot become feedback

- **WHEN** a run reaches a terminal condition
- **THEN** it scores all 400 evaluation tasks once from the unchanged accepted
  checkpoint
- **AND** it closes the lineage before another command can use that result for
  tuning

### Requirement: Muon and canonical protected objective

Every non-proof Example 21 block in `run` or `evolve` SHALL perform exactly 128
PP-Prop episode updates. Rank-two parameters SHALL use Optax Muon with
decoupled weight decay 0.1. Non-matrix parameters SHALL use Muon's AdamW
fallback. Proof mode SHALL remain exactly eight updates.

The training scorer SHALL report direct exact pass-at-1 for every task and a
target-aware loss. One query loss SHALL equal height cross-entropy plus width
cross-entropy plus the mean cross-entropy over all valid output cells. The
complete episode SHALL therefore divide row contributions by the total valid
grid-cell count, not by each row width.

At one stage, a candidate SHALL be rejected if it is non-finite, exceeds a
limit, or makes a previously exact task inexact. More exact tasks SHALL win.
At equal exact count, a candidate MAY continue only when its unresolved-task
loss decreases by at least `max(1e-6, 1e-4 * parent_loss)`. Equal capability
SHALL prefer fewer persistent bytes, then fewer neurons and recurrent edges.

#### Scenario: Smooth progress can continue

- **WHEN** a sibling preserves all solved tasks and exact count but reduces the
  unresolved-task loss by the protected threshold
- **THEN** it can become the next stage parent without a new exact task

#### Scenario: Training mastery changes the objective

- **WHEN** all 400 training tasks are exact
- **THEN** only pruning that preserves 400 exact tasks and reduces persistent
  resources can continue

### Requirement: Iterative structural lifecycle

One round SHALL train the accepted parent, compare edge-add and edge-prune
siblings, compare neuron-twin and neuron-prune siblings, revisit edges when the
neuron topology changes, and compare measured excitatory and inhibitory Dale
assignments. Each sibling pair SHALL start from one immutable parent and use
the same ordered 128-query schedule. Every selected topology, parameter set,
and active optimizer state SHALL become the downstream parent.

Structural ranking MAY use neural activity, effective sparse transmission,
task ownership, and active Muon moment magnitude. Reports SHALL call these
optimization evidence. They SHALL NOT call optimizer moments pre-clip
gradients or call the Dale proxy a causal lesion measurement.

Repeated model steps and full-corpus forward scoring SHALL use BrainState
compiled transforms. Python MAY coordinate the bounded stages whose sparse
shapes differ. It SHALL NOT dispatch repeated model steps with a bare `for` or
`while` loop.

#### Scenario: Accepted state crosses every stage boundary

- **WHEN** an edge, neuron, edge-revisit, or Dale sibling wins
- **THEN** its exact topology, parameters, active Muon state, stable neuron
  identifiers, task-owner codes, and Dale labels are restored as the next
  stage parent

#### Scenario: Bounded run stabilizes

- **WHEN** two consecutive rounds improve neither protected ARC progress nor
  mastery-preserving resources, or eight rounds complete
- **THEN** the run performs terminal evaluation and closes with the literal
  reason

### Requirement: Durable iterative artifacts and modular implementation

Format-1 NPZ files SHALL retain the compatible model, sparse topology, label,
and optimizer array schema. Digest-bound `run-state.json`,
`pending-transition.json`, and append-only `progress.jsonl` SHALL own the
cursor, immediate ancestry, stage identity, and reconciliation evidence. A
resume SHALL verify the NPZ bytes and all sidecar provenance before executing
another update. Accepted lineage evidence SHALL bind the immediate parent and
the selected source checkpoint's resolved path and SHA-256. Resume SHALL match
that evidence to durable progress before cleaning any staged source. An
interrupted accepted transition SHALL complete without retraining its mutation.
Terminal evaluation SHALL first persist an immutable
intent. Once result bytes are durable, resume SHALL never score them again. A
process death between scoring and result durability MAY replay the same
deterministic evaluation under the same intent, but SHALL emit only one result
and SHALL NOT expose partial evaluation evidence to training, selection,
stopping, or retry decisions.

The output directory SHALL contain versioned accepted checkpoints with
immediate-ancestry sidecars, `run-state.json`, `progress.jsonl`, a durable
terminal-evaluation intent while scoring is in flight, one terminal result
after close, `topology.png`, and `score-history.png`.
Progress SHALL separate scheduled cursor advance from per-arm and total
executed update counts. The PNGs SHALL refresh automatically after every
selected stage and at completion. The topology image SHALL describe the
executed graph, stable neuron lineage, task owners, Dale groups, and recurrent
edges. It SHALL NOT be described as anatomical or spatial brain imagery.

Example 21 MAY place corpus contracts, structural logic, the real adapter, and
the coordinator in focused modules with co-located suffix-style tests.
`examples/pp_prop/21-braincell-arc.py` SHALL remain the only executable Example
21 entry point.

#### Scenario: Crash recovery preserves one transition

- **WHEN** a process stops after journaling, checkpoint persistence, progress
  append, or state replacement
- **THEN** resume reconciles the same stage identity, retains byte-identical
  accepted state, and removes only verified rejected temporary candidates

#### Scenario: Closed recovery cannot start a new lineage

- **WHEN** the command reopens a closed compatible output directory
- **THEN** it verifies the accepted checkpoint and terminal result, refreshes
  any missing terminal plots or report data, and returns the same closed state
- **AND** it performs no model update, structural mutation, or held-out rescore

#### Scenario: Progress is visible during the run

- **WHEN** one selected stage completes
- **THEN** its direct score, unresolved loss, topology size, persistent bytes,
  dispositions, resource evidence, checkpoint ancestry, and elapsed time are
  durable in progress history
- **AND** the topology and score-history PNGs reflect the latest accepted state
