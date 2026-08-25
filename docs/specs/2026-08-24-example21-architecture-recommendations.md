# Example 21 architecture recommendations

Date: 2026-08-24
Status: Design direction approved for documentation. Implementation has not
started and requires a separate approval.

## Goal

Build the smallest useful biologically detailed ARC system:

**BrainCell Hodgkin-Huxley neurons + sparse recurrent connections + PP-Prop + a
direct ARC decoder.**

Make this work before increasing biological detail. Keep ordinary experiments
and tests short enough for rapid iteration.

## Remove from the proposed path

1. Remove generated synthetic tasks from primary training, iteration screens,
   and ARC performance claims. A tiny synthetic fixture may remain only as a
   mechanical unit test. It cannot qualify a prediction.
2. Remove MiniLSTM memories and the relation-state path.
3. Remove the current feature called latent reasoning. Its extra hidden
   computation does not demonstrate a process that considers choices, judges
   them, and revises an answer.
4. Remove BPTT training, BPTT comparison arms, and BPTT gradient oracles. Use
   PP-Prop for learning and tiny finite-difference checks when an independent
   numeric check is necessary.
5. Remove direct copy paths and answer shortcuts: identity copy residuals,
   direct query-grid-to-output wiring, handcrafted ARC rules, templates,
   retrieval, forests, candidate generators, repair, and reranking. The neural
   model may learn that copying is correct, but the code must not copy by
   construction.
6. Remove full evaluation and large diagnostic suites from normal iteration.
   Do not run the 400-task evaluation, causal matrices, extensive perturbation
   suites, or internal-state probes after every change.
7. Remove non-strict ARC scoring from routine results. Do not report pixel,
   cell, color, height, width, shape, or other averaged partial-credit scores.
8. Remove large result archives from ordinary runs. Do not save logits,
   activations, eligibility traces, loss histories, repeated targets, or large
   diagnostic arrays in `result.json`.

## Starting model

- Use BrainCell's simplest single-compartment Hodgkin-Huxley construction.
- Begin with sodium, potassium, and leak channels.
- Use signed recurrent current coupling in the first model. Do not assign an
  excitatory or inhibitory neuron type in the baseline. Add Dale types, AMPA,
  GABAa, and NMDA only in later measured stages.
- Keep channel conductances, reversal potentials, capacitance, and numerical
  integration settings fixed in the first working version.
- Use one recurrent layer with 2,048 Hodgkin-Huxley neurons.
- Use a sparse input projection with 32 target neurons for each of the 441
  event values. This gives 14,112 trainable input connections.
- Start with exactly 16,384 sparse directed recurrent connections. Never create
  a dense 2,048 by 2,048 recurrent matrix.
- Do not permit a connection from a neuron to itself unless a later discussion
  approves it.
- Use PP-Prop for every trainable counted arm.
- Measure the same small compiled workload on CPU and GPU when both are
  available. Use the faster valid backend for tests and experiments. Do not
  assume that GPU launch and compilation overhead wins for a small workload.
- Decode only from executed Hodgkin-Huxley population state. The prediction is
  one integer height, one integer width, and one integer color for every output
  cell.

The baseline has 30,496 biological connections: 14,112 input connections and
16,384 recurrent connections. Decoder weights are readout parameters. They are
not biological synapses and do not enter this connection count.

This baseline has eight outgoing recurrent connections per neuron on average.
It matches the starting scale that supported four-task specialization in
Example 20. Keep the topology sparse. The larger starting connection count does
not relax the three-minute proof, five-minute experiment, 100-millisecond
decoder, or one-minute pytest limits. Optimize compiled sparse execution before
you increase a time limit.

The hard biological-connection limit is 1,024 times the current neuron count.
For 2,048 neurons, this limit is 2,097,152 input plus recurrent connections.
This value is a safety limit, not a growth target. The executed model must
remain sparse. A stage must stop before this limit when runtime, memory, or
strict ARC behavior stops supporting growth.

## Exact baseline cell

Use `braincell.SingleCompartment` with these fixed values:

- length `10 um`, radius `5 um`, and capacitance `1 uF/cm²`;
- threshold `0 mV` and constant initial voltage `-65 mV`;
- `ReluGrad(alpha=0.3, width=1.0)` and solver `ind_exp_euler`;
- `SodiumFixed` reversal potential `50 mV` with `Na_HH1952`, conductance
  `120 mS/cm²`, temperature `309.15 K`, `q10=3`, reference temperature
  `309.15 K`, and shift `-45 mV`;
- `PotassiumFixed` reversal potential `-77 mV` with `K_HH1952`, conductance
  `10 mS/cm²`, and the same temperature, `q10`, reference temperature, and
  shift; and
- `IL` conductance `0.03 mS/cm²` and reversal potential `-54.387 mV`.

Call `init_state` and then `reset_state`. At `-65 mV`, verify sodium `p` as
`0.0529324853`, sodium `q` as `0.5961207535`, and potassium `p` as
`0.3176769141`, each within absolute error `1e-6`. Verify that spike state is
false. These values make the starting biological model reproducible.

## Model terms

Use the terms from BrainCell, BrainTrace, and BrainState as follows:

- A **neuron**, or **cell**, is one complete biological model unit. The first
  model has 2,048 BrainCell Hodgkin-Huxley neurons.
- A **compartment** is one modeled part of a neuron. The first model has one
  compartment in each neuron. A later model can have separate soma, dendrite,
  and axon compartments.
- A **recurrent connection** is one directed relation from a source neuron to a
  target neuron. The first model has 16,384 recurrent connections.
- An **input connection** is one directed relation from an ARC event value to a
  neuron. The first model has 14,112 input connections.
- A **synaptic mechanism** is the BrainCell process that transfers a signal
  across a connection. AMPA, GABAa, and NMDA are later BrainCell chemical
  synaptic mechanisms. The first model uses direct signed recurrent current
  coupling and no BrainCell chemical synaptic mechanism.
- A **synaptic weight** is the trainable strength of a synaptic connection. The
  first model has 14,112 input weights and 16,384 recurrent weights in
  BrainState parameter state.
- A **Dale sign** is the fixed excitatory or inhibitory type of a source
  neuron. All outgoing recurrent connections from that neuron use this type.
- A **BrainTrace ETP operation** is the trainable operation that lets the
  BrainTrace compiler connect a synaptic weight to the BrainCell hidden state
  that it changes.
- A **BrainState transform** compiles and runs a model operation. Use
  `brainstate.transform.jit` for one step. Use
  `brainstate.transform.for_loop` for repeated steps.

In the first model, each input or recurrent connection has one trainable signed
weight. Thus, the model has 30,496 biological connections, no BrainCell
chemical synaptic mechanisms, and 30,496 trainable biological-connection
weights. A later typed connection can use one or more chemical mechanisms,
such as AMPA or AMPA and NMDA. At that stage, the connection count and the
synaptic-mechanism count will not necessarily be equal.

Use these configuration names:

- `neuron_count = 2_048`
- `input_connection_count = 14_112`
- `recurrent_connection_count = 16_384`
- `biological_connection_count = 30_496`
- `synaptic_mechanism_count = 0` for the first model
- `trainable_biological_connection_weight_count = 30_496` for the first model

## Lossless ARC input

Give the model only the information contained in the ARC problem:

- demonstration input grids;
- demonstration output grids;
- the query input grid;
- grid and task boundaries;
- whether a grid is a demonstration input, demonstration output, or query;
- exact height and width;
- row and column coordinates; and
- integer cell colors from 0 through 9.

Present this information as a compact temporal sequence. Do not append
precomputed rotations, shifts, reflections, scaling relations, mirror maps,
routing programs, query patches, or a flattened copy of every possible spatial
relationship. The representation must be lossless: decoding the input sequence
must reconstruct every supplied ARC grid exactly.

Use this exact event order:

1. Emit one task-start event.
2. For each demonstration, emit its input-grid start event, 30 fixed row
   events, and its grid-end event.
3. Emit the same three parts for that demonstration's output grid.
4. Repeat steps 2 and 3 for each remaining demonstration in the given order.
5. Emit the query-input start event, 30 fixed row events, and its grid-end
   event.
6. Emit one input-end event.
7. Emit one shape-request event. Decode one height and one width.
8. Emit 30 row-request events in row order. Decode 30 colors at each event.
9. Keep only the decoded cells inside the selected height and width.

Each input event contains only these fields:

- event type;
- grid role;
- demonstration number;
- height and width;
- row number;
- a 30-value valid-cell mask; and
- one position-specific color from 0 through 9 for each of 30 columns.

Use one-hot values for each field that applies to the event. Use all zeros for
an inapplicable field. An absent row has an all-zero event, an all-zero validity
mask, and a false advance value. It leaves biological and PP-Prop eligibility
state unchanged. An unused demonstration block also has false advance values.
Do not include a task
identifier, target query output, engineered relation, or copy path. During
training, the target affects the loss after decoding. The target never enters
the model input. Do not add a neutral-input reasoning interval to the first
model.

Use capacity for ten demonstrations. A complete fixed episode has at most 705
events: one task event, 640 demonstration-grid events, 32 query-grid events,
one input-end event, one shape request, and 30 row requests. The default event
width is 441 values: seven event types, four grid roles, ten demonstration-slot
values, 30 height values, 30 width values, 30 row values, 30 validity values,
and 300 position-specific color values. A separate Boolean advance mask has one
value per event. This schedule is lossless and avoids the approximately 18,900
neural steps of a maximum cell-by-cell schedule.

## Minimal BrainCell-to-PP-Prop bridge

BrainCell already supplies JAX and BrainState-compatible biological dynamics.
Do not add another recurrent framework on top of it. Custom code should only:

1. hold 14,112 input weights and 16,384 recurrent weights in BrainState
   parameter state;
2. execute both groups through BrainTrace sparse ETP operations with fixed CSR
   topologies;
3. deliver the synaptic current to the BrainCell neurons;
4. apply PP-Prop to the sparse weights and direct output head;
5. run repeated steps with `brainstate.transform.for_loop`; and
6. decode one ARC grid directly from the resulting neural state.

A BrainCell synaptic layer that bypasses BrainTrace ETP operations is not
acceptable. Running in JAX does not by itself make a parameter PP-Prop
trainable.

`brainstate.transform.for_loop` is not part of the PP-Prop mathematics.
PP-Prop supplies the learning rule. The transform lowers repeated neuron steps
into one compiled XLA loop and carries BrainState values without a Python loop.
Use `jit` for one isolated step and `for_loop` for a repeated simulation.

The compiler must connect every intended recurrent or input parameter that
changes Hodgkin-Huxley state to the hidden states that it influences. Treat a
missing-hidden-state diagnostic for one of these temporal parameters as a hard
failure. Do not silently train an intended temporal parameter as non-temporal.

A direct output parameter reads hidden state but does not change future hidden
state. BrainTrace can correctly classify this parameter as non-temporal. Require
each direct output parameter to be present in training and each direct gradient
value to be finite. Require at least one nonzero gradient value in each height,
width, and color head. Do not require a hidden-state eligibility relation for a
true readout-only parameter.

## BrainCell version and simulation time

Use `braincell==0.1.0`. It is the latest published PyPI release on 2026-08-24.
Its declared requirements permit the current Python 3.14, BrainState 0.5.3,
BrainUnit 0.5.2, and JAX 0.11.0 environment.

A disposable check installed BrainCell 0.1.0 in the reference image. It
imported with BrainState 0.5.3 and JAX 0.11.0. One independent
exponential-Euler step completed on the CPU backend at `0.1 ms`. The observed
voltage was `-64.68634 mV`, and the observed spike value was `0`. This confirms
basic package and single-cell execution compatibility. It does not yet confirm
the sparse PP-Prop bridge or GPU performance.

A human brain has continuous physical dynamics. It does not use a numerical
solver or a fixed simulation timestep. These values belong to the simulation.

Start with these simulation values:

- `solver = "ind_exp_euler"`
- `dt = 0.1 ms`
- one integration step for each ARC input or output event
- ARC input current converted explicitly to `mA/cm²` with BrainUnit

The independent exponential-Euler solver is the BrainCell single-compartment
default shown in its published example. The `0.1 ms` timestep is a speed-first
starting value for Hodgkin-Huxley dynamics. It is not a claim about a biological
clock.

Reject a current with the wrong physical dimension. The compatibility check
confirmed that a BrainCell single-compartment neuron expects current density,
not total current. Keep this unit conversion in one named input-boundary
function.

In the temporary proof, repeat the single-cell trace at `0.05 ms`. Use two
integration steps per event in this comparison so the simulated duration stays
the same. Report the largest direct voltage difference, whether every state
remains finite, and whether spike events change. For the complete ARC path,
report whether the direct prediction and strict result change. Keep `0.1 ms`
only when all state is finite, the largest voltage difference is at most
`1 mV`, spike events are identical, and the direct prediction and strict result
are identical. If any gate fails, use `0.05 ms`.

## Direct output schedule

Do not average repeated height or width votes. After the model reads the ARC
input sequence:

1. one output step selects height and width directly from two 30-value score
   vectors;
2. thirty row steps each select 30 integer cell colors; and
3. the decoder keeps only the rows and columns inside the selected shape.

Each choice uses the largest score once. The fixed row-step index tells the
shared output head which row it is emitting; it does not supply query-cell
values or a handcrafted transformation. The scorer then applies zero-tolerance
grid equality.

## Neuron groups and Dale signs

The baseline starts with one untyped neuron population. Its recurrent weights
can be positive or negative independently. It has no Dale signs and makes no
Dale-law claim.

After the untyped baseline works, use explicit groups in reports and structural
decisions:

- excitatory Hodgkin-Huxley neurons;
- inhibitory Hodgkin-Huxley neurons;
- task-ownership groups derived from measured contribution; and
- exact structural-twin groups when neurons have the same structural role.

The last two follow the analysis style used by Example 20. They describe
neurons; they are not extra layers.

Do not assign neuron types at random and do not set a target E/I ratio. Measure
the untyped model first. Rank neurons by outgoing-weight signs, activity,
PP-Prop gradient mass, task ownership, and causal removal effect. PP-Prop
gradient mass is the per-task sum of absolute pre-clip parameter gradients
returned by `etrace_grad` across supervised request events. It is not a raw
eligibility value.

One candidate stage assigns `ceil(0.05 * remaining_untyped_neurons)` neurons an
excitatory Dale type. A separate candidate stage assigns the same count an
inhibitory Dale type. If both separate stages help, a later candidate can
include both accepted groups.
Select the neuron identities from the measured evidence. Compare each candidate
with the same untyped checkpoint and fixed real-ARC screen. Keep only a stage
that improves at least one direct per-task strict result without causing a
strict regression on another fixed-screen task.

After a neuron receives a Dale type, every outgoing recurrent connection must
use that sign. A Dale type is permanent in its accepted descendant checkpoint.
Neuron and connection pruning or addition must preserve every assigned type and
legal sign. Untyped neurons can remain in the model.

Call `braintrace.sparse_matmul` directly with a local differentiable Dale
`weight_fn`. Do not assume that `SparseLinear` accepts this argument. Use
`softplus(raw_weight)` for an excitatory source and its negative for an
inhibitory source. Keep an untyped source raw and signed. On type assignment,
use `inverse_softplus(max(abs(old_effective_weight), 1e-6))`. A new connection
from a typed source starts with effective magnitude `1e-6`; a new connection
from an untyped source starts with raw weight zero. New optimizer moments start
at zero.

Test BrainCell AMPA and GABAa mechanisms after a useful Dale-typed stage exists.
First apply the mechanism only to its matching accepted group. Do not combine
the Dale assignment and chemical-mechanism change in one experiment. A later
chemical-synapse arm uses a nonnegative conductance magnitude. The reversal
potential makes GABAa inhibitory; a negative conductance does not.

Do not set a permanent connection budget for each E/I block. Begin with equal
average outgoing connection counts across declared neuron groups. Let measured
task contribution, activity, and causal removal tests direct later pruning and
addition. Every stage must preserve Dale signs. Random structure may appear
only as a small necessary control.

## Neuromodulation boundary

BrainCell does not supply a built-in neuromodulatory system. Its published
synapse library supplies AMPA, GABAa, and NMDA mechanisms. These are useful
biological synapses, but they are not a dopamine, serotonin, acetylcholine, or
norepinephrine system. The first model must not claim neuromodulation.

BrainCell supports custom mechanisms through its open registries. A later stage
can add one small JAX-compatible modulatory state. That state can change one
declared property, such as synaptic plasticity strength, neuron excitability,
or a structural-growth threshold. Do not add several modulators at once. Keep
the stage only if direct strict ARC behavior improves and the complete run
stays inside five minutes.

## Memory boundary

Current Example 21 has no autobiographical or persistent episodic memory. Its
MiniLSTM states are temporary working memory. Its older fast-weight memory
stores and reads demonstration associations only inside one ARC episode. Model
and PP-Prop states reset before another evaluation batch. Checkpoints save
trainable parameter values, not experiences, timestamps, hidden states, or
retrievable episodes.

The first BrainCell model needs working state and learned synaptic memory only.
Long-term autobiographical memory is a later possibility. Do not claim that it
emerged unless the system retains a specific experience across task boundaries
and later retrieves that experience by its time or context without receiving
the record again as input. General information absorbed into weights is not by
itself an autobiographical memory.

## Attention boundary

The old Attention Residual mechanism mixed earlier decoder states. The later
query-routing mechanism mixed 900 query positions through handcrafted geometry
and learned routing programs. These mechanisms were output routing, not shared
cognitive attention among competing neural processes. They did not produce a
strict real-task improvement and are not part of the first BrainCell model.

Let competition begin through signed recurrent dynamics and learned structural
specialization. Later accepted Dale groups can add explicit E/I competition.
Add an explicit attention mechanism only when direct behavior identifies a
specific selection problem that recurrent dynamics do not solve.

## Visual evidence

The first model uses single-compartment neurons. A single-compartment neuron
has no dendrite or axon tree for `braincell.vis.plot2d` or
`braincell.vis.plot3d` to show.

For the first model, produce one optional compact 2-D network view with
Matplotlib. Label untyped neurons, any accepted excitatory or inhibitory groups,
task-ownership groups, and retained neuron and recurrent-connection counts.
Show the sparse recurrent connections without changing the executed topology.
Do not generate this plot during every iteration. Generate it for an accepted
structural checkpoint or when direct inspection is requested.

When a later stage adds multi-compartment cells with dendrites or axons, use
BrainCell's morphology visualization:

```python
braincell.vis.plot2d(morphology)
```

Use the 2-D Matplotlib view by default because it is suitable for headless runs
and has fewer rendering dependencies. On the first accepted morphology only,
compare it directly with:

```python
braincell.vis.plot3d(morphology)
```

Record each observed render time. Keep the 3-D PyVista or Plotly view only if
it shows a branch or spatial relationship that the 2-D view hides and the
complete requested operation stays within its time limit. Do not delay model
training or strict scoring to render a visualization.

Every retained figure must label the model stage, neuron count, recurrent
connection count, untyped and Dale-typed neuron counts, and represented task
groups. A morphology figure must also label the soma, dendrites, and axon when
those structures exist.

## Structural stages

Combine Example 18's structural evolution with Example 20's causal pruning and
physical compaction:

1. Build the one-layer sparse baseline with 2,048 neurons, 14,112 input
   connections, and 16,384 recurrent connections.
2. Train with PP-Prop on real ARC practice tasks for a bounded run.
3. Measure neuron and connection contribution from mean spike activity, direct
   voltage-readout effect, pre-clip `etrace_grad` gradient mass, and task
   ownership. Adapt Example 20's task-row normalization, maximum task
   protection, stable rank order, ownership groups, twin labels, and physical
   compaction. Correct its sparse direction for `input @ weight`: a CSR row is
   a source neuron and a CSR column is a target neuron. Thus, outgoing strength
   and transmission use the row index.
4. Do not prune while every fixed validation task is false. After at least one
   validation task passes, prune the lowest-contribution 5% of active neurons.
   Keep the stage only when no direct per-task strict result regresses.
5. Prune the lowest-contribution 5% of active recurrent connections. Test one
   block stage at a time. Do not copy Example 20's large coordinate-by-coordinate
   fixed-point search. Continue only while the complete block passes its direct
   strict gate and stays inside the run-time limit.
6. Add 5% more neurons or recurrent connections only for a measured deficit.
   Use 5% of the current active count and round up to one. Prefer under-served
   task groups, output positions, input-output bindings, and established Dale
   groups. For recurrent connections, sort source and target evidence and scan
   stable 256 by 256 candidate tiles. Keep no more than 65,536 pairs at one
   time. Keep the global best required count and stop when a later tile cannot
   enter it. Never make a dense neuron-by-neuron candidate matrix. Do not use
   unguided random regrowth as the default.
7. Train additions with PP-Prop. Accept an addition only when at least one
   direct per-task strict result improves and no other fixed-screen task
   regresses. Repeat the bounded structural cycle only while direct strict ARC
   behavior improves.
8. Physically compact accepted checkpoints. Report retained neurons, input and
   recurrent connections, untyped and Dale-typed counts, Dale violations,
   storage, VRAM, and direct latency. Remap input targets, recurrent source and
   target indices, readout rows, Dale labels, and optimizer state.

Changing synaptic-weight values must not rebuild topology. Adding or removing
recurrent connections can rebuild the static CSR topology and cause one
intentional recompilation.

## Fast iteration loop

Each ordinary arm has a hard five-minute end-to-end limit. This includes
startup, compilation, training, decoding, and scoring. Stop cleanly at the
limit and retain the latest complete checkpoint and small result.

Use these fixed real ARC practice-task identifiers:

- training screen: `d631b094`, `dc433765`, `b782dc8a`, `d06dbe63`,
  `aedd82e4`, `0b148d64`, `b2862040`, and `150deff5`;
- validation screen: `46f33fce`, `3428a4f5`, `d8c310e9`, and `09629e4f`.

Keep task order and seeds identical across comparable arms. Load these files
directly from the existing ARC practice directory. Do not build or verify a
runtime fingerprint index for the small screen. The temporary three-minute
proof uses training task `d631b094` and validation task `46f33fce`. A later
accepted stage can use more tasks, but it must keep the complete run inside
five minutes. The 400 evaluation tasks do not enter training or routine design
iteration.

The sole ARC performance metric in normal results is direct strict task
pass-at-1. A task passes only if its first prediction is exactly correct for
every test query. Do not calculate or save partial-credit averages.

The tolerance is zero. Predicted height and width must equal the target values.
Every predicted cell must equal the target integer color. One wrong dimension
or one wrong cell makes the query incorrect. One incorrect query makes the
task's strict pass-at-1 value false.

Observe behavior directly. For each validation query, the small result may
contain only:

- task identifier and query index;
- the directly predicted integer grid;
- the target integer grid;
- exact match: `true` or `false`.

The summary contains only the strict task count and direct task records. Source
revision, schema version, configuration hash, checkpoint hash, repeated
manifests, package versions, and similar provenance fields do not belong in the
ordinary result.

Training loss may drive PP-Prop internally. It is not an ARC score and is not a
reason to keep a change when strict behavior is flat or worse. Detailed loss
or state debugging must be an explicitly requested temporary diagnostic, not a
field in every ordinary result.

Do not run the full held-out evaluation during ordinary iteration. Consider a
larger real-ARC screen only after the fixed small screen shows a repeatable
strict improvement. Run the complete evaluation only with explicit approval.

## Small result files

An ordinary `result.json` should be measured in kilobytes, not megabytes. Set a
hard 256 KiB limit for the small iteration result. Fail the result writer if it
would exceed that limit and identify the oversized field.

Do not place these values in the ordinary result:

- logits or probabilities;
- hidden state or membrane trajectories;
- channel-gate trajectories;
- eligibility traces or gradients;
- per-update loss histories;
- unrelated dataset examples;
- repeated configurations; or
- plotting arrays.

The intended shape is:

```json
{
  "tasks": [
    {
      "task_id": "27a28665",
      "queries": [
        {"query_index": 0, "prediction": [[6]], "target": [[6]], "exact": true},
        {"query_index": 1, "prediction": [[2]], "target": [[1]], "exact": false},
        {"query_index": 2, "prediction": [[2]], "target": [[2]], "exact": true}
      ],
      "strict_task_pass_at_1": false
    }
  ],
  "strict_task_pass_at_1_count": 0
}
```

If a specific debugging run needs a large array, save it as a separately named
temporary diagnostic artifact with an explicit reason and retention decision.
Do not silently expand `result.json`. Do not generate a 32 MB JSON file for
every run.

## Compact checkpoint

Keep learned parameters and structural state in one compressed NumPy archive,
not in `result.json`. Use format value `1` and `allow_pickle=False`. Store only
the neuron identifiers and codes, input and recurrent CSR topology and values,
the direct readout values, Adam moments, three optimizer step counts, neuron
count, and integration substep count. The readout weight and bias share one
optimizer step count.

Do not store predictions, targets, losses, gradients, eligibility traces, or
biological runtime state in the checkpoint. Reset voltage, gates, spikes, and
eligibility state after load. Validate every array, endpoint, count, Dale sign,
and connection limit. Set a hard 32 MiB checkpoint limit. Write a child to a
new path with an atomic replace. Never overwrite its parent checkpoint.

## Speed requirements

- Normal experimental arms: at most five minutes end to end.
- Small compatibility checks: at most three minutes end to end.
- Run the CPU and GPU preflight in separate processes. Use the first query of
  `d631b094`, all 705 events, the 2,048-neuron baseline, the production PP-Prop
  forward and gradient path, and every request loss. Do not update an optimizer
  or write a file. Warm once and time three synchronized calls. Select the
  literal lower valid median; select CPU only for an exact tie. Freeze that
  backend for all comparable experiment arms. Do not select a new backend
  between arms.
- Warmed decoder: readout, argmax, and grid slicing from the 31 executed request
  states take at most 100 milliseconds for each of five synchronized calls for
  every validation query at batch size one on the selected backend. Record the
  backend and every individual time; do not replace them with an average.
- Record cold compilation separately and include it in the five-minute total.
- Record full neuron-rollout-to-grid time separately. Fast decoding does not
  prove that Hodgkin-Huxley simulation is fast.

## Pytest budget

The complete Example 21 pytest selection should finish in at most one minute
on the reference development environment. Focused tests should take seconds.

- Keep tests small, deterministic, and co-located with the code under test.
- Compile the smallest shapes that prove the contract.
- Reuse safe fixtures and compiled forms within a test session.
- Do not load the full ARC dataset for a unit test.
- Do not run five-minute experiments through pytest.
- Mark an experiment as a separate explicit command, not as a slow test.
- Report the total Example 21 pytest duration and the slowest tests.
- Treat a regression beyond the one-minute budget as a test-design or compile
  problem that must be fixed before adding more tests.

Start with one Example 21 source module and its co-located test:

- `examples/pp_prop/21-braincell-arc.py`
- `examples/pp_prop/21-braincell-arc_test.py`

Keep the ARC encoder, BrainCell model, sparse PP-Prop training, direct decoder,
strict scorer, bounded runner, and small result writer in the one example
module. Extract a helper only after it proves useful outside ARC. A generic
BrainCell sparse-PP-Prop synapse bridge may qualify; ARC-specific machinery
does not.

## Documentation requirements

Maintain two separate user-facing Markdown documents with the implementation:

1. `docs/specs/2026-08-24-example21-causal-explanation.md` explains what was
   changed, what direct evidence was observed, which state or parameter
   interventions changed the prediction, and what the evidence does not prove.
2. `docs/specs/2026-08-24-example21-system-model.md` explains the complete
   input-to-output mechanism: ARC data, temporal encoding, Hodgkin-Huxley
   states, sparse synapses, neuron groups, Dale signs, PP-Prop updates, direct
   decoding, strict scoring, runtime limits, and result fields.

Write both documents, the example module's user-facing text, docstrings, error
messages, command help, and result labels in ASD-STE100 Simplified Technical
English. Use short sentences, one meaning per sentence, consistent terms, and
direct active voice. Define an unavoidable technical term before using it.

The causal explanation must distinguish an observation from an inference. It
must show the actual prediction and target datums that support a claim. It must
not call a changed hidden state reasoning, understanding, or success when the
strict result is false.

The system model must remain a description of executed code. Do not describe a
planned component as implemented. Update the document when the executable
input, state, synapse, learning, decoder, or result contract changes.

## Clean-code requirements

- Use descriptive variable, function, class, and field names. Do not use an
  abbreviation unless it is a defined domain term such as ARC, HH, CSR, E/I,
  JAX, or PP-Prop.
- Keep functions small and give each function one clear responsibility.
- Avoid code comments. Express intent with names, types, small functions, and
  tests. Add a comment only when an external constraint cannot be expressed in
  code, and keep that comment concise.
- Keep required public API docstrings. Write them in NumPy style and
  ASD-STE100 Simplified Technical English.
- Remove dead branches, compatibility shims, duplicated validation, unused
  configuration, and speculative extension points from the new example.
- Do not create a helper, class, schema, or abstraction for one trivial use.
- Keep ARC-specific code in the one Example 21 module. Move code into a shared
  package only after another non-ARC use proves that it is reusable.

All user-facing errors must be sentence case, concise, clear, and
ASD-STE100-compatible. Each error must state what failed and the specific value
or action needed to correct it. Do not repeat the same instruction in several
sentences. Do not add generic text such as "An error occurred," "Invalid
input," or "Please try again" when a precise correction is available.

Examples:

- Good: `Set recurrent_connection_count to 16,384; received {actual}.`
- Good: `GPU was requested, but JAX selected CPU. Install GPU support or select CPU.`
- Bad: `Invalid configuration. An error occurred. Please check your settings and try again.`

## Minimum compatibility tests

Before calling the starting model operational, require only:

1. BrainCell imports under the pinned JAX, BrainState, and BrainUnit versions.
2. One Hodgkin-Huxley step and one compiled multi-step loop produce finite
   state.
3. A tiny finite-difference check agrees with the local PP-Prop derivative in
   a valid comparison regime.
4. BrainTrace reports every intended recurrent, input, and output parameter and
   every required voltage and channel-gate state. No intended temporal
   parameter may appear in a missing-hidden-state or non-temporal compiler
   diagnostic. Every readout-only gradient value must be finite. Each height,
   width, and color head must have at least one nonzero direct-gradient value.
5. The input relation contains exactly 14,112 sparse values. The recurrent
   relation contains exactly 16,384 sparse values. Neither relation creates a
   dense input or recurrent matrix.
6. Spike-path gradients are finite and nonzero.
7. In a Dale-stage proof, assigned Dale signs remain valid after one optimizer
   update. The untyped baseline reports that no Dale constraint applies.
8. A changed trained checkpoint changes the direct prediction.
9. No target, copy shortcut, rule, candidate system, or reranker enters the
   counted prediction path.
10. A bounded run completes on the faster measured valid backend and produces
    the small strict result.

Do not run BPTT to satisfy these tests.

Use a four-neuron, one-step finite-difference fixture. Give its one input row
CSR indices `[0, 1, 2, 3]`, row pointer `[0, 4]`, and raw values
`[0.1, 0, 0, 0]`. Use no recurrent drive, zero previous spikes, reset voltage
`-65 mV`, the production current transform, `0.1 ms`, PP-Prop `single-step`,
and trace decay `0.95`. Use
`mean(tanh((voltage + 65 mV) / 20 mV))` as the objective. Perturb only the first
raw input value by plus and minus `1e-3` from separate reset-state copies.
Require
`absolute_error <= 1e-5 + 1e-2 * max(abs(pp_prop), abs(finite_difference))`.
Do not finite-difference a hard spike value.

Use a separate deterministic spike fixture. After reset, set voltage to
`-0.001 mV`, keep the reset gates, use zero previous spikes, and supply input
drive `20` through the production bounded-current path. Require one `0.1 ms`
step to cross `0 mV` before checking for a finite nonzero surrogate gradient.

## Temporary proof requirement

The proposed BrainCell path makes machine-level sense, but it has not been
tested end to end. The measured V48 input probe below applies only to the old
model. It does not prove the compact input, Hodgkin-Huxley dynamics, sparse
PP-Prop bridge, or direct output schedule.

Before structural pruning, structural addition, greater biological detail, or
a larger ARC run, complete one temporary proof with a hard three-minute
end-to-end limit. It must use real ARC practice data. When CPU and GPU are both
available, time the same compiled preflight workload in separate processes and
continue with the faster valid backend. Freeze that backend for all comparable
arms. Perform all eight optimizer updates on training task `d631b094` only.
Use validation task `46f33fce` for forward loss inspection and strict scoring
only. Its gradient, eligibility, optimizer, and structural evidence must not
change the model. The proof must show:

1. The compact temporal input reconstructs every supplied demonstration input,
   demonstration output, and query input grid exactly.
2. Different real ARC inputs produce different finite Hodgkin-Huxley voltage,
   channel-gate, and spike states.
3. BrainTrace connects every intended temporal recurrent and input weight to
   the required Hodgkin-Huxley hidden states. Missing-hidden-state or unintended
   non-temporal diagnostics for these temporal weights fail the proof. Every
   readout-only gradient is finite, and each height, width, and color head has
   at least one nonzero direct-gradient value.
4. PP-Prop changes the intended sparse synaptic weights while preserving the
   exact recurrent-connection count and every assigned Dale sign. The untyped
   baseline reports that no Dale sign is assigned.
5. The model's directly decoded integer prediction changes after training.
6. The result records each actual prediction grid, target grid, zero-tolerance
   exact boolean, and strict task pass-at-1 value. It reports no partial-credit
   score or average.
7. For each proof query, a temporary diagnostic records pretraining and
   post-training shape loss and every valid row loss beside its actual
   prediction, target, exact value, and task strict value. Do not replace these
   data with an average.
8. Apply voltage-only, sodium-gates-only, potassium-gate-only, spikes-only,
   all-state, and null interventions at the decoder boundary. Record the
   prediction, request loss, and strict result before and after each one. The
   all-state reset must change at least one prediction byte. At least one
   individual intervention must change a prediction byte or request loss. The
   null intervention must change nothing. Do not claim that one state component
   caused strict success when its strict result is unchanged.

Also report the observed total wall time, the selected backend, and the CPU and
GPU comparison times when both were available. These runtime facts may be
printed beside the result; they do not need to expand the strict result JSON.

Passing this proof establishes only that the complete mechanism executes,
learns, and affects direct behavior. It does not establish useful ARC ability.
The proof fails if the prediction is unchanged, every intended weight is
unchanged, a target enters the inference input, a shortcut generates the
answer, or the three-minute limit expires before a complete result.

## Measured input-dependency check

A bounded GPU probe used the existing V48 checkpoint on the three real ARC
practice queries in task `27a28665`. It finished in 11.46 seconds. The probe
printed the actual predicted and target grids and did not calculate an average.

| Input variant | Query 0, target `6` | Query 1, target `1` | Query 2, target `2` | Strict task |
|---|---:|---:|---:|---:|
| Full current input | `6` exact | `2` wrong | `2` exact | false |
| Without demonstrations | `8` wrong | `7` wrong | `2` exact | false |
| Without demonstration outputs | `6` exact | `2` wrong | `2` exact | false |
| Without the 9,900-value routing block | `6` exact | `6` wrong | `6` wrong | false |
| Without query-cell values | `2` wrong | `2` wrong | `2` exact | false |
| Raw ARC fields only | `2` wrong | `8` wrong | `6` wrong | false |

The current checkpoint uses demonstration rows, query values, and the large
routing block. On this task, removing demonstration outputs changed nothing.
That is poor evidence that the current model learned the demonstrated
relationship. The raw-input failure does not show that raw ARC data are
insufficient; it shows that a model trained on engineered features depends on
those features. Train the new model on the compact lossless input from the
start.

The same probe classified the current color, routing, height, and width
parameters as non-temporal. That classification is valid for a readout-only
parameter. It is not evidence of a compiler failure. The new proof separates
temporal parameter connectivity from direct readout gradients.

## Later biological detail

ARC performance comes first. Do not add biological detail until the simple
single-compartment model completes the temporary proof and produces direct ARC
predictions.

Use this preferred order after the simple model works:

1. Keep sodium, potassium, leak, and signed recurrent current coupling as the
   measured baseline.
2. Add measured 5% Dale-type stages. Test AMPA and GABAa only after a useful
   matching Dale group exists.
3. Give declared neuron groups different channel sets or time constants. This
   can support specialization without an artificial neural-network layer.
4. Add one slow biological mechanism. Test HCN, calcium-dependent adaptation,
   or NMDA in separate arms. Do not add them together in the first test.
5. Add a BrainCell electrical junction only if observed behavior identifies a
   need for direct cell-to-cell coupling.
6. Add multi-compartment cells after a single-compartment model learns ARC
   tasks. Begin with a soma and one dendrite compartment.
7. Add axons, more dendrite branches, and region-specific mechanisms only after
   the small multi-compartment model passes its runtime and strict-score gates.
8. Load a realistic SWC, ASC, NeuroML2, or NeuroMorpho.Org morphology only
   after morphology has a measured purpose in the ARC system.
9. Add one custom neuromodulatory state last. Test one declared effect, such as
   a change in plasticity strength or neuron excitability.

Calcium-dependent adaptation is the preferred first slow mechanism. It gives a
neuron a longer biological activity history. This can support context without
an LSTM or a separate latent-reasoning module. Do not call this history
reasoning or memory unless a direct causal test supports that claim.

Use BrainCell state, mechanism, and current probes during a bounded diagnostic
run. Observe voltage, channel state, synaptic current, and spike state. Do not
save these large traces in a normal result. Keep only the minimum values that
support the stated conclusion.

Benchmark BrainCell numerical solvers on the same fixed input. Begin with an
exponential-Euler solver. Compare another solver only if stability or runtime
requires it. A faster solver is acceptable only when it preserves the direct
prediction and the required finite biological states.

Each later stage must meet all of these rules:

- Change one biological feature at a time.
- Keep JAX execution and BrainState transforms.
- Keep every intended trainable parameter connected through BrainTrace ETP.
- Use PP-Prop only.
- Use the same fixed real-ARC screen as the baseline.
- Report the strict task result and direct runtime for each task.
- Keep the complete run within five minutes.
- Remove the feature if it does not produce a direct strict ARC improvement.

Biological detail is not automatically better ARC reasoning. Keep a stage only
if it stays JAX-compatible, stays PP-Prop-trainable, fits the time and memory
limits, and improves direct strict behavior on the small fixed screen.

## Resolved implementation defaults

The documentation discussion resolved these starting values:

1. Use BrainCell 0.1.0. Its basic execution check passed.
2. Start with independent exponential-Euler integration at `0.1 ms` and use
   `0.05 ms` when the stability check requires it.
3. Start with no Dale types or E/I ratio. Test measured excitatory and
   inhibitory assignments in separate 5% stages after the baseline works.
4. Start with 14,112 input connections and 16,384 recurrent connections. Use no
   permanent E/I block budget. Never let their combined count exceed 1,024
   times the current neuron count.
5. Use the existing deterministic public-practice split. Use eight fitting
   tasks and four validation tasks for ordinary iteration.
6. Use the declared task, grid, cell, input-end, shape-request, and row-request
   event schedule.
7. Prune or add 5% of the active structure in each tested stage. Keep a stage
   only when the direct strict evidence supports it.

## Reference

- Chaobrain, *BrainCell documentation*,
  <https://brainx.chaobrain.com/braincell/>.
- Chaobrain, *BrainCell API reference*,
  <https://brainx.chaobrain.com/braincell/apis/braincell.html>.
- Chaobrain, *BrainCell modeling tutorials*,
  <https://brainx.chaobrain.com/braincell/tutorials/index.html>.
- Python Package Index, *BrainCell 0.1.0*,
  <https://pypi.org/project/braincell/0.1.0/>.
