# Example 21 latent-reasoning architecture

Status: approved design; implementation and capability measurements pending

Date: 2026-08-17

Branch: `feat/example21-latent-reasoning`

Follows:

- `docs/specs/2026-08-16-example21-zero-score-root-cause.md`
- `docs/specs/2026-08-17-example21-chunked-training-stream.md`
- `docs/specs/2026-08-17-example21-demonstration-channel-root-cause.md`

## Decision

Example 21 needs a task memory that explicitly stores demonstration bindings and
a separate recurrent workspace that repeatedly reads that memory. More training
of the current reservoir is not the next experiment.

The implementation order is:

1. remove padding positions from the learning-rule timeline;
2. run the current-reservoir BPTT versus finite-window pp-prop binding control;
3. add the diagonal-friendly fast-weight memory `S_K`;
4. make the query-conditioned workspace `H_r` re-read `S_K` at every latent
   tick and decode through one shared head at every depth;
5. train and gate binding first, then application at depths represented in the
   curriculum, and only then run the full ARC evaluation.

The padding correction is necessary bookkeeping, but it is not a binding
mechanism and must not be presented as one. The BPTT result must be added to the
measurement section of this spec before the architecture is called qualified.

## Motivation: why the existing reservoir cannot learn the required binding

### The behavioral fingerprint is superposition without binding

After the already-applied physical-state padding correction, a grouped probe
separates intact episodes from `no_context` with accuracy 0.9857, while intact
versus output-rotated demonstrations remains 0.5107, at chance. Rounded to the
precision appropriate for the diagnosis, this is presence = 0.99 and pairing =
0.51.

`_derange_task` in
`examples/pp_prop/21-latent-reasoning-in-context.py` preserves the exact
multisets of demonstration inputs and outputs and changes only which output is
paired with which input. The dissociation therefore says that demonstration
content is superposed in the terminal state, but the relation between the two
sides is not bound. It is the expected fingerprint of an order-insensitive
reservoir summary, not evidence that the model needs another budget increase.

The 4096-update run supports the same conclusion behaviorally:
`shuffled_demonstrations` stays within approximately 0.012 of intact at every
effort even though `no_context` separates strongly. Loss continues to improve
while pairing does not. The earlier budget diagnosis correctly explains the
first 96-update run; it does not explain this later dissociation.

### pp-prop makes reservoir-carried temporal binding invisible to learning

The production model returns this coordinate from
`LatentWorkspaceModel.etrace_config` in
`examples/pp_prop/latent_workspace_model.py`:

```python
ETraceConfig(
    trace_factorization="io_factorized",
    recurrence_scope="diagonal",
    decay=0.9,
)
```

The implemented pp-prop rule is documented in
`braintrace/_algorithm/pp_prop.py` and executed in
`braintrace/_algorithm/io_dim_vjp.py`:

```text
epsilon_t ~= epsilon_f,t outer-product epsilon_x,t
epsilon_x,t = alpha * epsilon_x,t-1 + x_t
epsilon_f,t = alpha * diag(D_t) elementwise epsilon_f,t-1
              + (1 - alpha) * diag(D_f,t)
```

This has two independent truncations that are fatal to the current design:

- the input/output trace is rank one rather than a full per-parameter
  influence matrix; and
- `recurrence_scope="diagonal"` discards cross-neuron entries of the
  hidden-to-hidden Jacobian.

The second point is load-bearing. A binding distributed across recurrent LIF
activity requires credit to follow how one neuron's earlier state changes
another neuron's later state. That path is precisely the off-diagonal temporal
credit that this coordinate drops. A relation may be present in physical
activity without the learning rule being able to assign the later query loss
to the cross-neuron dynamics that stored it.

The retained diagonal credit is short as well. With the configured
`alpha = 0.9`, an isolated contribution is attenuated approximately as
`0.9 ** delta_t`: about 0.042 after 30 ticks, 2.7e-5 after 100, and 1.9e-14
after 300. Exact values also include the local Jacobian factors, but the scale
already shows why a demonstration-to-query gap of tens to hundreds of rows is
not a useful learning path.

This is not an assertion that pp-prop is implemented incorrectly. It is an
approximate algorithm being asked to train a substrate whose useful credit is
located in the part of the Jacobian it deliberately omits. Nor is it a claim
that recurrent activity cannot physically retain anything. The measured
presence/pairing split says it retains a pooled summary. The claim is narrower
and stronger: **the current learning rule and the current reservoir are
mismatched for learning cross-neuron temporal binding.**

### Correction to the preceding root-cause spec

`2026-08-17-example21-demonstration-channel-root-cause.md` says the trace
horizon cannot be the binding constraint because the output heads read the
final spikes directly and are non-temporal. That statement is valid only for
the last head-to-state edge. It does not cover the earlier credit needed to
make the reservoir store a particular input/output association. The terminal
head can receive an exact current-window gradient while the recurrent storage
path that should have created a bound state remains invisible.

This spec supersedes that narrow conclusion. It does not retract the probe,
the padding measurement, or the finding that the terminal heads receive a
usable current-window signal.

## Reference architecture and claim boundary

The public BDH-CQ paper (arXiv:2608.09888) discloses this system-level
interface:

```text
S_t     = U_theta(S_t-1, D_t)
H_0     = E_theta(x_query, S_K)
H_r+1   = F_theta(H_r, S_K)
y_hat   = G_theta(H_R)
```

It explicitly assigns contextual memory `S_t` and reasoning workspace `H_r`
different roles. It also describes fast-weight and linear-attention views as
the simplest conceptual realization of associative memory, while withholding
the evaluated model's dimensions and exact update rules.

Example 21 will implement this disclosed interface, not claim to reproduce the
proprietary system. Comparisons must also retain the scale and data caveat: the
paper reports a 150M-parameter model trained on private curated data plus
ARC-AGI-1 training, RE-ARC, ConceptARC, ARC-Heavy, and ARC-GEN100K. The current
Example 21 training manifest contains only the 399 public ARC-AGI-1 training
tasks and its model is approximately 2.13M parameters before this change.

The success claim is deliberately narrower than depth generalization. Table 3
in the paper shows that even BDH-CQ improves sharply when the demonstration
context includes the target ordering length or nesting depth: ordering length
8 rises from 0/24 to 13/24 pass@2, and nesting depth 5 from 19/24 to 24/24.
Example 21 may claim **binding and application at demonstrated/trained depths**.
Performance beyond the maximum demonstrated depth is a stress result, never a
release gate or an extrapolation claim.

## Required architecture

### 1. Contextual fast-weight memory `S`

`S` is a dedicated BrainState hidden state, separate from LIF voltage, spikes,
and synaptic currents. On each valid demonstration row it performs an
associative write:

```text
k_t = K(input-side row features, pair metadata)
v_t = V(output-side row features, pair metadata)
w_t = demonstration_write_gate
S_t = lambda * S_t-1 + w_t * outer(k_t, v_t)
```

The primary representation is dense per-example memory shaped
`(batch, key_width, value_width)`. A factored representation is permissible
only if its read is algebraically tested against the dense form and does not
collapse the key/value association into separate pooled sums.

Properties that are non-negotiable:

- `S` is never encoded in reservoir activity. `neu.V`, `neu.get_spike()`,
  `ff_syn.syn.g`, and `rec_syn.syn.g` remain workspace/physical states, not the
  task memory.
- Writes are gated by the demonstration phase and side-valid channels exposed
  by `RowEventConfig`; query rows and zero-input latent ticks never write.
- The recurrence of `S` with respect to itself is `lambda * I`. This can be
  represented faithfully by pp-prop's diagonal approximation; there is no
  cross-memory-cell mixing for the trace to lose.
- `lambda` is a validated finite scalar in `[0, 1]` and is applied once per
  valid demonstration ingestion tick, not once per fixed-layout padding row.
- At the end of ingestion the state is named and treated as `S_K`. It is frozen
  throughout query ingestion and latent reasoning. Its value and digest must
  remain identical across every latent checkpoint.
- Key, value, and read projections use BrainTrace ETP operators if trainable.
  A first capability implementation may freeze deterministic projections, but
  it must record that choice and may not describe them as learned. Random
  projections must come from `brainstate.random`, never `jax.random`.

The `lambda * I` self-Jacobian is the reason this memory is compatible with
diagonal temporal credit. The relevant precedent is e-prop's store/recall
success through per-neuron diagonal slow state such as ALIF adaptation: the
learning rule can preserve credit when the state variable carrying the memory
has self-local dynamics. This is the design constraint, not an optional
optimization.

### 2. Separate query-conditioned workspace `H`

The existing recurrent LIF substrate may implement the workspace, but it no
longer owns contextual memory. Its interface becomes explicit:

```text
q_0 = Q(query rows)
m_0 = Read(S_K, q_0)
H_0 = E(q_0, m_0, physical_state)

q_r = Q_r(H_r, q_0)
m_r = Read(S_K, q_r)
H_r+1 = F(H_r, m_r)
```

`Read(S_K, q_r)` must execute at **every** latent tick, including the transition
that creates `H_0`; a one-time memory read followed by zero-input reservoir
ringdown does not satisfy this spec. The same `S_K` object is read at all
depths. There may be a shared learned read projection, but not depth-specific
memory tables or per-depth heads.

The repeated read converts the task from preserving an association in a
300-tick recurrent trace into applying a currently available structure at each
workspace step. That is the load-bearing reason the architecture is learnable
under pp-prop: the memory is maintained by a diagonal-friendly transition, and
each depth receives a fresh short structural path from `S_K` to its loss.

The existing zero-input latent contract remains: external ARC event input is
exactly zero after query ingestion. Re-reading internal `S_K` is not external
input and must be reported separately so the structural gate does not mistake
it for query replay.

### 3. One decoder, supervised at every executed depth

The current training row places one nonzero loss mask at
`query_stop - 1 + effort`, and `compact_readout` consumes only spikes. This is
replaced with one shared decoder `G_theta(H_r)` evaluated at checkpoint 0 and
after every latent transition through the sampled training effort:

```text
depths supervised on an effort-R update = {0, 1, ..., R}
```

All depths use the same parameter paths. No depth-specific decoder, adapter,
or output bias is allowed. The reduction divides by the total depth weight so
longer-effort updates do not silently receive a larger optimizer step solely
because they contain more losses. If non-uniform depth weights are introduced,
they must be explicit configuration, sum to one per update, and be ablated
against uniform weighting.

The decoder should consume the continuous workspace carrier used by the
transition (for example the explicit workspace state or LIF voltage), not rely
only on a one-tick spike sample. Spikes and currents remain diagnostic outputs.
Whichever carrier is selected must be identical between training and
evaluation.

Deep supervision is structural, not cosmetic. It gives pp-prop a learning
signal adjacent to each application of `Read(S_K, q_r)` and `F`, rather than
asking one terminal loss to assign credit across the complete latent rollout.
Removing intermediate losses is a required ablation and is expected to impair
the finite-window gradient and demonstrated-depth gate.

## Stage 0: fix the padding/trace timeline first

The physical-state padding correction already in `_packed_advances` is
necessary but incomplete. `LatentWorkspaceModel.cell_step` restores voltage
and synaptic currents when `advance=False`, so the physical trajectory is
frozen. However, `braintrace/_algorithm/sequence.py` explicitly defines
`mask` as a **loss-only** gate: the learner is called and its eligibility trace
advances at every sequence position. The fixed 330-row layout therefore still
filters pp-prop's trace through padding positions even when physical state is
unchanged.

Training rows must be compacted onto the learning-rule timeline:

1. gather demonstration rows selected by the existing matched advance
   schedule, in their original order;
2. append the valid query rows;
3. append the zero-input latent rows;
4. place any static-shape padding only after the deepest supervised latent
   checkpoint.

Every learner invocation before a supervised loss then corresponds to a real
ingestion or latent transition. Static tensor shapes are retained by padding
the unused suffix, but suffix trace evolution is harmless because the learner
is reset before the next update and no later loss consumes it. Semantic
checkpoint indices become relative to the compact prefix, not the fixed ARC
layout.

Evaluation does not propagate eligibility traces and may retain the current
fixed layout and matched physical advance schedule. The intact, no-context,
and shuffled arms must continue to use byte-identical timing.

This stage is accepted only when a regression constructs padded and compact
streams with the same semantic advancing rows and proves:

- physical states and logits agree at every semantic checkpoint;
- before the fix, finite-window pp-prop gradients differ because padding
  advances the trace;
- after the fix, the production training stream matches the compact reference
  gradient;
- no false advance position occurs before any supervised depth; and
- the terminal/deep-supervision indices still select the intended outputs.

Use `chunked_online_param_gradients` for the gradient comparison. A
whole-sequence VJP is forbidden for this assertion because it removes the
finite boundary at which the trace approximation is visible.

Passing this stage permits the architecture experiment. It does **not** permit
the claim that binding is fixed.

## Stage 1: current-reservoir BPTT oracle control

Before adding `S`, measure whether the current reservoir can represent a
minimal association when trained with exact temporal credit.

### Dataset

Use a deterministic generated binding family with these properties:

- two to four one-cell colors/symbols per episode;
- demonstrations define a fresh bijection for every episode;
- the query asks for one demonstrated input under that episode's bijection;
- train and validation episodes use disjoint permutations/seeds;
- the gap is short and fixed initially, so the test isolates binding rather
  than long-span decay;
- the shuffled control rotates demonstration outputs, preserving input and
  output multisets exactly; and
- chance accuracy and the trivial marginal predictor are recorded.

The legacy model is selected through an explicit configuration such as
`context_memory_width=0`. It must use the current reservoir, current row-event
encoding, and one shared decoder; a separate toy architecture would not answer
the capability question.

### Two optimization arms and one gradient check

1. Train the minimal current reservoir with exact BPTT through a
   `brainstate.transform.for_loop` or checkpointed transform. Evaluate intact,
   shuffled, and no-context validation episodes.
2. Train the same initialization and schedule with production pp-prop.
   Learning-rule measurements must use finite windows shorter than the
   sequence.
3. On a tiny deterministic instance, compare exact gradients from
   `bptt_param_gradients` with finite-window gradients from
   `chunked_online_param_gradients`. If the helper's fixed sum-of-squares loss
   is used, wrap the model so its output is the supervised residual and
   unsupervised steps return zero; do not substitute the whole-sequence
   `online_param_gradients` path.

The run records validation accuracy, intact-minus-shuffled gap, loss curves,
per-parameter-group movement, gradient norms/cosines, sequence length, gap,
chunk size, seed, and commit.

### Interpretation gate

- **BPTT fails binding:** the current reservoir lacks the required
  architecture even with exact credit. `S_K/H_r` is necessary independently of
  the online learning rule.
- **BPTT binds and finite-window pp-prop fails:** the reservoir has physical
  capacity but pp-prop truncation is the blocker. `S_K/H_r` is the principled
  workaround because it relocates storage and per-depth credit into
  diagonal-friendly, short structural paths.
- **Both bind at the short gap:** increase only the preregistered gap in fixed
  increments until reaching the real episode range. Do not tune model width or
  optimizer between arms. A short-gap tie does not refute the measured
  full-episode failure.
- **pp-prop binds while BPTT fails:** fail closed as an oracle, seed, or
  evaluation defect; do not interpret the architecture.

Whichever valid outcome occurs must be written into the measurement section
below before Stage 2 is merged.

## Stage 2: implementation surface

### `examples/pp_prop/latent_workspace_model.py`

Extend `ModelConfig` with validated memory/workspace fields while keeping an
explicit legacy mode. Add dedicated context-memory and query/workspace states
to `LatentWorkspaceModel`, reset and snapshot them with the other BrainState
states, and expose selected diagnostics without stacking the full memory at
every tick.

Refactor the current call path so `cell_step` performs one of three explicit
roles from data-carried phase gates:

- demonstration ingestion: update `S` and physical/workspace state;
- query ingestion: freeze `S_K`, construct/query the initial workspace; or
- latent reasoning: zero external event, freeze and read `S_K`, update `H`.

`run_context`, `run_packed_stream`, `run_selected_packed_stream`, and
`run_latent_trajectory` must preserve their compiled BrainState loop/scan
drivers. Repeated model execution must not move into a Python `for`/`while`
loop. `run_selected_packed_stream` should retain only selected workspace
outputs plus compact `S_K` evidence such as a norm/digest or a selected final
snapshot; returning `(time, batch, key_width, value_width)` would undo its
memory-saving purpose.

`compact_readout` becomes the single workspace decoder. Its public compatibility
surface may be retained with a deprecation-safe argument name, but training and
evaluation must pass the same continuous `H` carrier.

`etrace_config` remains pp-prop with the declared coordinate. This change is
about making the model compatible with that approximation, not silently
switching the production learner to BPTT.

### `examples/pp_prop/21-latent-reasoning-in-context.py`

- Wire memory/workspace configuration through `_model_config`.
- Make `_training_row` build a compact learning-rule prefix and a depth mask
  covering every executed checkpoint.
- Keep `_training_chunks` random-stream identity and fixed compiled shape.
- Keep `_train_model` as one jitted outer driver containing
  `brainstate.transform.for_loop`; do not hide ETP operations behind an
  additional cell-level jit boundary.
- Preserve `_derange_task` as the pairing intervention.
- Extend `_arm_sequences` and `_evaluate` with memory/workspace diagnostics and
  the required legacy/new-architecture comparisons.
- Rename result fields such as `terminal_supervision_only`; stale evidence is
  worse than omitting it.

Training and evaluation checkpoint formats must either remain backward
compatible or carry an explicit schema/version bump. Legacy checkpoints load
only in `context_memory_width=0` mode unless a tested migration is provided.

## Capability curriculum and gates

Unit correctness gates are necessary but not sufficient. The rollout proceeds
through these behavioral gates in order.

### Gate A: associative binding

Use fresh per-episode color permutations with identical input/output marginals
between intact and shuffled arms. On at least 256 held-out query instances:

- intact exact accuracy is at least 0.80;
- the Wilson lower bound for intact is above chance;
- intact exceeds shuffled by at least 0.25 absolute;
- shuffled remains compatible with its chance/marginal baseline; and
- `S_K` differs when pairings differ even though the two marginal multisets are
  identical.

If this gate fails, stop. Do not buy a full ARC run, increase neuron count, or
describe the system as in-context learning.

### Gate B: demonstrated-depth application

Add controlled generated tasks in which a newly demonstrated operator must be
applied iteratively. Training and validation use new bindings but the same
declared depths, for example depths `{1, 2, 4, 8}`. Each evaluation depth must
have support at that depth in its demonstrations or training curriculum.

For each nonzero trained depth, report exact accuracy at `H_0` and at the
matching `H_r`. The architecture passes when:

- the final demonstrated-depth checkpoint is above its chance baseline at
  every supported depth;
- at least two nonzero depths improve by 0.15 absolute or more over `H_0` on a
  family whose target cannot be solved by one application; and
- shuffled demonstrations materially reduce final-depth accuracy.

Monotonic improvement at every tick is not required. Cherry-picking a best
undisclosed tick is forbidden; depth selection is fixed by the task. Results
beyond the largest trained/demonstrated depth are reported under
`depth_stress_only` and cannot satisfy this gate.

### Gate C: pp-prop learnability ablations

On the same seeds and data, compare:

- full `S_K` re-read plus deep supervision;
- read `S_K` only once;
- terminal-only supervision; and
- legacy reservoir (`context_memory_width=0`).

The full design must win the binding and demonstrated-depth metrics. The
finite-window gradient must change when either repeated reads or intermediate
losses are removed. These are mechanism checks; whole-sequence gradients are
not admissible evidence.

### Gate D: full ARC qualification

Only after Gates A-C pass, train on the declared real data and run the retained
400-task evaluation. The result must include:

- exact pass@1/pass@2, shape, and pixel metrics at all declared checkpoints;
- intact, shuffled-demonstrations, and no-context arms;
- pairing-sensitive state/read diagnostics;
- parameter movement and compiler evidence;
- training-data manifest and explicit comparison with the retained 4096-update
  baseline; and
- one exact held-out ARC solution (`pass@1 > 0`) before the headline "ARC score
  repaired" is allowed.

If the mechanism gates pass but exact ARC remains zero, report a successful
binding architecture with insufficient ARC task coverage/scale. Do not erase
the useful result, but do not call the original score problem complete.

## Co-located test plan

Tests stay beside the modules under test, principally:

- `examples/pp_prop/latent_workspace_model_test.py`
- `examples/pp_prop/21-latent-reasoning-in-context_test.py`

The implementation must maintain greater than 90% meaningful coverage of the
new paths. Required cases include:

1. exact `S_t = lambda * S_t-1 + outer(k_t, v_t)` numerics;
2. `jax.jacrev` on a tiny memory update equals `lambda * I` for self-state;
3. query and latent rows do not mutate `S_K`;
4. reset, snapshot, restore, batch isolation, dtype, and zero-width legacy
   mode;
5. two episodes with equal key/value marginals but different pairings produce
   different memory and reads;
6. every latent transition reads memory, demonstrated by a per-tick ablation
   or perturbation whose effect appears at the corresponding next workspace;
7. the same decoder parameter object/path serves every depth;
8. deep-supervision masks cover exactly depths `0..R` and normalize per
   update;
9. compact and padded physical trajectories agree while compact finite-window
   traces match the reference;
10. intact/shuffled/no-context schedules remain timing matched;
11. the BPTT and finite-window pp-prop controls use identical initial
    parameters and data order;
12. selected-checkpoint execution agrees with the full trajectory without
    retaining all `S` states;
13. zero external input at every latent tick remains exact;
14. invalid decay/width/phase shapes fail loudly; and
15. AST guards continue to reject bare Python model-driving loops and direct
    `jax.random` use.

The bug-first rule applies to the padding contamination: add the failing
finite-window regression before changing stream construction. The BPTT oracle
is a capability experiment, not a flaky unit test; keep a tiny deterministic
smoke case in CI and retain the larger seeded artifact as qualification
evidence.

## Performance and Example 20 optimizations

The two safe ideas retained from Example 20 are deterministic preparation
reuse and named non-inline compiled phase boundaries. They are conditional,
not automatic ports.

For Example 21:

- immutable encoded evaluation arrays may be memoized by complete content and
  configuration identity;
- a reusable, named `brainstate.transform.jit(..., inline=False, name=...)`
  wrapper around the selected-checkpoint evaluation phase may be added only if
  a trace/compile count shows that the current same-shape `run_arm` calls do
  more than one compilation; and
- fresh-process cold compile time and warm execution time must both be
  measured before accepting the change.

The following are prohibited because Example 20 already showed that they
change the algorithm or numerics:

- candidate/probe batching that changes cascade decisions;
- concatenating intact, shuffled, no-context, repeat, or ablation arms into a
  larger batch without a GPU equality proof;
- caching `repeat_intact`, which is an intentional determinism check;
- reduction reordering, changed sparse arithmetic, or tolerance changes; and
- wrapping `cell_step`, memory writes, or learner updates in a boundary that
  hides ETP primitives from the compiler.

Evaluation already invokes the nested `run_arm` repeatedly at one shape, so
JAX may be compiling it only once. If profiling confirms that, no new jit is
added. An optimization with no useful measured reduction is rejected.

Dense memory cost must be reported as
`batch * key_width * value_width * dtype_bytes`, separately for training and
the 419-query evaluation batch. Selected-checkpoint evaluation must remain
bounded by checkpoint count rather than full stream length. Any default memory
width that causes a new recovered or fatal GPU OOM fails qualification.

## Rollout and rollback

Implementation lands in small branch commits in this order:

1. this spec;
2. failing padding/trace and oracle tests;
3. compact learning-rule timeline fix and measurements;
4. `S_K/H_r` model states and unit tests;
5. repeated reads, shared decoder, and deep supervision;
6. binding and demonstrated-depth datasets/gates;
7. full evaluation/report schema; and
8. any separately measured Example 20-derived optimization.

`context_memory_width=0` (or an equivalently explicit flag) preserves the
legacy reservoir path for the oracle, regression comparison, and rollback. The
new architecture must not become the scientific default until Gates A-C pass.
Rollback is configuration-only while both modes coexist; remove neither the
legacy control nor old result readers in this change.

Every phase records the seed, config, commit, device/backend, compiler report,
test command, and artifact path. A failed gate stops the rollout at that phase
rather than triggering an unplanned width, budget, or dataset sweep.

## Measurement record

### Retained pre-change evidence

- physical padding fix, 512 updates: presence 0.9857; pairing 0.5107;
- physical padding fix, 4096 updates: intact and shuffled within approximately
  0.012 at every effort; best intact pixel approximately 0.4144, shape
  approximately 0.3842, exact 0;
- current pp-prop coordinate: IO-factorized, diagonal recurrence, decay 0.9;
- current training: one terminal loss; current latent phase: zero-input
  reservoir ringdown; current readout: terminal spikes.

### Required BPTT-control result

Pending. Before Stage 2 is merged, replace this paragraph with the exact
command/config/artifact and one of the interpretation-gate outcomes. Record
both behavioral validation and the finite-window gradient measurement. A
whole-sequence pp-prop VJP does not satisfy this requirement.

### Required post-change results

Pending: Gate A binding, Gate B demonstrated-depth application, Gate C
mechanism ablations, then Gate D full ARC qualification.

## Explicit non-claims

Until the corresponding gates pass, this work does not claim:

- that removing padding enables binding;
- that pp-prop approximates BPTT element-wise;
- that more neurons, synapses, updates, or a different learning rate solve the
  presence/pairing dissociation;
- that Example 21 reproduces BDH-CQ's proprietary implementation, training
  mixture, scale, score, or cost;
- that success at a trained depth extrapolates to a deeper unseen one; or
- that lower training loss or nonzero pixel accuracy alone is latent
  reasoning.

