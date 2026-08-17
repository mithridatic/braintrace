# Example 21 latent-reasoning architecture

Status: Stage 2 implemented on the feature branch; structural validation in
progress; capability Gates A--D pending

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

#### Chosen compiler-safe capability implementation

The first Gate A candidate is explicitly opt-in at `context_memory_width=32`
and `memory_decay=1.0`; width zero remains the default. `lambda=1` is the
non-fading endpoint admitted by the memory equation and has self-Jacobian `I`.
It avoids discarding early rows across the static maximum of 300 valid
demonstration rows. Smaller decay values remain supported ablations and must not
be confused with pp-prop's separate trace decay of 0.9.

Key and value inputs use matched row-event features: the relevant side-valid
bit, normalized row position, row-position one-hot, side-specific normalized
height/width, side-specific height/width one-hots, mask, and position-specific
colors. Event-valid and phase channels remain gates. Demonstration identity is
excluded because it is absent from a query and would prevent lookup. At the
standard ARC capacity each side has 424 raw features. Separate deterministic
fixed bases, drawn from `brainstate.random` without consuming the legacy model's
parameter stream, map those features to the 32-dimensional memory coordinate.
The fixed choice is reported honestly; it is not called a learned key/value
projection.

The literal outer write is retained and receives an initially-one trainable
per-cell modulation:

```text
W_t = outer(k_t, v_t)
S_t = lambda * S_t-1 + w_t * (M_write elementwise W_t)
```

`M_write` is routed through `braintrace.element_wise` before the write. This
keeps the associative outer product explicit while giving pp-prop a
position-preserving trainable operation whose output coordinate matches `S`.
The frozen-ones modulation is a required ablation of the learned version.

This shape choice follows a measured compiler constraint rather than taste.
Direct trainable key/value projections produce `(batch, memory_width)` while
their first hidden target is `(batch, memory_width, memory_width)`; BrainTrace
correctly excludes those relations as shape-changing under IO-factorized
pp-prop. In contrast, the elementwise write modulation is classified
`all_direct` to `S`, and `S` retains the exact `lambda * I` self-transition the
learning rule can represent.

Use an explicit `reasoning_query` hidden state for `q_r`. A trainable shared
workspace-to-query projection maps continuous `H_r` to that state; the raw
`S_K`/`q_r` contraction is then consumed by one trainable shared
memory-read-to-workspace projection. The raw read, not a stopped copy, feeds the
workspace projection and therefore its structural gradient. A stopped copy may
populate the diagnostic `memory_read` state so differently shaped diagnostic
states do not merge into one compiler hidden group. The compiler prototype
classified the write modulation, workspace-query projection, and memory-read
projection as `all_direct` with no exclusions or warnings. A finite-window
chunk-size-one probe gave each path a nonzero pp-prop gradient. These compiler
and finite-window properties are release tests, not one-off observations.

For width 32, dense `S` storage is 4,096 bytes at Example 21's training batch
one, 262,144 bytes for the batch-64 binding gate, and 1,716,224 bytes for a
419-query evaluation batch. This is state storage only; compiler trace and
temporary allocation remain separately measured.

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

Use a fresh uniformly relabelled single 10-cycle `f` over the ARC colors in
each episode. Demonstrations contain all ten one-cell bindings `c -> f(c)` in a
random presentation order, so the input and output marginals are each exactly
one copy of every color. The query supplies a balanced color `x`. Training and
validation cycle mappings are unique and disjoint. The shuffled arm deranges
the ten demonstrated outputs while retaining their exact marginal and timing;
the generator must verify that its shuffled composed target differs from the
intact target at every reported depth.

Query ingestion is the first application of `S_K`, not depth zero before an
application:

```text
H_0 target = f(x)
H_r target = f ** (r + 1)(x)
```

The qualifying latent depths are exactly `{1, 2, 4, 8}`. An effort-`R`
training update carries the target trajectory `f(x), f**2(x), ..., f**(R+1)(x)`
and supervises every checkpoint `H_0..H_R` against its own target with uniform
weights summing to one. Reusing the final target at every checkpoint is
forbidden because it would train `H_0` to skip the iterative computation. The
4,096-update schedule contains exactly 1,024 updates at each qualifying effort.

The preregistered production regime is 4,096 updates, batch 64, 512 held-out
episodes, 2,048 neurons, 16,384 recurrent edges, readout width 128, color rank
16, memory width 32, memory decay 1.0, pp-prop trace decay 0.9, learning rate
0.003, and gradient clipping norm 1.0. Its 262,144 training mappings plus 512
validation mappings fit within the `9! = 362,880` unique 10-cycle catalog. Data
must be generated and staged in deterministic fixed-shape chunks; materializing
the complete `(4096, 19, 64, 47)` event tensor on device is not required and
must not change schedule or random-stream identity. The exact training and
validation seeds and fixed staging-chunk size remain pending preregistration;
they must be frozen in the Gate B configuration before its code or qualifying
run exists and may not be selected after inspecting results.

A 10-cycle makes the shortcut boundary exact. For every qualifying `r`,
`f**(r+1)(x) != f(x)`, so one application cannot solve the final target. The
fixed marginal-only baseline is `1/10`; even a shortcut given `x` and `f(x)` but
none of the remaining pairings can do no better than `1/8`. Gate B therefore
uses `1/8` as its conservative chance boundary. For each qualifying depth,
report exact accuracy and Wilson intervals at `H_0` and the matching `H_r`, plus
the intact, shuffled, and no-context matching-depth results. The architecture
passes only when:

- the intact Wilson 95% lower bound at `H_r` is above `1/8` at every supported
  depth;
- at least two nonzero depths improve by 0.15 absolute or more over `H_0` when
  both are scored against `f**(r+1)(x)`;
- intact exceeds shuffled by at least 0.15 absolute at every supported depth;
- all 512 held-out episodes satisfy the exact marginal, disjoint-mapping, and
  no-one-step-shortcut checks; and
- `H_0` scored against its proper one-step target `f(x)` has a Wilson 95% lower
  bound above `1/8`, so a later-depth result is not credited to an undefined
  initial workspace.

Monotonic improvement at every tick is not required. Cherry-picking a best
undisclosed tick is forbidden; depth selection is fixed by the task. Results
beyond the largest trained/demonstrated depth are reported under
`depth_stress_only` and cannot satisfy this gate.

### Gate C: pp-prop learnability ablations

Run five arms on the byte-identical Gate A and Gate B schedules, with separate
optimizer state and identical initialization on every shared parameter path:

1. full `S_K` re-read plus per-checkpoint supervision;
2. query-only memory read plus per-checkpoint supervision;
3. full `S_K` re-read plus terminal-only supervision;
4. legacy reservoir (`context_memory_width=0`) with the same target schedule;
5. full reads and supervision with `M_write` frozen at its initial all-ones
   value and excluded from optimizer updates.

The query-only arm performs the ordinary query read and must be byte-identical
to the full arm through `H_0`. On every latent tick it performs no `S_K` read
and receives zero memory-read drive, leaving only recurrent workspace ringdown.
Repeatedly injecting a cached `m_0` is a different intervention and does not
satisfy this arm. The terminal-only arm executes the same full forward
trajectory but masks every loss except the declared matching terminal. The
frozen-write arm retains the literal outer product and reports the excluded
optimizer path; it is not allowed to remove or reinitialize the memory.

Define Gate A `binding_gap` as intact accuracy minus shuffled accuracy. Define
Gate B `depth_accuracy` as the mean intact matching-depth accuracy over
`{1, 2, 4, 8}`. The full arm must independently pass Gates A and B and satisfy
these three blocking pairwise margins:

- versus query-only: `depth_accuracy` is at least 0.15 higher and
  `binding_gap` is no more than 0.02 lower;
- versus terminal-only: `depth_accuracy` is at least 0.10 higher and
  `binding_gap` is no more than 0.02 lower;
- versus legacy: `binding_gap` is at least 0.25 higher and `depth_accuracy` is
  at least 0.15 higher.

The frozen-write arm is characterization-only by default and cannot block Gate
C. Report full-minus-frozen differences for both metrics. A margin of at least
0.05 on both `binding_gap` and `depth_accuracy` is required only for a later
explicit claim that learned `M_write` modulation is necessary or load-bearing.
If either margin is smaller, Gate C may still pass through the three blocking
comparisons above, but the honest conclusion is that learned write modulation
was not shown to be needed.

Add a deterministic finite-window mechanism oracle on one shared depth
episode. It uses `chunked_online_param_gradients` with chunk size 1 strictly
shorter than the sequence. Per-checkpoint residuals are multiplied by the
square root of their declared loss weight so the helper's sum-of-squares loss
implements the intended weighted objective. For full versus query-only and
full versus terminal-only, retain global and per-parameter-group gradient
norms, L2 difference, relative deviation, cosine, and digests. The full
gradient must be finite and nonzero; each comparison must have relative
deviation at least `1e-3` and absolute L2 difference greater than
`max(1e-8, 1e-4 * full_gradient_norm)`. Removing reads must change both
`workspace_query_projection.weight` and `memory_read_projection.weight` under
the same thresholds. These are mechanism checks; whole-sequence gradients are
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

### Qualifying container provenance

Every qualifying container run uses the pinned GPU image, and that image must
contain a working Git executable. The launch mounts the worktree's common Git
directory read-only, points `GIT_DIR` at the worktree administrative directory
inside that mount, points `GIT_WORK_TREE` at the mounted worktree, and sets
`GIT_OPTIONAL_LOCKS=0`. It also supplies explicit expected-full-commit and
expected-clean assertions; omitting either assertion is a qualification
failure.

The harness must obtain the live full commit and porcelain-clean status from
Git at both process start and process end. Both observations must agree with
the supplied assertions, and the start and end commits must agree with each
other. Environment variables or report fields that merely assert a commit and
clean state without this live Git agreement are metadata only and cannot
qualify a run. Retain the resolved image digest and the exact preflight command,
exit status, and output with the run artifact so the Git-capable image and
mount/environment contract can be audited before expensive execution.

### Retained pre-change evidence

- physical padding fix, 512 updates: presence 0.9857; pairing 0.5107;
- physical padding fix, 4096 updates: intact and shuffled within approximately
  0.012 at every effort; best intact pixel approximately 0.4144, shape
  approximately 0.3842, exact 0;
- current pp-prop coordinate: IO-factorized, diagonal recurrence, decay 0.9;
- current training: one terminal loss; current latent phase: zero-input
  reservoir ringdown; current readout: terminal spikes.

### Required BPTT-control result

Exact command, executed from the clean worktree root:

```powershell
docker compose run --rm -e BRAINTRACE_SOURCE_COMMIT=f0e170d299fa5b99de0bd6f4467a9f0eba16077f -e BRAINTRACE_SOURCE_DIRTY=0 gpu python examples/pp_prop/latent_workspace_binding_control.py --output var/example21-binding-control/f0e170d-preregistered-full.json --training-updates 10000 --batch-size 64 --validation-episodes 512 --gap-steps 1 --neuron-count 2048 --recurrent-edges 16384 --readout-width 128 --color-rank 16 --learning-rate 0.003 --gradient-chunk-size 1
```

Completed on the clean source commit
`f0e170d299fa5b99de0bd6f4467a9f0eba16077f` with the production topology and
the complete preregistered evidence regime: 2,048 neurons, 16,384 recurrent
edges, readout width 128, color rank 16, four bindings, a one-tick gap, batch
64, 10,000 identical optimizer updates per arm, 512 held-out episodes, learning
rate 0.003, pp-prop trace decay 0.9, and finite-window chunk size 1. The run used
640,000 unique training mappings and 512 unique validation mappings with zero
overlap. Intact, shuffled, and no-context streams were timing matched, and the
BPTT and pp-prop arms began from the byte-identical parameter digest
`8ba7de55710a7ec6b75783f88fe67e66a38dcd826fd46e2a13929636a6241392`.

The retained strict-JSON artifact is
`var/example21-binding-control/f0e170d-preregistered-full.json`, SHA-256
`d7c921ee9c23db81e0b6f7cefdd7ac47169f71243b4441ca442e43d1ac4f08fe`.
It records `source.dirty=false`, JAX 0.11.0 on `cuda:0`, all seeds and schedule
digests, complete loss curves, parameter movement, compiler diagnostics, and
allocator statistics.

Exact BPTT did **not** bind: intact validation was `131/512 = 0.255859`,
shuffled was `115/512 = 0.224609`, and the pairing gap was only `0.031250`, far
below the preregistered `0.80` intact and `0.25` gap gate. No-context accuracy
was `48/512 = 0.093750`. Its held training probe also failed to bind
(`0.208984` intact versus `0.210938` shuffled), so the result is not explained
only by held-out mapping generalization. Production pp-prop likewise failed:
intact was `114/512 = 0.222656`, shuffled was `131/512 = 0.255859`, the gap was
`-0.033203`, and no-context was `46/512 = 0.089844`.

Both optimizers were active. BPTT loss moved from `2.303612` to `1.428634`
with a final-64 mean of `1.400210`; pp-prop moved from `2.303612` to `1.417677`
with a final-64 mean of `1.421543`. Feed-forward, recurrent, readout, and color
decoder parameter groups all had substantial nonzero movement. Convergence
near `ln(4) = 1.386294`, together with chance-level pairing and roughly
`0.09` no-context performance, is the expected fingerprint of learning the
episode's four-value set without learning which value belongs to the query key.

The learning-rule check used `chunked_online_param_gradients`, never the
whole-sequence online helper. With sequence length 6 and chunk size 1, the BPTT
gradient norm was `3.26644`, the pp-prop norm was `3.26623`, and total relative
deviation was `0.00450354`. The temporally exposed feed-forward and recurrent
groups had relative deviations `0.255812` and `0.143816`, respectively, despite
high cosine agreement. This preserves the separate finding that pp-prop drops
temporal credit; it is not the explanation for the BPTT arm's binding failure.

The recorded interpretation is
`legacy_architecture_necessary_bptt_also_fails_binding`. In the terminology of
the gate above, the current reservoir lacks the required binding architecture
even when supplied exact temporal credit, so Stage 2 proceeds with explicit
`S_K/H_r`. BPTT remains a diagnostic oracle only; pp-prop remains the production
learner. This one preregistered control establishes failure of the current
reservoir on the minimal fresh-binding task at the declared topology and
budget. It is not a theorem that every reservoir or every possible optimizer
schedule must fail.

The artifact reports 132.127 seconds total wall time: 74.041 seconds for data
generation, 17.150 seconds for BPTT compile plus 10,000 updates, 20.784 seconds
for pp-prop compile plus 10,000 updates, and 12.614 seconds for the finite-window
oracle. Peak device bytes in use were 1,297,731,328 and the peak allocator pool
was 2,183,135,232 bytes against a 12,884,901,888-byte limit.

### Stage 2 implementation record: structural evidence only

Stage 2 is implemented on `feat/example21-latent-reasoning` in four bounded
commits: `5a68c10` exposes the associative key/value feature-index contract;
`b506a6d` adds the pp-prop associative workspace and co-located model tests;
`65fe456` integrates it into the Example 21 training, selected evaluation,
diagnostics, report, and named outer evaluation JIT; and `186c636` adds the
Gate A runner and its co-located tests. Width zero remains the default and
byte-compatible legacy path. Width 32 with decay 1.0 remains an explicit
candidate until the capability gates pass.

The implementation has dedicated `context_memory`, `query_encoding`,
`reasoning_query`, diagnostic `memory_read`, and continuous
`workspace_carrier` states. Demonstration rows apply the gated
`lambda * S + outer(k, v)` update through the trainable element-wise write
modulation; query and latent rows freeze `S_K`. Query keys accumulate across
multiple query rows. Query ingestion and each advancing zero-input latent tick
construct a reasoning query, contract it with the same `S_K`, and feed the raw
read through the shared memory-read projection before the shared continuous
workspace decoder. The selected evaluator retains checkpoint workspace/read
values and one final `S_K` snapshot rather than a time-stacked memory tensor.

The co-located compiler regression requires all three associative parameter
paths -- `memory_write_scale`, `workspace_query_projection.weight`, and
`memory_read_projection.weight` -- to appear in the pp-prop eligibility graph
with `all_direct` classification. It also requires `context_memory` and
`workspace_carrier` not to collapse into one hidden group. Its learning-rule
probe uses `chunked_online_param_gradients` with finite chunk size 1 and requires
a finite nonzero gradient on each path; it does not substitute a whole-sequence
gradient and is not a Gate A accuracy result. The implemented model audit has
five hidden groups, five ETP weights, four excluded weights, four warnings, and
zero compiler errors. The three required routes are respectively bound to the
isolated `context_memory`, `reasoning_query`, and continuous workspace/LIF
groups. The four expected exclusions are decoder/head routes: the
readout-projection weight-to-weight path and the non-temporal height, width, and
color heads; they do not include an associative route. On the five-step
structural probe, chunk-size-one pp-prop L1 gradients are `0.1845880449` for the
write modulation, `13.0777063370` for the workspace-query projection, and
`10.5579442978` for the memory-read projection.

The retained all-10-color `10 x 10` RFF evidence at commit `b506a6d` records,
for the selected width-32 configuration, gate-native minimum diagonal
`0.8072461486`, maximum off-diagonal `0.3530838788`, and margin
`0.4541622698`, with zero-event norm `0`; the standard ARC encoding records
`0.7478269935`, `0.3440000117`, and `0.4038269818`, respectively, also with
zero-event norm `0`. Width 64 is comparison-only: its gate-native values are
`0.8254998922`, `0.2295354307`, and `0.5959644318`, and its standard ARC values
are `0.7770660520`, `0.2572668493`, and `0.5197992325`. These are structural
Gram-matrix measurements, not learned binding evidence; width 32 remains the
selected configuration and neither width has passed a capability gate.

The evaluation integration has a named non-inline outer JIT around the
selected-checkpoint runner. Its structural regressions require one trace across
same-shape dynamic calls and byte-exact equality with the direct selected
runner, including nonzero associative memory and reads, on both cold and warm
calls. Cold/warm wall time is recorded by the focused test, but no retained
full-scale performance artifact or Example 20 speedup claim has been accepted.
Likewise, the new co-located tests cover the required memory, state, gating,
selected-diagnostic, legacy, compiler, finite-window, and JIT paths, but no
repository-wide coverage percentage is claimed here while structural
validation remains in progress.

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
