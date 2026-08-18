# Example 21 latent-reasoning architecture

Status: Stage 2 and Stage 2.1 implemented; authenticated Gates A and B passed;
capability Gates C--D pending

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

#### Stage 2.1: diagonal-safe carrier stabilization

The failed Gate A diagnostic below causally localizes its initiating
optimization failure to the first Adam update of the dense readout projection,
not to a change in memory, read, spikes, or voltage. Before another Gate A run,
memory mode caps each example's continuous workspace carrier only at the two
places where its scale enters a dense projection. With fixed `C = 1.0`, define

```text
d(H) = stop_gradient(max(C, ||H||_2))
H_cap = C * H / d(H)
```

where the norm and maximum are per example over the final carrier dimension.
Use `H_cap` as the input to memory-mode `readout_projection` and
`workspace_query_projection`. Do not replace, normalize, or stop the raw `H`
stored in `workspace_carrier`; recurrence, snapshots, diagnostics, and state
restoration continue to use the raw carrier. The memory read and
`memory_read_projection` are unchanged. Because `d(H)` is stopped, the local
Jacobian is exactly `(C / d(H)) * I`: scalar diagonal, identity below the cap,
and therefore compatible with pp-prop's diagonal recurrence approximation.
The cap adds no parameter and `C` is fixed rather than tuned.

The cap is memory-mode-only. `context_memory_width=0` must not execute it, and
the complete width-zero forward outputs, states, gradients, parameter paths,
and serialized reports remain byte-exact to the pre-Stage-2.1 legacy path.
Co-located tests must cover below-cap identity, above-cap norm, the exact
scalar-diagonal Jacobian, unchanged raw carrier state, both capped projection
sites, and width-zero byte identity before an implementation can be accepted.

Stage 2.1 has two fail-closed admission checks before the 10,000-update Gate A
rerun. The one-update check uses update zero of the exact production schedule
(`training_schedule_sha256 =
25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b`),
the production topology and batch 64, the declared seeds, and exactly one
pp-prop-plus-Adam update. Retain separate H0 and H1 measurements immediately
before and after that update. It passes only when:

- pre- and post-update H0/H1 cross-entropies, color logits, capped carriers,
  gradients, pp-prop factors, Adam factors, and parameter updates are finite;
- each post-update H0 and H1 cross-entropy is no more than its corresponding
  pre-update value plus `1.0`;
- the post-update maximum absolute color logit is below `10` at each depth;
- every observed per-example capped-carrier norm is at most `1 + 2e-6`; the
  fixed `2e-6` allowance covers only backend rounding when a float32 unit-norm
  vector is remeasured in float64 and does not change the mathematical radius;
- the compact-readout recomputation and equivalent capped-readout residual are
  each at most `STAGE21_DECODER_REPLAY_ATOL = 3e-5`. This allowance is separate
  from the carrier radius and its norm tolerance. It is the fixed upper envelope
  `4 * eps_float32 * (sqrt(2048) + sqrt(128)) = 2.70e-5`, rounded upward to
  `3e-5`, for the alternate batch-64 and flattened batch-128 GEMM reductions.
  The prior unauthenticated diagnostic measured `4.3641776e-6` before the update
  and `9.8720193e-6` afterward; those values confirm the envelope but do not set
  it and no task result enters the derivation. The query-projection capped
  residual retains its `1e-6` bound; and
- the required associative, readout, and color-decoder gradient/factor group
  norms are finite and nonzero.

The second check is a fixed 256-update smoke run with batch 64 and the complete
production topology, memory width 32, decay 1.0, optimizer, model/data seeds,
and held-out evaluation protocol; only `training_updates=256` is abbreviated.
Its generated schedule digests are retained before results are inspected. It
passes only if every retained loss, state, logit, gradient/factor, Adam state,
and parameter tensor is finite; the final-64 mean training loss is below the
pre-update initial mean (the arithmetic mean of the retained update-zero H0 and
H1 cross-entropies); and intact held-out predictions contain at least two
distinct colors at both H0 and H1. A one-color prediction histogram at either
supervised depth is the prior collapse and fails the smoke. If either admission
check fails, stop before the full Gate A run; do not change `C`, learning rate,
budget, width, or data in response without a new preregistered amendment.

Any Stage 2.1 smoke or Gate A rerun must use an image built with the exact clean
source revision in its OCI revision label. Alongside each result, retain and
hash a preflight sidecar containing the resolved image ID and revision, exact
command, stdout/stderr, exit status, read-only common-Git mount, `GIT_DIR`,
`GIT_WORK_TREE`, `GIT_OPTIONAL_LOCKS=0`, expected commit/clean assertions, and
live Git agreement. No rerun can qualify from the result JSON alone.

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
- shuffled is not demonstrably above its chance/marginal baseline; and
- `S_K` differs when pairings differ even though the two marginal multisets are
  identical.

If this gate fails, stop. Do not buy a full ARC run, increase neuron count, or
describe the system as in-context learning.

### Gate B: demonstrated-depth application

Each episode uses one mapping `f` sampled without replacement from the finite
catalog of single 10-cycles over the ten ARC colors. Demonstrations contain all
ten one-cell bindings `c -> f(c)` in a random presentation order, so their input
and output marginals are each exactly one copy of every color. Query colors `x`
must be stratified as evenly as integer split sizes permit. The intact,
shuffled, and no-context arms share the same episode, query, target, event
timing, and padding. The shuffled arm applies a derangement to the ten
demonstrated outputs while preserving their exact marginal; the no-context arm
zeros the demonstrations while preserving their timing. Both controls are
scored against the intact target. Before training, the generator must verify
that every shuffled composed answer differs from the intact answer at every
qualifying depth.

Query ingestion is itself the first application of `S_K`; `H_0` is not a
pre-application state. One subsequent latent tick advances the composition by
exactly one application. Thus checkpoint targets are defined by

```text
target(H_0) = f(x)
target(H_r) = f ** (r + 1)(x), for r >= 1.
```

The qualifying efforts are exactly `R in {1, 2, 4, 8}`. Their supervision and
matching-depth scores are fixed as follows:

| Effort | Per-checkpoint training targets | Matching-depth score |
| --- | --- | --- |
| `R = 1` | `H_0=f(x)`, `H_1=f**2(x)` | `H_1` against `f**2(x)` |
| `R = 2` | `H_0=f(x)`, ..., `H_2=f**3(x)` | `H_2` against `f**3(x)` |
| `R = 4` | `H_0=f(x)`, ..., `H_4=f**5(x)` | `H_4` against `f**5(x)` |
| `R = 8` | `H_0=f(x)`, ..., `H_8=f**9(x)` | `H_8` against `f**9(x)` |

Every effort-`R` update supervises all `R + 1` checkpoints with weight
`1 / (R + 1)` each. Reusing `f**(R+1)(x)` as the target at every checkpoint is
forbidden because it trains `H_0` to skip the declared iterative computation.
For the same arm and held-out episode, the recorded `H_0` prediction must be
byte-identical across requested efforts: future rollout length cannot change a
past checkpoint.

The production regime is exactly 4,096 updates, batch 64, and 512 held-out
episodes, with 2,048 neurons, 16,384 recurrent edges, readout width 128, color
rank 16, memory width 32, memory decay 1.0, pp-prop trace decay 0.9, learning
rate 0.003, and gradient clipping norm 1.0. The schedule contains exactly 1,024
updates, or 65,536 training episodes, at each qualifying effort. It therefore
uses 262,144 distinct training cycles and 512 further distinct validation
cycles, with zero overlap. This consumes 262,656 of the `9! = 362,880` possible
single 10-cycles and leaves 100,224 unused; no cycle may be repeated to create
an effectively unbounded stream. The same 512 validation episodes are used at
all four efforts and do not consume the catalog four times.

The catalog/split seed is `20260818`. Mapping IDs use the Gate A affine
algorithm with modulus `9!`: `brainstate.random.RandomState(20260818)` draws
`offset` uniformly from `[0, 9!)` and `multiplier` from `[1, 9!)`; while
`gcd(multiplier, 9!) != 1`, update
`multiplier = multiplier % (9! - 1) + 1`; then catalog position `i` has ID
`(offset + multiplier * i) mod 9!`. The first 262,144 positions are the training
mapping IDs and the next 512 are the validation mapping IDs. A cycle ID is the
standard lexicographic Lehmer rank of a permutation `(p_1, ..., p_9)` of
`1..9`, anchored at zero: its visitation order is
`0, p_1, ..., p_9, 0`, which defines `f` on all ten colors.

Training uses episode/presentation seed `32021`; validation uses
episode/presentation seed `92021`. For each split, one
`brainstate.random.RandomState(seed).rand(count, 10)` score row per episode is
converted to its demonstration order by `np.argsort(..., axis=1,
kind="stable")`. For global episode schedule index `i`, query color is
`i mod 10` and consumes no random draw, making each split as balanced as its
integer size permits. Training effort order is the deterministic cycle
`(1, 2, 4, 8)` repeated exactly 1,024 times; it uses no effort-selection RNG.

The shuffled-output control tests color rotations `s = 1..9` in ascending
order, with `g_s(c) = (f(c) + s) mod 10`, and selects the first whose ten
rotated demonstration outputs derange all ten intact pairings and whose
composed query answer differs from the intact answer at every effort in
`{1, 2, 4, 8}`. Failure to find such a rotation invalidates the episode before
training; it is not permitted to relax a depth or keep an accidental matching
target.

Data is generated and transferred in exactly 32 fixed-shape staging chunks of
128 updates. Chunk boundaries may change only host/device staging:
concatenating the chunks must reproduce the global episode order, effort order,
queries, presentation permutations, events, targets, and hashes exactly. Every
update uses one padded 19-tick shape and explicit advance/supervision masks, so
the implementation builds one compiled training driver rather than four
effort-specific model shapes. An effort-`R` example advances exactly through
`H_R`; every remaining position in the fixed `T=19` suffix has
`advance=False`, zero loss weight, and a dummy non-semantic target. That suffix
causes no model-state evolution. Eligibility machinery may update unused
post-terminal trace internals, but no later loss may consume them; the padded
objective and parameter gradient must match a truncated `[:11 + R]` reference.
The prohibition is on interleaved padding or trace aging before a semantic
loss, not on unobserved trace state after `H_R`. The active prefix, including
`H_0`, must be byte-identical to the compact reference. A Python host loop is
permitted only over the 32 chunks and may only invoke that one prebuilt `T=19`
JIT; its internal 128-update loop must use `brainstate.transform.for_loop`. No
model, step, or eligibility-trace call may execute directly in the Python loop.
Materializing the complete `(4096, 19, 64, 47)` event tensor on device is not
required.

The held-out split is also an exact preregistered schedule, not a set from which
episodes may be selected after training. The retained artifact must record and
qualification must recompute dtype-and-shape-framed SHA-256 digests for all nine
validation fields below. These values are fixed by the generator contract above:

| Validation field | SHA-256 |
| --- | --- |
| `mapping_ids` | `b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8` |
| `query_colors` | `c7e70f56cca66d920d5d690a902b9943f2fcfdff7003fa4bbb3580070738d67e` |
| `presentation_orders` | `0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0` |
| `shuffled_shifts` | `15af1f04589cc523d89b66d2f07027158d69068901d786eecfd259a156f2f2d0` |
| `targets_by_depth` | `a438d64347dc4ec5cfc639342d8b142c785e497ddf06728eb03f8ccfb42d3cd6` |
| `advance_masks` | `b88b3593d9df51260fbafa4a937159c3da3f56fc33335a30993c0ff8a7462ac8` |
| `intact` | `5683aa84aa2ef8a1ff623e5e0b60afb3451e617728f0363d3ad84f2ea52dacde` |
| `shuffled` | `abd5eb4ab2e2a685faeb8f6bf785ad2deb97721b00e8d194b4f65d4995516be3` |
| `no_context` | `45fd14d3faefad83b0ce6d908456320afa67944b361159cfe503fdfab591162d` |

The digest framing is `dtype.str`, logical shape, then contiguous array bytes,
in that order. A different validation ID, query, presentation order, rotation,
event, target, or advance mask invalidates the run before any behavioral score
is considered.

The retained strict-JSON artifact has `schema_version=1` and
`control="example21_demonstrated_depth_gate_b"`. Its prerequisite section must
authenticate the Gate A result SHA-256
`3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632`,
manifest SHA-256
`69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf`,
and source commit `4737e9172b1c6ca99347af5b2c83fc795a294a16`. At that prerequisite
revision, `latent_workspace_model.py` has SHA-256
`467022c79123b976dd5cebc8d5ae5da37d1373bc46477133003b0b263abd8216`
and `latent_workspace_task.py` has SHA-256
`cfaec054bd42f6dccf9fb24c5fbec0cd703fdef17ba8d3b6dd68bf78366de18b`;
Gate B must recompute and retain both source hashes.

Gate B keeps model seed `2108` and the Gate A semantic topology. Its
`RowEventConfig(max_demonstrations=10)` necessarily increases input width from
41 to 47, so the Gate A full parameter count and initialization SHA are not
expected to match and must not be used as an identity check. Before training, a
fresh Gate-B init-only GPU preflight must pin the complete Gate B configuration,
parameter count, initialization SHA, and compiler-path report. That one Gate B
initialization SHA must then be unchanged across efforts and evaluation arms.
Any source, preflight, or cross-arm mismatch stops before a qualifying training
run. Gate A is prerequisite capability evidence, not a checkpoint whose learned
weights initialize Gate B.

The formal Gate B run must authenticate and retain the complete init-only
admission result and its signed launch manifest, then recompute the admission
qualification from the retained result before the first update and again before
signing the formal artifact. Copying only the inner initialization dictionary is
insufficient. Missing finite-parameter evidence, a changed stored qualification,
or any result, preflight, manifest-file, or bundle-digest mismatch stops the run.

The 10-cycle construction fixes the relevant shortcut baselines. Its uniform
output marginal is `1/10`. For every supported `R`, the final target differs
from `f(x)`, and conditional on only `x` and `f(x)` each possible
`f**(R+1)(x)` is uniform over the other eight colors. A shortcut that merely
copies the first lookup is therefore always wrong, while a method given only
`x` and `f(x)` has accuracy `1/8`. This does not claim that an unrestricted
feed-forward function of the complete `S_K` is mathematically incapable of
composition; the required matching-depth improvement is the behavioral test
for iterative use.

For every effort, retain exact counts, accuracies, prediction histograms and
hashes, and Wilson 95% intervals for `H_0` and every supervised checkpoint in
all three arms. `H_0` is reported twice without changing its prediction: once
against its proper target `f(x)`, and once against that effort's matching final
target `f**(R+1)(x)`. Gate B passes only if all of these conditions hold:

- intact matching-depth `H_R` has Wilson 95% lower bound strictly above `1/8`
  at every `R in {1, 2, 4, 8}`;
- at least two of the four efforts have
  `accuracy(H_R, f**(R+1)(x)) - accuracy(H_0, f**(R+1)(x)) >= 0.15`;
- intact matching-depth accuracy minus shuffled matching-depth accuracy is at
  least `0.15` at every effort;
- neither shuffled nor no-context has a Wilson 95% lower bound above `1/8` at
  any matching depth;
- `H_0` against its proper target `f(x)` has Wilson 95% lower bound strictly
  above `1/8`; and
- all 512 held-out episodes pass the distinct/disjoint-cycle, exact-marginal,
  balanced-query, timing, target-trajectory, shuffled-answer, no-copy-shortcut,
  and cross-effort `H_0` identity checks.

Any missing checkpoint, non-finite loss/logit/state/factor, all-one prediction
collapse at a supervised depth, schedule/configuration/provenance mismatch, or
failed condition above makes the result fail closed. Stop before Gate C, Gate
D, or an ARC run if Gate B fails. Do not reseed, change the declared budget,
depth set, topology, or thresholds under the same result; a changed regime
requires a new preregistered amendment and artifact. Monotonic improvement at
every tick is not required, and no undisclosed best tick may be selected.
Results beyond effort 8 are `depth_stress_only` and cannot satisfy Gate B.

### Gate C: pp-prop learnability ablations

Gate C is one paired causal experiment over two separate canonical regimes.
Gate A and Gate B are not byte-identical to each other: Gate A has a six-tick,
width-41 event stream, while Gate B has a 19-tick, width-47 event stream. Each
of the five interventions below is trained once from scratch in each regime,
for exactly ten fresh pp-prop trainings. Within a regime, every arm uses
byte-identical mapping IDs, presentation order, queries, encoded events,
targets, advance masks, and held-out intact/shuffled/no-context examples. No
trained checkpoint or optimizer state is carried between arms or regimes.

The canonical Gate A regime is exactly 10,000 updates at batch 64 and 512
held-out episodes, with model seed `2108`, split seed `20260817`, training
episode seed `31021`, and validation episode seed `91021`. Its training and
validation schedule SHA-256 values are respectively
`25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b`
and `80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd`;
its flattened training and validation mapping-ID SHA-256 values are
`fbd48ad9a8d3ecb0dd0812abbbda35953def52862785ce048e17b2eb9fdd3499`
and `a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3`.
The canonical Gate B regime is exactly the 4,096-update, batch-64, 512-held-out,
32-by-128-chunk regime above, with catalog seed `20260818`, training episode
seed `32021`, validation episode seed `92021`, and deterministic effort order
`(1, 2, 4, 8)` repeated 1,024 times. Its eight global training-field hashes are:

| Gate B training field | SHA-256 |
| --- | --- |
| `events` | `a1937b7f8d5d4da5f30216847cc63d022d9ec46d5cf152b25f5a30a59a1eb84f` |
| `targets` | `4082d2fd1440e9d14b0c81c754158f05b8056137a9116aee667f8d112312184c` |
| `loss_weights` | `044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31` |
| `advance_masks` | `2fc1b2acd9f73e567684d2a85f44c4009c5941ce262a527589066117ec27a4cc` |
| `mapping_ids` | `78c2d8aaa9e874dbcc1c25363875ff8aec0356a711d2426e09f2e79c76c72cb7` |
| `efforts` | `c7ca75132501bda8e6b5695a48a1ae5cde22da587f4658f7721bd4e3adcd58e6` |
| `query_colors` | `38b4cecef323dce16b0478fdd3874c9383804c913c39aaf017ce34554dcd37cb` |
| `presentation_orders` | `0650be382b381d7ab14b642c6fcdb16ae410e70a4c5821b10643bce41e3f7ca5` |

All ten trainings use production pp-prop, 2,048 neurons, 16,384 recurrent
edges, readout width 128, color rank 16, input gain 4.0, recurrent gain 0.8,
trace decay 0.9, learning rate 0.003, clipping norm 1.0, and no BPTT update.
Memory arms use width 32 and decay 1.0; only the declared legacy arm changes
memory width to zero. No arm changes a budget, topology, optimizer, or data seed
in response to an observed result.

Every Gate B arm must also reproduce all nine validation hashes frozen in the
Gate B table above. The canonical data schedule excludes only the declared
terminal-only loss intervention: that arm retains identical targets and timing
but reports a separate dtype-and-shape-framed digest of its effective loss
weights. Arm execution order is fixed as `full`, `query_only`, `terminal_only`,
`legacy`, and `frozen_write` for Gate A, followed by the same order for Gate B.
Each arm resets the model RNG to `2108` and creates a new zero-valued Adam state.
Arm order therefore cannot select or perturb initialization.

The five fixed interventions are:

1. full `S_K` re-read plus per-checkpoint supervision;
2. query-only memory read plus per-checkpoint supervision;
3. full `S_K` re-read plus terminal-only supervision;
4. legacy reservoir (`context_memory_width=0`) with the same target schedule;
5. full reads and supervision with `M_write` frozen at its initial all-ones
   value and excluded from optimizer updates.

The query-only arm performs the ordinary query read and must be byte-identical
to the full arm through `H_0` when both are evaluated from the same parameter
and state snapshot. Independently trained arms are not required to remain equal
after their parameters diverge. On every latent tick the query-only arm performs
no `S_K` read and receives zero memory-read drive, leaving only recurrent
workspace ringdown. Repeatedly injecting a cached `m_0` is a different
intervention and does not satisfy this arm. The terminal-only arm executes the
same full forward trajectory. In Gate A its effective loss weights are zero
except for weight `1.0` at `H_1`; in a Gate B effort-`R` update they are zero
except for weight `1.0` at `H_R`. Thus every arm retains total supervision weight
one: terminal-only does not silently reduce the effective learning rate by
keeping the full arm's `1 / (R + 1)` terminal weight. Full, query-only, legacy,
and frozen-write retain Gate A weights `0.5` at each of `H_0,H_1` and Gate B
weights `1 / (R + 1)` at every `H_0..H_R`.

The frozen-write arm retains the literal outer product and reports the excluded
optimizer path; it is not allowed to remove or reinitialize the memory.
`memory_write_scale` must be all ones before and after both trainings and absent
from optimizer updates, while remaining visible to the compiler and eligibility
report. The frozen-write arm must produce complete finite evidence even though
its behavioral margin is characterization-only.

Before any behavioral run, one authenticated `gate_c_init` admission at the
same source revision and image as the formal run fixes initialization. Its
strict result has `schema_version=1`,
`control="example21_gate_c_initialization_admission"`, and path
`var/example21-causal-gate/<head>-gate-c-init.json`, with companion
`.preflight.json` and `.manifest.json` files. Its only passing interpretation is
`gate_c_initialization_admission_passed`; failure records
`gate_c_initialization_admission_failed_stop` and stops before training.
Its exact top-level fields are `schema_version`, `control`,
`qualification_regime`, `prerequisites`, `regimes`, `initialization`,
`source_start`, `source_end`, `source_files`, `environment`, and
`qualification`, with `qualification_regime="preregistered_full"`.
Its exact prerequisite keys are `gate_a` and `gate_b`.
`initialization` has exact regime keys `gate_a` and `gate_b`; each regime has
exact keys `canonical_full`, `legacy`, `shared_paths`,
`arm_initialization_refs`, and `optimizer_paths`. The exact
`arm_initialization_refs` keys are the five arm names; `full`, `query_only`,
`terminal_only`, and `frozen_write` all reference one canonical full-tree
digest rather than four independently sampled initializations.
The admission may instantiate, copy, compile, and inspect the four distinct
full/legacy regime topologies, but it performs no optimizer update and computes
no held-out behavioral metric.

For Gate A, the complete full/query-only/terminal-only/frozen-write parameter
tree has 646,940 values and initial SHA-256
`b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8`.
For Gate B it has 659,228 values and initial SHA-256
`aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d`.
These whole-tree identities are scoped within a regime; Gate A and Gate B are
not required to share an initialization SHA because their input widths differ.
Seed equality alone is not identity evidence.

The width-zero legacy tree has exactly the following six same-shaped shared
parameter paths:

- `color_factor_head/weight`;
- `ff_syn/comm/weight`;
- `height_head/weight`;
- `readout_projection/weight`;
- `rec_syn/comm/weight`; and
- `width_head/weight`.

The full tree's only additional paths are `memory_read_projection/weight`,
`memory_write_scale`, and `workspace_query_projection/weight`. For each regime,
the admission copies rather than merely reseeds the six shared values into the
legacy model, requires byte equality path by path, and retains a sorted
path/dtype/shape/bytes digest of their intersection. It separately pins the
complete legacy parameter SHA, compiler paths, optimizer path set, and finite
zero-valued Adam state. The legacy parameter counts are fixed now: 514,844 for
Gate A and 527,132 for Gate B. Their exact initial SHAs and the two shared-path
intersection digests are fixed by the authenticated `gate_c_init` run on the
qualifying GPU image before any behavioral training; CPU-derived hashes are not
admissible substitutes. The formal result must authenticate the complete init
result and manifest and reproduce every initial and shared-path digest before
the first update. Each formal legacy training arm starts from those copied
canonical shared-path values, not from a separately sampled width-zero tree.

Define Gate A `binding_gap` as terminal `H_1` intact accuracy minus terminal
`H_1` shuffled accuracy. Define Gate B `depth_accuracy` as the arithmetic mean
of intact matching-depth accuracy over `{1, 2, 4, 8}`. Historical Gate A and
Gate B learned weights are prerequisites, not warm starts or substitutes for
the contemporaneous full controls. The newly trained full arm must independently
pass Gates A and B at the Gate C revision and satisfy these exact blocking
inequalities, where `G` is `binding_gap` and `D` is `depth_accuracy`:

- `D_full - D_query_only >= 0.15` and
  `G_full - G_query_only >= -0.02`;
- `D_full - D_terminal_only >= 0.10` and
  `G_full - G_terminal_only >= -0.02`; and
- `G_full - G_legacy >= 0.25` and `D_full - D_legacy >= 0.15`.

The frozen-write arm is characterization-only by default and cannot block Gate
C through a behavioral margin. Report `G_full - G_frozen_write` and
`D_full - D_frozen_write`. A margin of at least `0.05` on both metrics is
required only for the separate interpretation
`learned_memory_write_modulation_necessary`. If either margin is smaller, Gate C
may still pass through the three blocking comparisons above, but the retained
characterization is `learned_memory_write_modulation_not_shown_necessary`.

The deterministic finite-window mechanism oracle uses Gate B validation episode
index zero, intact arm, effort `R=8`, batch one, and the fresh authenticated Gate
B full initialization. Its literals are mapping ID `232423`, mapping array
`[6, 7, 5, 2, 0, 4, 8, 9, 1, 3]`, query color `4`, presentation order
`[6, 2, 5, 3, 8, 7, 4, 9, 0, 1]`, shuffled shift `1`, and targets
`[0, 6, 8, 1, 7, 9, 3, 2, 5]` for `H_0..H_8`. All 19 advance values are true.
The float32 `(19, 47)` intact event sequence has dtype-and-shape-framed SHA-256
`36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec`.
Any mismatch stops before the oracle is evaluated.

The oracle uses `chunked_online_param_gradients` with chunk size 1, strictly
shorter than the sequence. At a supervised checkpoint its wrapper emits
`sqrt(weight) * sqrt(classification_cross_entropy)` and emits exact zero at an
unsupervised checkpoint, so the helper's sum-of-squares loss is exactly the
declared weighted cross-entropy objective. Full, query-only, and terminal-only
start from the same parameter/state snapshot. For full versus query-only and
full versus terminal-only, retain global and per-parameter-path gradient norms,
L2 difference, relative deviation, cosine, and digests. Globally and for every
retained path, define relative deviation in the fixed orientation
`||g_arm - g_full||_2 / ||g_full||_2` and cosine as
`<g_arm, g_full> / (||g_arm||_2 * ||g_full||_2)`. Each record includes
`relative_deviation_defined` and `cosine_defined`. Relative deviation is JSON
null exactly when the full norm is zero; cosine is JSON null exactly when either
norm is zero. No sentinel number, NaN, or infinity is permitted. The full and
compared-arm norms must be finite and nonzero globally for both comparisons;
otherwise the oracle fails. Each global comparison must have relative deviation
at least `1e-3` and absolute L2 difference greater than
`max(1e-8, 1e-4 * full_gradient_norm)`. Removing reads must change both
`workspace_query_projection/weight` and `memory_read_projection/weight` under
the same thresholds, with finite nonzero full and query-only norms on both
paths. An unrelated path with a zero norm is retained with the null/defined
encoding above but does not by itself fail the oracle. These are mechanism
checks; whole-sequence gradients are not admissible evidence.

Gradient digests use one canonical framing. Parameter paths are sorted by their
slash-separated names. For each path, flatten its gradient subtree in JAX tree
order and hash the UTF-8 prefix `example21-gate-c-gradient-path-v1`, a NUL byte,
the UTF-8 parameter path, then for every leaf its zero-based decimal leaf index,
`numpy.dtype.str`, comma-separated logical shape, and contiguous C-order bytes,
with one NUL byte between fields. The global digest hashes the UTF-8 prefix
`example21-gate-c-gradient-global-v1`, then each sorted parameter path and its
lowercase hexadecimal per-path digest with NUL separators. Norms, differences,
and cosines flatten the same leaves and accumulate products in NumPy float64.
Missing paths, booleans in numeric fields, non-finite values, or a different
tree order fail the oracle.

The formal artifact has target `formal_gate_c`, `schema_version=1`,
`control="example21_pp_prop_learnability_gate_c"`, and path
`var/example21-causal-gate/<head>-formal-gate-c.json`, again with companion
preflight and manifest files. Its passing interpretation is exactly
`gate_c_passed_pp_prop_learnability_mechanism`; any blocking failure records
`gate_c_failed_stop_no_causal_mechanism_conclusion`. The result has exact arm
keys `full`, `query_only`, `terminal_only`, `legacy`, and `frozen_write`, each
with exact regime keys `gate_a` and `gate_b`. It retains and qualification
recomputes configuration, schedules, initialization, optimizer and compiler
paths, finite training telemetry, parameter movement, raw held-out counts,
prediction histograms and hashes, derived metrics and margins, and the mechanism
oracle. Embedded qualification booleans are never trusted.
Its exact top-level fields are `schema_version`, `control`,
`qualification_regime`, `learner`, `prerequisites`, `regimes`, `arms`,
`mechanism_oracle`, `source_start`, `source_end`, `source_files`, `environment`,
`qualification`, and `total_wall_seconds`, with
`qualification_regime="preregistered_full"` and `learner="pp_prop_only"`.
Its exact prerequisite keys are `gate_a`, `gate_b`, and
`gate_c_initialization`.

For both Gate C targets, `source_files` has exactly these six repository-relative
path keys, each mapped to its recomputed lowercase SHA-256 at the qualifying
source revision:

- `examples/pp_prop/latent_workspace_model.py`;
- `examples/pp_prop/latent_workspace_task.py`;
- `examples/pp_prop/latent_workspace_binding_control.py`;
- `examples/pp_prop/latent_workspace_binding_gate.py`;
- `examples/pp_prop/latent_workspace_depth_gate.py`; and
- `examples/pp_prop/latent_workspace_ablation_gate.py`.

The launcher is bound by the clean source HEAD, immutable image revision, fixed
argv, and authenticated preflight/postflight manifest rather than added as a
seventh scientific `source_files` key.

Both Gate C targets require one clean source revision, an immutable GPU image
whose OCI revision label is that exact commit, live Git agreement at process
start and end, the read-only common-Git mount, exact fixed argv with no topology,
seed, budget, or threshold knobs, and strict JSON that rejects duplicate keys,
NaN, and infinity. `formal_gate_c` must use the same source and image as its
authenticated `gate_c_init` prerequisite and must reauthenticate all prerequisite
files before training and before manifest signing.

Gate C also authenticates, rather than merely cites, the retained Gate A bundle:
source `4737e9172b1c6ca99347af5b2c83fc795a294a16`, result
`3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632`,
preflight `d1d54406d0972d52ac10cddec7e6d1ed38c55481d51e21989e444fe7c3f03d08`,
manifest `69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf`,
and bundle `ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875`.
It likewise authenticates the retained Gate B bundle: source
`dafa64a8b4c3848241baa117affa55b632518a8e`, result
`6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766`,
preflight `91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f`,
manifest `99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab`,
and bundle `be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851`.
Gate B authentication must recursively validate its init result
`edd058a66287e766b05c9bd1c6df31f4eed354a2e9a3e9028254935cdb744278`,
preflight `544e400f4b157dc1446216245ae0bddd38c6c93d3e59c6900890757ebe971c26`,
manifest `15070dbd9caf99c4e60690f78d1c6c3fec78ed700d385db00bff2c9aaa07bd49`,
and bundle `f16943967f04b858d614b2b821e38e7a5b198dfaa8357f5a1f1878abe73df828`.
The strict loaders recompute both scientific qualifications, artifact references,
bundle formulas, exact commands, and postflight evidence; pass flags alone are
not prerequisite evidence. A prerequisite, schema, schedule, initialization,
finite-evidence, oracle, or blocking-margin failure stops before Gate D.

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

The Docker recipe keeps its static image labels early, installs Git before the
source `COPY`, and places the commit-dependent source-revision build argument
and OCI revision label after the reusable dependency and source-install
layers. Changing the metadata value alone must therefore not invalidate those
earlier cache layers; ordinary source-content changes may still invalidate the
`COPY` and source-install layers. This ordering is only a build-cache invariant:
it does not weaken or replace the exact-label, live-Git, clean-tree, or
retained preflight requirements for a qualifying run.

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

### Retained post-architecture Gate A diagnostic

The retained diagnostic artifact is
`var/example21-binding-gate/0a33ee4-preregistered-full.json`, 290,964 bytes,
SHA-256
`c67512326c1d380bad93371685c8e8e4f10d49710a0605804ac8f916fa84b278`.
It is strict finite JSON and records matching live-Git observations at process
start and end for clean source commit
`0a33ee44f4dfd104e473b23c26bb790b04efa129`. Its recorded GPU image ID,
`sha256:d3f7838f5f591bb16c0109294b9ba4a799c01da63801103c3895cb82288a8d42`,
matches the retained local image. The configuration digest independently
recomputes to
`456bbe7c59b3d78db2afa9bb11751db161c34ea7bd82205124d4a76f4867697c`.

Independent CPU reconstruction confirms 640,000 unique training mappings, 512
unique held-out mappings, zero overlap, exact intact/shuffled input and output
marginals, and a different pairing on every held-out episode. Its mapping and
schedule digests are byte-identical to the legacy control. The qualification
schema and calculation also recompute exactly: 25 of 28 criteria are true. The
false criteria are intact accuracy at least 0.80, intact Wilson lower bound
above pairing chance, and intact-minus-shuffled at least 0.25.

Behavior failed decisively. At both H0 and H1, intact, shuffled, and no-context
accuracy are all `48/512 = 0.093750`, Wilson interval
`[0.0714405, 0.1221102]`, with pairing chance `0.25` and binding gap `0.0`.
Every arm predicts color 6 on all 512 examples. Yet intact and shuffled memory
differ on `512/512` episodes, with mean L2 difference `2.69420`; their H0/H1
read differences are `1.36865` and `3.47370`, and workspace differences are
`6.70289` and `6.94864`. All three associative compiler paths are direct and
move. The result therefore contains pairing-sensitive internal state but no
learned use of it at the output.

The retained loss sequence exposes the optimization collapse. It begins at
`2.376590`, then jumps to `96.581032`, `1332.781616`, and `1338.512939`
across the first updates before converging to `2.302586`, approximately
`ln(10)`. The dense `readout_projection` moves by L2 `8.53337` over the full
run and the color-decoder group by `3.16696`; finiteness and parameter movement
therefore do not establish a usable readout.

A separate deterministic replay of the first production batch localizes the
initiating failure; these replay measurements are diagnostic evidence and are
not fields in the retained Gate JSON. Before the update, H0/H1 cross-entropies
were `2.3483` and `2.4049`, with maximum absolute color logits `0.828` and
`1.189`. After exactly one pp-prop-plus-Adam step, they became `72.319` and
`124.933`, with maximum absolute logits `111.82` and `199.70`, while H0
voltage, spikes, and memory read were nearly unchanged. Restoring only
`readout_projection.weight` to its pre-update value rescued cross-entropy to
`2.256` and `2.396`. Restoring only the color head left `74.35` and `128.08`;
restoring only the associative-memory parameter paths left `72.06` and
`116.60`.

The first readout update has maximum absolute delta `0.00299998` across nearly
every coordinate and L2 `1.5353` over 262,272 parameters, near the dense
`0.003 * sqrt(262272)` first-Adam-step limit. This intervention evidence
causally locates the initiating blow-up at the dense readout update. Gradient
norm clipping happens before Adam; on the first step Adam's moment
normalization largely cancels that common magnitude scaling. Stage 2.1
therefore bounds the readout's carrier input while leaving the raw state and
pp-prop-visible diagonal structure intact.

This diagnostic did not formally close Gate A. Its directory contains no
retained exact preflight command, stdout/stderr, exit status, or mount/environment
sidecar required by the qualifying-container provenance contract, and the
image's OCI revision label is `uncommitted`. The live start/end source evidence
and exact image ID make the behavioral failure useful, but they do not waive
the preregistered sidecar. It motivated Stage 2.1; the subsequent authenticated
run below supersedes it for the Gate A decision. At the time of this diagnostic,
Gates B--D remained unrun; the authenticated Gate B result recorded below now
supersedes that historical status.

### Authenticated Stage 2.1 Gate A result

Gate A passed on the clean source commit
`4737e9172b1c6ca99347af5b2c83fc795a294a16`. The retained strict-JSON result is
`var/example21-binding-gate/4737e9172b1c6ca99347af5b2c83fc795a294a16-formal-gate-a.json`,
350,308 bytes, SHA-256
`3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632`.
Its authenticated manifest has bundle SHA-256
`ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875`
and records `bundle_valid=true`, `process_succeeded=true`,
`artifact_schema_verified=true`, and
`scientific_qualification_passed=true`. Live Git agreed at process start and
end, and the retained image digest is
`sha256:e9320a08de4079bd97393ba4188e05e507848cf8756c20f5cc2c2cbd4dcd31bf`.
The configuration digest is
`456bbe7c59b3d78db2afa9bb11751db161c34ea7bd82205124d4a76f4867697c`;
the training and validation schedule digests are respectively
`25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b`
and `80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd`.

At both retained checkpoints H0 and H1, intact accuracy is
`512/512 = 1.0` with Wilson 95% interval
`[0.9925530243, 1.0]`; shuffled accuracy is `0/512 = 0.0` with interval
`[0.0, 0.0074469757]`; and no-context accuracy is
`41/512 = 0.080078125`. The intact-minus-shuffled binding gap is exactly `1.0`.
Every held-out intact/shuffled memory pair differs, while exact input/output
marginals and timing are preserved. Training loss moves from
`2.30258512496948` to `6.51925624595151e-09`, with final-64 mean
`4.74524219496418e-06`. Both Stage 2.1 admission artifacts and every formal
Gate A qualification criterion passed; the recorded interpretation is
`gate_a_passed_associative_binding`. Total wall time was
`114.739125904998` seconds.

This closes associative binding at Gate A under the declared production
topology and pp-prop learner. By itself it does not establish repeated
demonstrated-depth application, depth extrapolation, ARC accuracy, or the Gate C
causal mechanism claim. The subsequent Gate B result below supplies the
separate demonstrated-depth evidence; the other claims remain gated.

### Authenticated Gate B demonstrated-depth result

Gate B passed on the clean source commit
`dafa64a8b4c3848241baa117affa55b632518a8e`. Live Git agreed at process start
and end, and the authenticated GPU image digest was
`sha256:35349cb07c49e275b15c5c563a8d75fa08b49d4b0829d86939c1c09fb1ef6d16`.
The retained formal result is
`var/example21-depth-gate/dafa64a8b4c3848241baa117affa55b632518a8e-formal-gate-b.json`,
180,875 bytes, with SHA-256
`6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766`.
Its preflight and manifest have SHA-256 values
`91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f`
and `99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab`,
respectively. The authenticated manifest records bundle SHA-256
`be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851`
and sets `bundle_valid`, `process_succeeded`, `artifact_schema_verified`, and
`scientific_qualification_passed` all to `true`.

The run executed all 4,096 preregistered pp-prop updates in 32 chunks, exactly
1,024 updates at each effort. Training loss started at
`2.3025851249694824`, ended at `1.0975582599639893`, and had final-64 mean
`0.5023841231595725`; every retained state, logit, gradient, pp-prop trace,
Adam state, and parameter finiteness check passed. Total wall time was
`142.8470507660022` seconds. Proper one-step H0 accuracy was
`511/512 = 0.998046875`, with Wilson 95% interval
`[0.9890207218, 0.9996551422]`. The same H0 prediction was byte-identical
across effort reports and scored `0/512` against every later matching target,
excluding the one-step-copy shortcut.

| Effort | Intact matching H_R | Wilson 95% interval | Shuffled | No context | Intact minus H0 | Intact minus shuffled |
| --- | --- | --- | --- | --- | --- | --- |
| `1` | `489/512 = 0.955078125` | `[0.9334960573, 0.9698822812]` | `8/512 = 0.015625` | `48/512 = 0.093750` | `0.955078125` | `0.939453125` |
| `2` | `438/512 = 0.855468750` | `[0.8223623277, 0.8832808380]` | `30/512 = 0.058593750` | `65/512 = 0.126953125` | `0.855468750` | `0.796875000` |
| `4` | `264/512 = 0.515625000` | `[0.4723816267, 0.5586356553]` | `45/512 = 0.087890625` | `51/512 = 0.099609375` | `0.515625000` | `0.427734375` |
| `8` | `187/512 = 0.365234375` | `[0.3246747562, 0.4078011864]` | `55/512 = 0.107421875` | `56/512 = 0.109375000` | `0.365234375` | `0.257812500` |

All four intact matching-depth Wilson lower bounds are strictly above the
preregistered `1/8` boundary, every intact-minus-shuffled gap exceeds `0.15`,
and all four depths improve over H0 by more than `0.15`. Neither control is
demonstrably above chance. In particular, the effort-2 no-context point estimate
is slightly above `1/8`, but its Wilson lower bound is only `0.1008675355`, so
it does not satisfy the control-side evidence threshold. Every formal Gate B
criterion passed, and the recorded interpretation is
`gate_b_passed_demonstrated_depth_application`.

This closes demonstrated-depth application through the trained effort set
`{1, 2, 4, 8}`. Accuracy declines with depth, from `0.955078125` at effort 1
to `0.365234375` at effort 8, so the result is evidence of learned composition
at demonstrated depths rather than depth-invariant performance. It does not
establish extrapolation beyond effort 8, the Gate C causal ablations, Gate D,
or any nonzero exact ARC score; those remain pending.

### Stage 2 implementation record: structural evidence only

Stage 2 is implemented on `feat/example21-latent-reasoning` in four bounded
commits: `5a68c10` exposes the associative key/value feature-index contract;
`b506a6d` adds the pp-prop associative workspace and co-located model tests;
`65fe456` integrates it into the Example 21 training, selected evaluation,
diagnostics, report, and named outer evaluation JIT; and `186c636` adds the
Gate A runner and its co-located tests. Width zero remains the default and
byte-compatible legacy path. Width 32 with decay 1.0 is the selected
architecture that passed Gate A and remains fixed for Gate B; width zero is the
legacy control.

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
Gram-matrix measurements, not learned binding evidence. The authenticated Gate
A result above separately supplies learned binding evidence for the selected
width-32 configuration; width 64 remains comparison-only and unqualified.

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

Complete: the Stage 2.1 one-update and 256-update admissions and authenticated
Gate A result passed at commit `4737e9172b1c6ca99347af5b2c83fc795a294a16`;
the authenticated Gate B demonstrated-depth result passed at commit
`dafa64a8b4c3848241baa117affa55b632518a8e`. Pending: implement and run the
now-frozen Gate C mechanism ablations and, only after a Gate C pass, Gate D full
ARC qualification. Any failed or unauthenticated gate stops this sequence.

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
