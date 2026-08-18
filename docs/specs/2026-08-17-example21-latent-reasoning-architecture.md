# Example 21 latent-reasoning architecture

Status: Stage 2, Stage 2.1, and Gate C2 controls implemented; authenticated
Gates A and B and the fresh same-HEAD `gate_c_init` admission passed; formal
Gate C and the authenticated Gate C2 controls failed; formal Gate C2, Gate D,
and the ARC test remain stopped. Gate C3 controls are preregistered below but
are not implemented or run

Date: 2026-08-17

Amendment date: 2026-08-18

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
Gate A and 527,132 for Gate B. The authenticated `gate_c_init` admission on the
qualifying GPU image fixed the Gate A legacy whole-tree SHA-256 as
`8ba7de55710a7ec6b75783f88fe67e66a38dcd826fd46e2a13929636a6241392`
and its shared-path intersection SHA-256 as
`3222375e87d72bc2fa69713cb818af49835333dcb524f61bdd403bab7d2043b3`.
It fixed the Gate B legacy whole-tree SHA-256 as
`4d1bca77eafed499753457ac9afe359c14361623fa604ea0eec011982d2687d2`
and its shared-path intersection SHA-256 as
`ed5260e609cdc499a58a3ec11a121aecaef2159b7db6a2683c547153d1c0dbf8`.
CPU-derived hashes are not admissible substitutes. The formal result must
authenticate the complete init result and manifest and reproduce every initial
and shared-path digest before the first update. Each formal legacy training arm
starts from those copied canonical shared-path values, not from a separately
sampled width-zero tree.

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
the same thresholds, with finite nonzero full norms on both paths. Their
query-only norms may be exact zero because removing latent reads makes those
operations causally dead after `H_0`; relative deviation remains defined by the
nonzero full denominator and cosine is JSON null under the rule above. An
unrelated path with a zero norm is likewise retained with the null/defined
encoding and does not by itself fail the oracle. These are mechanism checks;
whole-sequence gradients are not admissible evidence.

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

### Authenticated Gate C initialization admission

The initialization-only Gate C admission passed at clean source commit
`c2eb27b4d51c07e4b68bd29d81101bbfff0351b8` on immutable GPU image
`sha256:e8d0d3208742281dfda9ea1a3e73ddc8e96c402fc478297b29c2876f9af7d521`.
The retained strict-JSON result is
`var/example21-causal-gate/c2eb27b4d51c07e4b68bd29d81101bbfff0351b8-gate-c-init.json`,
SHA-256
`54553ace9f5c1e2450da2f2f567107d6002625d8e7b91626b16324be270292a0`.
Its preflight SHA-256 is
`c0adca3b874ecf963b411923f8ff690a5ebfaa424dd9add8f68d24c108b62fb9`;
its manifest SHA-256 is
`669da34c5e41831cd4b6491c5d01e2c3df6b752be7e707eca8050fc3d75dbed9`;
and its authenticated bundle SHA-256 is
`a472b88f653d48bccacdef7173e7464a4dcd3d8e1d9f69a604b3cc3890e98c55`.
The manifest records `17.65792469999724` seconds for the child command
(`17.658` seconds rounded); the complete launcher invocation took `30.394`
seconds.

The admission fixed these GPU-scoped initialization identities before formal
training:

| Regime | Canonical full count and SHA-256 | Legacy count and SHA-256 | Shared-path SHA-256 |
| --- | --- | --- | --- |
| Gate A | `646940`, `b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8` | `514844`, `8ba7de55710a7ec6b75783f88fe67e66a38dcd826fd46e2a13929636a6241392` | `3222375e87d72bc2fa69713cb818af49835333dcb524f61bdd403bab7d2043b3` |
| Gate B | `659228`, `aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d` | `527132`, `4d1bca77eafed499753457ac9afe359c14361623fa604ea0eec011982d2687d2` | `ed5260e609cdc499a58a3ec11a121aecaef2159b7db6a2683c547153d1c0dbf8` |

All ten arm-specific optimizer admissions -- five arms in each regime -- have
finite, all-zero fresh Adam state and `executed_updates=0`; their included and
excluded parameter paths match the declared interventions. All 14 frozen
initialization criteria passed, including authenticated Gate A/B prerequisites,
exact source and compiler topology, copied canonical-to-legacy shared bytes,
arm initialization references, optimizer isolation, and zero behavioral
updates.

This result establishes only an authenticated, reproducible starting point for
the ten formal trainings. It contains no behavioral training, held-out metric,
causal margin, or finite-window mechanism-oracle evidence and therefore does
not pass behavioral Gate C or support a latent-reasoning mechanism conclusion.
The formal Gate C result must still reauthenticate this admission, reproduce
the pinned identities before its first update, and satisfy every existing
behavioral and mechanism criterion above.

### Authenticated formal Gate C FAIL: post-run observation

This section records the formal run after execution. It does not amend the
preregistered arms, schedules, thresholds, criteria, or stop rule above.

The formal Gate C child ran at clean source commit
`59b27d7be5cc9c37845da7bb2c81ae7203935338` on immutable GPU image
`sha256:128bca1ece0fd81e0236fa61137ffd82a9e9b54339cb583f156d64d03073bc71`.
The retained strict-JSON identities are:

| Artifact | Repository-relative path | SHA-256 |
| --- | --- | --- |
| preflight | `var/example21-causal-gate/59b27d7be5cc9c37845da7bb2c81ae7203935338-formal-gate-c.preflight.json` | `ac8c4f41d460ed91b05a8ea477456616c0ce9f43cadabea9c9ba04d7c35a9383` |
| result | `var/example21-causal-gate/59b27d7be5cc9c37845da7bb2c81ae7203935338-formal-gate-c.json` | `daf05ee63edad5183d3f509c73d9e0aeb7cb8d4c06565323e0e2a58959442e17` |
| manifest | `var/example21-causal-gate/59b27d7be5cc9c37845da7bb2c81ae7203935338-formal-gate-c.manifest.json` | `dd8a34855d4b12aa6958721a8871b829d5ad2583a4764c3a95385fba2224ee78` |

The manifest's authenticated bundle SHA-256 is
`0494b4c4258205a38cd9369980667fd62f93b55890fc9206420096d13873a3bb`.
It records `bundle_valid=true`, `process_succeeded=true`,
`artifact_schema_verified=true`, `failure=null`, and
`scientific_qualification_passed=false`. The child returned zero after
`1450.5695271` seconds because this is a completed scientific fail, not a
launcher or process failure. Preflight and postflight both recorded the same
clean HEAD and image.

The strict qualifier recorded 11 of 14 criteria true. The true criteria are
`blocking_behavioral_margins`, `canonical_schedules_complete`,
`compiler_and_training_complete`, `exact_configuration`,
`fresh_isolated_optimizers`, `frozen_write_complete`, `full_gate_a_passed`,
`initialization_authenticated`, `prerequisites_authenticated`,
`schema_and_control`, and `source_and_gpu_authenticated`. The three false
criteria are exactly:

- `full_gate_b_passed`;
- `mechanism_oracle_complete`; and
- `paired_h0_identity`.

The recorded interpretation is
`gate_c_failed_stop_no_causal_mechanism_conclusion`, and the overall
qualification is false.

The formal full arm passed Gate A. At both recorded checkpoints 0 and 1, its
intact accuracy was `512/512 = 1.000000000`, shuffled accuracy was
`0/512 = 0`, no-context accuracy was `59/512 = 0.115234375`, and the
intact-minus-shuffled binding gap was `1.000000000`. Its binding diagnostic
also recorded all `512/512` intact/shuffled final `S_K` pairs different and an
exact-zero no-context `S_K`.

The formal full arm did not reproduce the prerequisite Gate B result. Its
proper H0 accuracy was `510/512 = 0.996093750`, with Wilson lower bound
`0.9858705929`. Its preregistered effort metrics were:

| Effort | Intact, count and accuracy | Intact Wilson lower | Shuffled, count and accuracy | No context, count and accuracy | Intact minus H0 final target | Intact minus shuffled |
| --- | --- | --- | --- | --- | --- | --- |
| `1` | `439/512 = 0.857421875` | `0.8244703208` | `17/512 = 0.033203125` | `56/512 = 0.109375000` | `0.857421875` | `0.824218750` |
| `2` | `201/512 = 0.392578125` | `0.3512301417` | `48/512 = 0.093750000` | `66/512 = 0.128906250` | `0.392578125` | `0.298828125` |
| `4` | `62/512 = 0.121093750` | `0.0956215685` | `48/512 = 0.093750000` | `53/512 = 0.103515625` | `0.121093750` | `0.027343750` |
| `8` | `65/512 = 0.126953125` | `0.1008675355` | `52/512 = 0.101562500` | `52/512 = 0.101562500` | `0.126953125` | `0.025390625` |

Efforts 4 and 8 fail both full-control requirements: their intact Wilson lower
bounds are not strictly above `1/8`, and their intact-minus-shuffled gaps are
below `0.15`. Exactly efforts 1 and 2 improve over their H0 final targets by
at least `0.15`. The control-side confidence requirements pass. These facts
explain the artifact's `full_gate_b_passed=false`, but the post-run schedule
identity defect below prevents attributing the non-replication solely to model
science.

The two regime summaries record the same aggregate arm metrics and causal
margins:

| Arm | Binding gap | Mean intact depth accuracy | Full-minus-arm binding gap | Binding threshold | Full-minus-arm depth accuracy | Depth threshold | Recorded result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | `1.000000000` | `0.37451171875` | -- | -- | -- | -- | baseline |
| `query_only` | `1.000000000` | `0.16357421875` | `0` | `>= -0.02` | `0.21093750000` | `>= 0.15` | blocking pass |
| `terminal_only` | `0.996093750` | `0.10156250000` | `0.003906250` | `>= -0.02` | `0.27294921875` | `>= 0.10` | blocking pass |
| `legacy` | `0.017578125` | `0.09960937500` | `0.982421875` | `>= 0.25` | `0.27490234375` | `>= 0.15` | blocking pass |
| `frozen_write` | `0.998046875` | `0.14501953125` | `0.001953125` | `>= 0.05` | `0.22949218750` | `>= 0.05` | nonblocking characterization fail |

Thus all three blocking margin comparisons passed. Frozen-write passed its
depth margin but not its binding margin, so its recorded interpretation is
`learned_memory_write_modulation_not_shown_necessary`. Passing relative
ablation margins does not override a failed absolute full control or a failed
experimental-control identity.

The finite-window mechanism oracle was incomplete because the query-only
comparison failed one of its two required paths. For
`workspace_query_projection/weight`, the full norm was
`2.5744135613e-9`, the query-only norm was zero, the L2 difference was
`2.5744135613e-9`, and relative deviation was `1.0`. The preregistered
absolute requirement is a difference strictly greater than
`max(1e-8, 1e-4 * full_norm) = 1e-8`, so this path failed the magnitude floor.
The required `memory_read_projection/weight` path passed, and the terminal-only
comparison passed, but neither result can make the oracle complete.

The failed paired-H0 identity is an implementation/control defect, not an
admissible H0 treatment difference. Full and query-only began with matching
parameter SHA-256 values in each regime --
`b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8`
for Gate A and
`aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d`
for Gate B -- but did not remain byte-identical through checkpoint 0. In both
regimes the intact compact outputs matched while hidden-state hashes differed;
the shuffled compact outputs and hidden-state hashes both differed; and the
no-context outputs and states matched. The query-only memory-read policy was
therefore applied on the ordinary query tick that produces H0 in a way that
changed authenticated state. This violates the frozen requirement that the
query-only intervention begin only after H0 and invalidates a causal
interpretation of that comparison.

Post-run audit also found a canonical Gate B schedule-identity defect that the
strict qualifier did not reject. The full Gate C arm actually trained with a
loss-weight tensor of dtype `<f4`, shape `[4096, 19]`, and SHA-256
`84d1060278f90bed56ba6b9d76a5a918d065b19bd66fe16bdf4ae6e2bebd90e7`.
The canonical Gate B schedule is the float64/global loss-weight identity
`044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31`.
The Gate C artifact copied that canonical digest into
`training.data_identity.training_global_sha256.loss_weights` while separately
recording the different tensor it actually consumed. Consequently,
`canonical_schedules_complete=true` did not prove exact schedule reuse. This
is a protocol defect, and the full-control Gate B non-replication must not be
attributed solely to scientific performance.

Gate C therefore remains failed for both recorded scientific criteria and
protocol/control validity. There is no supported causal mechanism conclusion,
no Gate D qualification, and no new Gate D ARC score; the retained exact-ARC
baseline remains zero. Per the preregistered stop rule, no Gate D or ARC test
was run. Repair requires a new source revision and a fresh authenticated Gate C
run; this result must not be reinterpreted or patched in place.

### Gate C2: post-failure protocol amendment

This amendment is preregistered after observing the authenticated Gate C v1
failure above. It does not modify, supersede, upgrade, or reinterpret that
result. Gate C v1 at source
`59b27d7be5cc9c37845da7bb2c81ae7203935338`, its schema-1 artifact, its three
failed criteria, and its stop decision remain retained exactly as recorded.
Gate C2 is a new paired causal experiment with ten fresh trainings. It retains
the five arms, two regimes, architecture, seeds, budgets, data, behavioral
thresholds, and mechanism-oracle thresholds frozen above; it changes only the
two protocol defects identified after Gate C v1.

The exact Gate C2 result identity is:

- launcher and child target `formal_gate_c2`;
- result path
  `var/example21-causal-gate/<head>-formal-gate-c2.json`, with matching
  `.preflight.json` and `.manifest.json` files;
- `schema_version=2`;
- `control="example21_pp_prop_learnability_gate_c2"`;
- `qualification_regime="preregistered_gate_c2_full"`;
- `learner="pp_prop_only"`;
- passing interpretation
  `gate_c2_passed_pp_prop_learnability_mechanism`; and
- failing interpretation
  `gate_c2_failed_stop_no_causal_mechanism_conclusion`.

Gate C2 retains the formal Gate C top-level field set exactly:
`schema_version`, `control`, `qualification_regime`, `learner`,
`prerequisites`, `regimes`, `arms`, `mechanism_oracle`, `source_start`,
`source_end`, `source_files`, `environment`, `qualification`, and
`total_wall_seconds`. Its exact prerequisite keys are `gate_a`, `gate_b`,
`gate_c_initialization`, and `gate_c2_controls`. The existing `gate_c_init`
target remains a schema-1 initialization admission, but the old artifact is not
reusable: a new admission must be generated at the exact clean Gate C2 source
HEAD and immutable image. The `gate_c2_controls` admission and
`formal_gate_c2` must use and reauthenticate that same HEAD/image admission
before their work and again before manifest signing. All ten Gate C2 arms start
from its authenticated identities with fresh zero-valued Adam state; no Gate C
v1 trained parameter or optimizer state is a warm start.

Gate C2 has these exact 15 qualification criteria:

1. `schema_and_control`;
2. `exact_configuration`;
3. `prerequisites_authenticated`;
4. `initialization_authenticated`;
5. `canonical_schedules_complete`;
6. `consumed_gate_b_loss_weights_exact`;
7. `fresh_isolated_optimizers`;
8. `compiler_and_training_complete`;
9. `full_gate_a_passed`;
10. `full_gate_b_passed`;
11. `blocking_behavioral_margins`;
12. `paired_h0_operational_equivalence`;
13. `frozen_write_complete`;
14. `mechanism_oracle_complete`; and
15. `source_and_gpu_authenticated`.

Embedded criterion booleans are not trusted. A schema-2 Gate C2 qualifier must
recompute all 15 from retained raw evidence. Gate C v1 keeps its original
schema, control, field names, 14-criterion qualifier, and recomputation path;
the implementation must add separate Gate C2 constants and validation rather
than changing the meaning of a v1 artifact.

#### Gate C2 pretraining control admission

Gate C2 adds a separate authenticated admission that must pass before any of
the ten formal trainings begin. Its exact target is `gate_c2_controls`; its
result path is
`var/example21-causal-gate/<head>-gate-c2-controls.json`, with matching
`.preflight.json` and `.manifest.json` sidecars. Its exact identity is
`schema_version=1`,
`control="example21_gate_c2_pretraining_control_admission"`,
`qualification_regime="preregistered_gate_c2_pretraining_controls"`, and
`learner="pp_prop_only"`. Its exact top-level result keys are
`schema_version`, `control`, `qualification_regime`, `learner`,
`prerequisites`, `regimes`, `mechanism_oracle`, `source_start`, `source_end`,
`source_files`, `environment`, `qualification`, and `total_wall_seconds`.
`prerequisites` has exactly `gate_a`, `gate_b`, and
`gate_c_initialization`; the last is the newly authenticated schema-1
`gate_c_init` bundle from the same clean source HEAD and immutable image.

Each of `regimes.gate_a` and `regimes.gate_b` has exactly `spec`, `config`,
`schedule_identity`, `paired_h0_operational_equivalence`, and
`query_only_latent_no_read`. The admission's `mechanism_oracle` is the unchanged
fresh-initialization finite-window oracle frozen below. The admission performs
no behavioral training or optimizer update: it must not call an arm training
step, it must not materialize an Adam optimizer, and no model parameter is
changed. A read-only inference, state probe, or finite-window gradient probe is
not a behavioral or optimizer update.

The controls result retains raw no-update evidence at
`environment.execution_and_update_evidence`. For this target `environment` has
exactly `backend`, `devices`, `image_digest`, `jax`, `python`, and
`execution_and_update_evidence`. The evidence object has exactly
`instrumented_training_entry_points`, `trainer_factory_calls`,
`trainer_factory_call_count`, `training_step_calls`,
`training_step_call_count`, `optimizer_constructor_calls`,
`optimizer_instance_count`, `optimizer_update_calls`,
`optimizer_update_call_count`, `model_factory_calls`,
`model_constructor_calls`, `materialized_roles`, and `complete`.
`instrumented_training_entry_points` is this exact sorted list of audit labels:

```text
braintools.optim.Adam.__init__
braintools.optim.Adam.update
examples.pp_prop.latent_workspace_ablation_gate.GateCTrainer.train_chunk
examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer
examples.pp_prop.latent_workspace_binding_gate._PPPropTrainer.train
examples.pp_prop.latent_workspace_binding_gate._make_pp_prop_trainer
examples.pp_prop.latent_workspace_depth_gate._DepthPPPropTrainer.train_chunk
examples.pp_prop.latent_workspace_depth_gate._make_pp_prop_trainer
```

The two dataclass-callable labels name the compiled `train`/`train_chunk`
objects returned by their corresponding factories; the audit wrapper records
invocation at that boundary even though the callable is stored as a field.

`trainer_factory_calls`, `training_step_calls`, `optimizer_constructor_calls`,
and `optimizer_update_calls` are raw invocation arrays and must all be empty;
their four integer counts must therefore all be zero. In particular,
`optimizer_instance_count=0`: this admission creates no Adam optimizer, so an
invented step-zero optimizer report cannot substitute for the stronger zero-
instance evidence. The finite-window pp-prop algorithm state is a gradient
probe, not an Adam optimizer and not a behavioral update.

Every model construction must pass through one audited controls-model factory.
`model_factory_calls` and `model_constructor_calls` are the ordered semantic
role-name arrays, must be byte-exact to each other, contain no duplicate, and
must equal the sorted exact keys of `materialized_roles`. Each role value has
exactly `regime`, `probe`, `policy`, `initialization_tree`,
`expected_parameter_sha256`, `before_parameter_sha256`,
`after_parameter_sha256`, and `parameters_equal`. The before digest is recorded
immediately after construction, and `initialization_tree` must be
`canonical_full`. The after digest is recorded after every probe using that
role, and both must equal the expected complete-tree digest authenticated by the
new `gate_c_init` admission, using the authenticated complete parameter-tree
framing frozen under the perturbation evidence below. The qualifier recomputes
every complete parameter digest, requires exact before/after equality for every
materialized role, and requires `complete=true`. Any unregistered construction,
trainer/optimizer instance, call-log entry, missing role, or parameter change
makes `no_behavioral_or_optimizer_updates=false`.

Audit installation is transactional. If replacing any trainer, training-step,
Adam-constructor, or Adam-update boundary raises, the context manager restores
every boundary it replaced earlier in that entry attempt, empties its internal
restoration stack, and re-raises. A partially installed no-update audit may not
remain active after a failed `__enter__`.

`qualification` has exactly `criteria`, `passed`, and `interpretation`.
`criteria` has these exact nine keys:

1. `schema_and_control`;
2. `exact_configuration`;
3. `prerequisites_authenticated`;
4. `initialization_authenticated`;
5. `canonical_schedules_complete`;
6. `no_behavioral_or_optimizer_updates`;
7. `paired_h0_operational_equivalence`;
8. `mechanism_oracle_complete`; and
9. `source_and_gpu_authenticated`.

All nine are independently recomputed from raw evidence. Passing requires all
nine and records `gate_c2_pretraining_controls_passed`; otherwise the admission
records `gate_c2_pretraining_controls_failed_stop` and stops before all ten
trainings. Its authenticated loader returns exactly `target`, `source_head`,
`image_digest`, `bundle_sha256`, `manifest_sha256`, `preflight_sha256`,
`result_sha256`, and `admission`. The bundle digest is exactly
`SHA256(UTF8("example21-launch-bundle-v1\0" + target + "\0" + head +
"\0" + preflight_sha256 + "\0" + result_sha256))`. The preflight, result,
and manifest are rehashed after validation to reject a pre-signing or loading
TOCTOU change.

`formal_gate_c2.prerequisites.gate_c2_controls` is that exact authenticated
eight-key wrapper. Before update one, the formal result must copy the admission's
two `schedule_identity`, `paired_h0_operational_equivalence`, and
`query_only_latent_no_read` records and its `mechanism_oracle` record exactly.
The schema-2 qualifier compares each copied JSON value with the authenticated
admission using exact canonical JSON equality; a digest-only reference or an
embedded pass boolean is insufficient. It repeats source, image, file, and
three-sidecar authentication immediately before manifest signing.

#### Gate C2 consumed Gate B loss weights

For the four nonterminal-supervision Gate B arms -- `full`, `query_only`,
`legacy`, and `frozen_write` -- each canonical encoded chunk's
`encoded.loss_weights` is the exact host array passed as the `loss_weights`
argument to `trainer.train_chunk`. Gate C2 may not regenerate those values
through Gate C's `_loss_weights` helper, change their dtype, renormalize them,
or pass a different array at that API boundary. Concatenating the exact
arguments in fixed execution order must produce this report:

```json
{
  "dtype": "<f8",
  "shape": [4096, 19],
  "sha256": "044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31"
}
```

This remains the existing `training.loss_weights` raw-evidence location. Its
digest frames the concatenated array's `numpy.dtype.str`, logical shape, and
contiguous C-order bytes exactly as the canonical Gate B global digest does.
The report must be computed from the same arrays passed to the trainer, not
from a separately regenerated expected array. The independent
`consumed_gate_b_loss_weights_exact` criterion compares each of the four
nonterminal-arm reports directly with the pinned literal above. It may not
derive its expected value from the helper that produced the consumed value.

The `terminal_only` arm remains the sole declared loss intervention. For each
encoded chunk it constructs `zeros_like(encoded.loss_weights)` and sets weight
`1.0` only at `H_R` for that update, retaining the canonical float64 dtype and
shape. Its separately pinned concatenated report is:

```json
{
  "dtype": "<f8",
  "shape": [4096, 19],
  "sha256": "f381a6b856be26071898fc7427ee1f098bbb333b3305dc3f833c5e80750e1970"
}
```

Gate A supervision is unchanged. The canonical schedule report continues to
carry Gate B's global loss-weight digest, including for `terminal_only`, because
that report identifies the common encoded schedule; `training.loss_weights`
identifies the exact effective tensor actually supplied to each arm. Criterion
`consumed_gate_b_loss_weights_exact` validates all five Gate B arm reports: the
four nonterminal reports must equal the `044616...` float64 report and the
terminal-only report must equal the independently pinned `f381a6...` float64
report. The criterion fails if any arm is missing, if either full digest literal
does not match, or if any dtype, shape, or byte-framed digest differs.

#### Gate C2 operational H0 equivalence

A read-only diagnostic on the pinned production image showed why Gate C v1's
byte-identity result cannot isolate the query-only policy. In the reduced
production-topology GPU probe, replaying one full model with the same compiled
driver from the same restored snapshot produced a maximum hidden-state
difference of `1.192e-7`. Two copied full models with separate JITs produced a
maximum difference of `2.384e-7` on the intact and shuffled streams. The
full/query-only comparison was within that same numerical scale, and every
compact output was byte-identical. These diagnostic observations are rationale
for the amended protocol, not Gate C2 qualification evidence. They do not
change the Gate C v1 decision: its frozen byte-identity criterion failed, so its
recorded result remains failed.

Gate C2 makes GPU byte equality after execution an explicit non-claim. It uses
the project's pre-existing numerical reproducibility bound rather than a
threshold fitted to the Gate C observations: maximum per-example RMS difference
must be at most `1e-6` for the compact output and every hidden-state leaf. All
values must be finite. The compared hidden-state path set, tree and leaf order,
dtype, and logical shape must match exactly, and decoded predictions must be
exactly equal.

In each Gate C2 regime the old `paired_h0_identity` regime field is replaced by
`paired_h0_operational_equivalence`. Its exact top-level keys are `backend`,
`checkpoint`, `intervention_boundary`, `rms_tolerance`,
`initialization_parameter_sha256`, `streams`, and `passed`. The fixed values are
`backend="canonical_production_sparse"`, `checkpoint=0`,
`intervention_boundary="after_ordinary_query_h0_before_first_latent_tick_h1"`,
and `rms_tolerance=1e-6`. A dense or `jax_raw` substitute is not admissible.

`initialization_parameter_sha256` has exact keys `full_reference`,
`full_replay`, `copied_full`, and `query_only`. Every value must equal the
authenticated canonical parameter digest for that regime. The exact stream
keys remain `intact`, `shuffled`, and `no_context`. Each stream record has exact
keys `initial_state_sha256`, `comparisons`, and `passed`.
`initial_state_sha256` has the same four role keys as the parameter record. The
four models or replays receive independently materialized deep copies of one
canonical initialized hidden-state snapshot; the copies must not share mutable
state, and all four initial snapshot digests must be byte-identical before the
first event.

`comparisons` has exact keys `same_full_replay`,
`copied_full_separate_jit`, and `copied_full_vs_query`. The first runs one full
model and one compiled production driver twice from separate exact snapshot
copies. The second runs two copied full models through separately constructed
and compiled production drivers. The third runs a copied full model and the
query-only model through separately constructed and compiled production
drivers. Each comparison record has exact keys `compact`, `hidden_paths`,
`predictions`, and `passed`.

`compact` and every value in `hidden_paths` use one exact floating
difference-record schema: `left`, `right`,
`per_example_sum_squared_difference`, `per_example_compared_value_count`,
`per_example_rms_difference`, `per_example_max_abs_difference`,
`sum_squared_difference`, `rms_difference`,
`max_per_example_rms_difference`, `max_abs_difference`, `within_tolerance`.
Each `left` and `right` endpoint has exactly `dtype`, `shape`, `sha256`,
`value_count`, `per_example_finite_count`,
`per_example_nonfinite_count`, `finite_count`, and `nonfinite_count`; left and
right geometry is retained separately and must agree. The endpoint digest uses
the project's dtype-and-shape-framed contiguous C-order bytes. Every
per-example vector has length 512. The qualifier recomputes endpoint totals,
`sum_squared_difference`, both RMS aggregates, both maxima, and the tolerance
decision in NumPy float64 from those vectors. It requires all nonfinite counts
to be zero and all counts to agree with the endpoint shapes. A nonfinite
difference is an immediate failure rather than a JSON coercion.

The compact endpoint geometry is exactly dtype `<f4`, shape `[512, 1180]`.
`hidden_paths` has exactly these sorted key, dtype, and shape entries:

```text
context_memory#0       <f4 [512, 32, 32]
ff_syn/post/V#0        <f4 [512, 2048]
ff_syn/syn/g#0         <f4 [512, 2048]
memory_read#0          <f4 [512, 32]
query_encoding#0       <f4 [512, 32]
reasoning_query#0      <f4 [512, 32]
rec_syn/syn/g#0        <f4 [512, 2048]
workspace_carrier#0    <f4 [512, 2048]
```

A missing, additional, renamed, or reordered leaf, or any dtype or shape
change, fails closed. The batch leading axis defines examples. For each example,
RMS is `sqrt(sum_squared_difference / compared_value_count)` and the required
bound is independently `<= 1e-6` for every one of the three comparisons,
without scaling it by an observed control residual.

Any compared bool or integer array uses the exact discrete difference-record
schema `left`, `right`, `per_example_hamming_count`, `hamming_count`, and
`exact_equal`. Its endpoint schema is the same separate dtype, shape, digest,
and count schema above, and it qualifies only at Hamming distance zero.
`predictions` has exactly `left`, `right`, `per_example_hamming_count`,
`hamming_count`, and `equal`. Each prediction endpoint has exactly `dtype`,
`shape`, `sha256`, `histogram`, and `count`; predictions are explicitly cast to
dtype `<i4` with shape `[512]`, and `histogram` is the ten-count vector for
decoded colors 0 through 9. The qualifier requires each histogram to sum to
512, recomputes `hamming_count` from the 512-entry zero-or-one vector, and
derives `equal` from Hamming distance zero plus equal endpoint digests and
histograms. It does not trust the embedded equality value.

All three comparisons must independently satisfy the `1e-6` bound for the
compact output and every hidden leaf and must have exact decoded-prediction
equality. The qualifier recomputes each comparison and the enclosing `passed`;
embedded booleans are not trusted.

This equivalence evidence is operational rather than a required source-code
shape. Gate C2 does not require one graph, require `lax.cond`, forbid the
existing pre-einsum mask, or claim that dead-branch arithmetic is physically
unexecuted. BrainTrace's default `ControlFlowPolicy(cond="convert")` may
if-convert an ETP-relevant conditional so that both branch bodies execute and a
selection discards the dead value. Leaving such a conditional opaque instead
would make weights inside it error or be excluded. Gate C2 therefore qualifies
the selected values and their causal influence, not physical instruction
nonexecution.

`query_only_latent_no_read` has exact keys `streams`, `perturbations`,
`full_positive_control`, `removed_path_finite_window_influence`, and `passed`.
`streams` has exactly `intact`, `shuffled`, and `no_context`. In the Gate A
regime each stream has exactly tick `H1`; in Gate B each has exactly ticks `H1`
through `H8`. Every tick record has exactly `selected_read`, `selected_drive`,
`cached_read_probe`, and `cached_h0_read_reused`, with the last value required
to be false.

`selected_read` and `selected_drive` are raw zero-array records with exact keys
`dtype`, `shape`, `sha256`, `value_count`, `finite_count`, `nonfinite_count`,
`zero_count`, `sum_of_squares`, `max_abs`, and `exact_zero`. The selected read
must be `<f4 [512, 32]`; the selected projection drive must be
`<f4 [512, 2048]`. The qualifier independently constructs an all-zero array of
the recorded geometry, applies the project's dtype-and-shape-framed digest,
and requires its digest to equal `sha256`. It also requires `finite_count` and
`zero_count` to equal the shape product, `nonfinite_count=0`, and both numeric
aggregates to be exact zero.

`cached_read_probe` has exactly `source_cached_memory_read`, `plus_11`,
`minus_11`, and `passed`. `source_cached_memory_read` is captured from the
query-only model's `memory_read#0` state immediately before the named latent
tick: at H1 this is the state after the ordinary-query H0 boundary, and at each
later Gate B tick it is the state after the preceding latent tick. It uses the
floating endpoint schema above and must have dtype `<f4`, shape `[512, 32]`,
`value_count=16384`, and no nonfinite values. It is a retained source record,
not an asserted zero array.

`plus_11` and `minus_11` each have exactly `replacement`, `boundary`,
`selected_read`, `selected_drive`, `continuation`, and `passed`. Their
`replacement` records have exactly `fill_value`, `dtype`, `shape`, and
`sha256`. The two replacement tensors are respectively all `+11.0` and all
`-11.0`, with dtype `<f4`, shape `[512, 32]`, and project-framed digests:

```text
+11.0  156517ec70f2d721974202ac8581ca7f15594db382051fafbac40fb9057c81bc
-11.0  b5725644875e21d4fce1fe5116695c12d18af3d9b8f243cbdd6878c3404201f6
```

The qualifier independently reconstructs both arrays and recomputes each
digest as
`SHA256(ASCII(dtype.str) || ASCII(str(shape)) || contiguous_C_bytes)`. The
source digest must differ from each replacement digest; an intervention that
does not change the cached-read bytes is inadmissible.

Each replacement's `boundary` has exactly `before_replacement`,
`after_replacement`, `changed_paths`, `unchanged_paths`,
`parameters_equal`, `only_memory_read_replaced`, and `passed`.
`before_replacement` and `after_replacement` each have exactly `hidden_paths`,
`hidden_state_tree_sha256`, and `parameter_tree_sha256`. `hidden_paths` has all
eight frozen H0 paths and geometries above, and every value uses the floating
endpoint schema above. `before_replacement.hidden_paths.memory_read#0` must
equal `source_cached_memory_read`; the corresponding `after_replacement`
endpoint must equal the pinned replacement. `changed_paths` is exactly
`["memory_read#0"]`; `unchanged_paths` is the sorted other seven H0 paths.
Every unchanged before/after endpoint digest must be equal, and both parameter
tree digests must be equal to each other and to the authenticated canonical
parameter digest for the regime.

Each boundary tree digest is independently recomputed by starting a field list
with `UTF8("example21-gate-c2-cached-read-boundary-state-v1")`, then appending,
for every sorted complete hidden path, its UTF-8 path, ASCII leaf index, ASCII
`numpy.dtype.str`, ASCII comma-joined logical shape, and the endpoint's
lowercase ASCII `sha256`, and hashing the fields joined by one NUL byte. The
endpoint `sha256` is computed at evidence generation directly from that
snapshot's actual array using the project-framed dtype, shape, and contiguous
C-order bytes; it is then retained in `hidden_paths`. The tree qualifier
recomputes only from those retained endpoint records and may not assume absent
raw tree bytes. The before and after hidden-state tree digests must differ
because the source and replacement `memory_read#0` endpoint digests differ.
Production must capture two distinct immutable tree snapshots on opposite
sides of the replacement and retain their separate endpoint and tree digests.
Comparing one post-replacement tree with itself, aliasing the two snapshot
objects, or populating both evidence sides from one capture is a failure even
if an embedded equality boolean is true.

The qualifier recomputes `parameters_equal` as exact equality of the two
complete parameter-tree digests plus equality to the authenticated canonical
parameter digest. It recomputes `only_memory_read_replaced` from the exact
one-element `changed_paths`, the exact seven-element `unchanged_paths`, the
different source/replacement `memory_read#0` endpoint digests, equality of all
seven other endpoint records, and `parameters_equal`. The two fields must equal
those recomputed predicates. `boundary.passed` must equal their conjunction;
none of these embedded booleans is authoritative.

The replacement branch's `selected_read` and `selected_drive` use the raw
zero-array schema and must remain exact zero. `continuation` has exactly
`ticks` and `passed`. For Gate A H1, `ticks` has exactly `H1`; for a Gate B
probe started at Hn, it has every tick from Hn through H8 in order. Each
continuation tick has exactly `compact`, `hidden_paths`, `predictions`, and
`passed`, comparing an unmodified query-only continuation with the replacement
continuation from independent copies of the same captured before-boundary
state. `compact`, all eight frozen hidden paths, and `predictions` reuse the
independently recomputable comparison schemas above: every floating comparison
must independently be finite and within `1e-6`, every discrete comparison must
have Hamming distance zero, and decoded predictions must be exact.

Each continuation tick's `passed` must equal the conjunction of its recomputed
compact, hidden-path, and prediction predicates. `continuation.passed` must
equal the conjunction over its exact required tick set. Each sentinel's
`passed` must equal the conjunction of its reconstructed replacement,
`boundary.passed`, exact-zero selected read, exact-zero selected drive, and
`continuation.passed`; `cached_read_probe.passed` must equal the conjunction of
the `plus_11` and `minus_11` sentinel predicates. Any disagreement fails
closed.

The qualifier does not trust `cached_h0_read_reused`. It derives that value as
false only when both fixed replacements have valid, distinct before/after
boundaries, exact-zero selected read and drive, and passing numeric and decoded-
prediction comparisons at every retained continuation tick. Otherwise the
tick, stream, regime, and admission fail closed. This probe is required for
Gate A H1 and Gate B H1 through H8 in every one of `intact`, `shuffled`, and
`no_context`; a boolean alone cannot certify nonreuse. The passing conclusion
is narrowly that the declared `memory_read#0` state captured before each named
tick does not influence these query-only continuations. It does not establish
the absence or nonreuse of any unprobed cache path elsewhere in the model.

`perturbations` has exactly `plus_7` and `minus_7`. Each has exact keys
`replacement`, `streams`, and `passed`. The replacements are respectively the
all-`+7.0` and all-`-7.0` tensors with dtype `<f4` and shape
`[512, 32, 32]`; their dtype-and-shape-framed digests are respectively
`b7b1338c1b2b0124633638a1823ec4e7a4ba8be321eb7306153c0ca8db8c696e`
and `815cda0e5c57f2387a6c645d372de7ed2df8e9b9be232aeaef6534da35194572`.
`replacement` has exactly `fill_value`, `dtype`, `shape`, and `sha256`. These
are the only preregistered perturbation tensors; there is no universal claim
over arbitrary finite `S_K` values.

Each perturbation repeats the exact regime stream and tick keys above. A tick
record has exactly `source_s_k_sha256`, `replacement_s_k_sha256`,
`source_replacement_differ`, `non_s_k_state`, `parameters`, `selected_read`,
`selected_drive`, `continuation`, and `passed`. `replacement_s_k_sha256` must
equal the pinned replacement digest and differ from the actual source digest.
`non_s_k_state` and `parameters` each have exactly `paths`, `framing`,
`left_tree_sha256`, `right_tree_sha256`, `left_value_sha256`,
`right_value_sha256`, `tree_equal`, and `values_equal`; both exact equality
results must recompute true before the continuation. The sole changed boundary
value is `context_memory#0`, or `S_K`; the parameter tree and every other state
path remain byte-identical.

For every `+7` and `-7` tick probe, the `left_*` fields come only from one
immutable source-boundary snapshot captured immediately before the S_K
replacement. The `right_*` fields come only from a second immutable snapshot
captured after replacing `context_memory#0` and before executing the
continuation. The source snapshot's actual `context_memory#0` array produces
`source_s_k_sha256`; the post-replacement snapshot's actual array produces
`replacement_s_k_sha256`. The seven non-S_K state paths populate the left and
right sides of `non_s_k_state`, and the complete parameter tree populates the
left and right sides of `parameters`. Thus `source_replacement_differ` together
with exact seven-state and complete-parameter equality recomputes the predicate
that only `context_memory#0` changed at this boundary.

The two complete boundary snapshots must be distinct, non-aliased captures on
opposite sides of the replacement. Comparing the source tree with itself, the
post-replacement tree with itself, deriving both left/right records from one
capture, or mutating a shared object after the first capture is an immediate
failure. `source_replacement_differ`, every `tree_equal`, every `values_equal`,
and the tick's `passed` must equal their recomputed predicates. Each
perturbation's `passed` must equal the conjunction over its exact required
streams and ticks, and `query_only_latent_no_read.passed` must include both
perturbation predicates together with all other required child predicates;
embedded pass/equality booleans are never authoritative.

`non_s_k_state.paths` is exactly this sorted seven-leaf list:

```text
ff_syn/post/V#0
ff_syn/syn/g#0
memory_read#0
query_encoding#0
reasoning_query#0
rec_syn/syn/g#0
workspace_carrier#0
```

Every entry is leaf index zero and has the `<f4` dtype and batch-512 shape
frozen in the H0 table above. `framing` is exactly
`nul_joined_gate_c2_non_s_k_state_v1`. For each side the qualifier constructs a
tree-geometry field list beginning with
`UTF8("example21-gate-c2-non-s-k-tree-v1")`, then, for every sorted path,
appends UTF-8 path, ASCII leaf index, ASCII `numpy.dtype.str`, and ASCII
comma-joined logical shape. `tree_sha256` is SHA-256 of those fields joined by a
single NUL byte. The value field list instead begins with
`UTF8("example21-gate-c2-non-s-k-state-v1")` and appends the same four fields
plus the leaf's contiguous C-order bytes before the next path; `value_sha256` is
the SHA-256 of that NUL-joined list. Missing, additional, reordered, or aliased
paths and any leaf, dtype, shape, or byte difference therefore fail closed.

`parameters.paths` is exactly the authenticated sorted canonical-full tree:
`color_factor_head/weight`, `ff_syn/comm/weight`, `height_head/weight`,
`memory_read_projection/weight`, `memory_write_scale`,
`readout_projection/weight`, `rec_syn/comm/weight`, `width_head/weight`, and
`workspace_query_projection/weight`. `framing` is exactly
`authenticated_gate_c_parameter_array_digest_v1`. Its tree digest uses the
same sorted path/leaf-index/dtype/shape geometry framing with domain
`example21-gate-c2-parameter-tree-v1`. Its value digest is exactly the existing
authenticated Gate C parameter framing: initialize SHA-256, then for every
sorted complete parameter path update it, without separators, with UTF-8 path
and each leaf's ASCII `numpy.dtype.str`, ASCII Python tuple shape, and contiguous
C-order bytes. Both left and right value digests must equal the corresponding
`gate_c_init.initialization.<regime>.canonical_full.parameter_sha256`,
not merely each other. A subset-tree digest cannot pass either record.

The perturbed tick's `selected_read` and `selected_drive` use the raw zero-array
schema and must still be exact zero. `continuation` has exactly `compact`,
`hidden_paths`, `predictions`, and `passed` and reuses the independently
recomputable comparison schemas above between the unperturbed and perturbed
query-only continuations. Its hidden set is the frozen H0 set above excluding
the deliberately replaced `context_memory#0`. Every retained float comparison
must independently meet `1e-6`, every discrete comparison must have Hamming
distance zero, and decoded predictions must remain exact. Both `+7` and `-7`
probes must pass every required stream and latent tick.

`full_positive_control` has exactly `plus_7`, `minus_7`, and `passed`. Each
replacement value has exactly `replacement`, `streams`, and `passed`, and its
stream and tick keys match the query-only probe. A full-policy tick has exactly
`source_s_k_sha256`, `replacement_s_k_sha256`, `source_replacement_differ`,
`non_s_k_state`, `parameters`, `selected_read_difference`,
`selected_drive_difference`, `continuation`, and `passed`.
`selected_read_difference` and `selected_drive_difference` use the floating
difference-record schema above; the other records reuse the exact perturbation
schemas. All selected-value records must be finite. For each replacement and
each regime, at least one retained stream/tick must have
`max_abs_difference > 0` in its selected read or drive; this proves that the
perturbation probe reaches a live contextual path without requiring every
individual query vector to respond. The continuation and nested prediction
records are retained, but the positive control does not preregister that a
decoded class must change.

Gate C2 control execution is bounded without weakening that evidence. For each
regime and stream, the producer runs one complete query-only latent baseline
and one complete full-policy latent baseline, and captures independent,
host-resident boundary snapshots for H0 through the last pre-H8 state from
those rollouts. These snapshots must not share mutable leaf storage. Restoring
and replaying each boundary against its baseline slice is a required reduced
structural test, not an additional production call. In production, every copied
boundary must instead be byte- and digest-identical to the corresponding
pre-tick state returned by the one baseline rollout. Changing one captured
boundary must leave every other boundary byte-unchanged. The producer may not
rerun an H0-to-Hn prefix merely to obtain a boundary it already captured.

The two cached-read sentinels still execute every required suffix from their
named Hn through H8 because those continuation records cover the full suffix.
Each query-only or full-policy `S_K` replacement executes only its named tick,
because its retained continuation record covers only that tick. For the exact
Gate B geometry across all three streams this gives 99 query-only latent calls
(three streams times the sum of one length-eight baseline, 16 cached-read
suffixes, and 16 one-tick `S_K` interventions), 51 full-policy latent calls
(three streams times the sum of one length-eight baseline and 16 one-tick
interventions), and six H0-prefix calls, for 156 model-driver calls total within
the Gate B `query_only_latent_no_read` subreport. Paired-H0, removed-path,
mechanism-oracle, Gate A, and other Gate C2 drivers are outside this counter.
Any additional call inside this named Gate B subreport fails the bounded-
workload test.

Only a `gate_c2_controls` result uses compact streamed JSON. Its writer uses a
strict `json.JSONEncoder` with `allow_nan=false`, `sort_keys=true`, and compact
separators, emits UTF-8 chunks directly to the temporary file, appends one
newline, flushes and fsyncs, then atomically replaces the final path. The exact
maximum encoded size is 201,326,592 bytes (192 MiB), including the final
newline. Crossing that limit deletes the temporary file and fails before
replacement. Gate C v1, Gate C initialization, and formal Gate C retain their
existing indented byte format. The compact result must strict-parse to the same
JSON value that the qualifier recomputes; formatting never changes evidence.

`removed_path_finite_window_influence` has exact keys `gradient_chunk_size`,
`start_state`, `objectives`, `global`, `live_paths`, `removed_paths`, and
`complete`. It uses `gradient_chunk_size=1` and
`start_state="materialized_h0_stop_gradient"`, so the common ordinary-query H0
contribution is not mistaken for a removed latent-read contribution. The Gate A
record's `objectives` has only `gate_a_h1`: intact canonical Gate A validation,
episode index zero, batch one, checkpoint H1, and the frozen classification
cross-entropy. The Gate B record's `objectives` has only
`gate_b_index0_r8_h8`: the already pinned intact validation episode index zero,
batch one, effort R8, checkpoint H8, and the frozen query-only canonical
weighting. The H0 prefix is executed once and materialized; it is outside the
gradient window. Gate A then executes one H1 tick. Gate B executes H1 through
H8 with chunk size one, but only H8 is selected into this removed-path
objective. It does not reuse the terminal-only unit weight and does not start at
H7.

The finite-window helper must lower its repeated chunk execution through one
`brainstate.transform.scan` (or an equivalent BrainState state-carrying loop),
not a bare Python `for` or `while` loop. Each scan iteration evaluates exactly
one chunk and adds that chunk's parameter gradient to an explicit fixed-shape
carry while BrainState carries the hidden and eligibility-trace states. This is
still the finite-window `chunked_online_param_gradients` path: combining the
whole suffix into one multi-step VJP is forbidden because that would erase the
chunk-boundary trace test. This compiled scan and its post-initialization
callback are an explicit Gate C2 mode of the helper used by only these two
removed-path objectives. The retained Gate C v1 execution path and the helper's
legacy default mode remain unchanged.

Both objective records retain raw inputs rather than only coordinate labels.
Every array digest below is the project framing
`SHA256(ASCII(dtype.str) || ASCII(str(shape)) || contiguous_C_bytes)`.
`source_contract` has exactly `metadata`, `events`, `advances`, `targets`,
`canonical_loss_weights`, `h0_prefix`, and `schedule_cross_bound`.
`continuation` has exactly `source_indices`, `source_events`, `batched_events`,
`advances`, `targets`, `selection_mask`, `base_checkpoint_weights`,
`effective_loss_weights`, `packed_inputs`, `h0_gradient_boundary`,
`source_slice_exact`, and `passed`. An event or
packed-input record has exactly
`dtype`, `shape`, and `sha256`; an exact-zero event record additionally has
`fill_value=0.0`. Advance, target, mask, and weight records have exactly
`dtype`, `shape`, `sha256`, and `values`. `h0_prefix` additionally has
`source_indices`. The qualifier regenerates the canonical schedule from the
authenticated regime configuration, extracts the named episode, and compares
every dtype, shape, value, and byte-framed digest before gradients run.

`gate_a_h1` has exactly `regime`, `stream`, `validation_episode_index`,
`batch_size`, `checkpoint`, `source_contract`, `continuation`,
`raw_cross_entropy`, `base_checkpoint_weight`, `weighted_cross_entropy`, and
`passed`, with fixed coordinates `gate_a`, `intact`, zero, one, and `H1`.
Its `source_contract.metadata` is exactly:

```json
{
  "mapping_id": 850050,
  "input_colors": [2, 5, 7, 8],
  "output_colors": [6, 5, 3, 8],
  "presentation_order_indices": [0, 2, 1, 3],
  "query_index": 3,
  "query_color": 8,
  "target": 8,
  "demonstration_indices": [0, 1, 2, 3],
  "h0_index": 4,
  "h1_index": 5
}
```

The Gate A source arrays and H0 prefix are pinned as follows:

```text
events                  <f4 [6, 41]  213fa1ede3635169cba47db69ad36cfab86e759e6f7e35e02e2d07687f71d36b
advances                |b1 [6]      42817343a401805d2af9b07c45738f71274aab865d20efb8fb1980e1ed7dc450
targets                 <i4 [1]      88c7413927e162658f4518fd4a62598fe8b0ea6e2ba5fa334940fdfc49ac845a
canonical_loss_weights  <f4 [6]      a13746d7d9b7bc9b071cfccfc55a0e8c54ad8454f4bcbf808f5235334c5a6c45
h0_prefix indices 0:5   <f4 [5, 41]  401cb20096483b305ef7e4383f03377b354fe167aaeffbd06df03f8251b021b2
```

The literal advances are six `true` values, the target values are `[8]`, and
the canonical source weights are `[0, 0, 0, 0, 0.5, 0.5]`. The source contract
uses `h0_prefix.source_indices=[0, 1, 2, 3, 4]`. It
must also match Gate A's pinned validation-schedule digest
`80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd`
and validation-mapping-ID digest
`a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3`.

Gate A continuation `source_indices=[5]`. Its exact-zero `source_events` is
`<f4 [1, 41]` with digest
`e9c01c22b9b1bfa0f9bc74cde1820fab5ad99037f581e27df1625db565d6c239`;
`batched_events` is `<f4 [1, 1, 41]` with digest
`7e0242875d49aef8e5b0c716cd7993f29895e26cf1dcc67cb2f198c1c351f5df`.
`advances` is `|b1 [1]`, values `[true]`, digest
`265e6b573637524100e6222332a3c4a92ba0cba78532eae2701915ac823cc05c`;
`targets` is `<i4 [1]`, values `[8]`, digest
`88c7413927e162658f4518fd4a62598fe8b0ea6e2ba5fa334940fdfc49ac845a`;
and `selection_mask` is the same `|b1 [1]` `[true]` record. Both
`base_checkpoint_weights` and `effective_loss_weights` are `<f4 [1]`, values
`[0.5]`, digest
`355fb8c16f46b517379b869a44896dcc28ef8344457510a7fb6533a0ce7ed8d9`.
The exact packed event/float-advance/float-target/float-weight input is
`<f4 [1, 1, 44]` with digest
`aee2f5f2f2a672f091c4d02e24ade55262e8ebe50ddd32317c4ead5f8e5b84c5`.

`gate_b_index0_r8_h8` has exactly `regime`, `stream`,
`validation_episode_index`, `batch_size`, `effort`, `checkpoint`,
`source_contract`, `continuation`, `raw_cross_entropy`,
`base_checkpoint_weight`, `weighted_cross_entropy`, and `passed`, with fixed
coordinates `gate_b`, `intact`, zero, one, eight, and `H8`. Its source metadata
is exactly:

```json
{
  "mapping_id": 232423,
  "mapping": [6, 7, 5, 2, 0, 4, 8, 9, 1, 3],
  "query_color": 4,
  "presentation_order": [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
  "shuffled_shift": 1,
  "h0_through_h8_targets": [0, 6, 8, 1, 7, 9, 3, 2, 5]
}
```

The source arrays are:

```text
events                  <f4 [19, 47] 36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec
advances                |b1 [19]     c45890e2f9f99fa66ffa09db8f685dc4d138c5f0e1ca0346a044f2dbbf1290a9
targets                 <i4 [19]     c4af41cac4f5eb682df15e7d6cf92b0c134b943fae1abfe99b0bfc4c2ddb27e0
canonical_loss_weights  <f8 [19]     205496cf3f437986dc5b65bec81d423848179306a7ce9e1a391dcc22c7340197
h0_prefix indices 0:11  <f4 [11, 47] a445ffd2a62e56808e15b6205cb7825fef7ed9a63a78b74d3942d89d7b6409a8
```

The literal source advances are 19 `true` values. Source targets are ten
demonstration zeros followed by `[0, 6, 8, 1, 7, 9, 3, 2, 5]`. Canonical R8
weights are ten zeros followed by nine float64 `1/9` values.
`h0_prefix.source_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. The source
contract must also match every pinned Gate B validation-field digest in the
Gate B schedule section above, including `mapping_ids`, `query_colors`,
`presentation_orders`, `targets_by_depth`, `advance_masks`, and `intact`.

Gate B continuation `source_indices=[11, 12, 13, 14, 15, 16, 17, 18]`. Its
exact-zero `source_events` is `<f4 [8, 47]` with digest
`87460e7b0e6ea0b632c89c84afa56b7c85c759bc8828b02e275fe4ac3a6be57a`;
`batched_events` is `<f4 [8, 1, 47]` with digest
`3d2da82783d3194730d1a4671d06df1254ef298fbe2260ee5e7c86474e111a32`.
`advances` is `|b1 [8]`, eight `true` values, digest
`8a707039840658f227e7fe98005429cf19641eb26bf528a80e5f08512099d6ad`;
`targets` is `<i4 [8]`, values `[6, 8, 1, 7, 9, 3, 2, 5]`, digest
`be444afd7597cbf6dd40160bf9f1341387db6bd7c7a5ac521e44ab76fa06b590`;
and `selection_mask` is `|b1 [8]`, values
`[false, false, false, false, false, false, false, true]`, digest
`d47a78c1c74031880ca53a15b96001d509253ea84d3d0b05bf24bbdd2f0846c0`.
`base_checkpoint_weights` is `<f4 [8]`, eight float32 `1/9` values, digest
`8888babf4975234431878a07158aff6fe97254086106dc5d25d4207ff509ac45`;
`effective_loss_weights` is `<f4 [8]`, seven zeros followed by float32 `1/9`,
digest `b15eb5519fa66d2d01c020c1c3f5f93a62a6b015bcfa7e1c171e71770a729b61`.
The exact packed input is `<f4 [8, 1, 50]` with digest
`5061ac4deaaf0e6bc153f0766aac8ba630d9d29c7a235a0998fc8121072eb910`.

For each objective, `base_checkpoint_weight` is the selected scalar array
record (`<f4 [1] [0.5]` with Gate A digest `355fb8...`, and
`<f4 [1] [float32(1/9)]` with Gate B digest
`c587060a4599f096433183ee7bc88de3234021291f1815e332950b80025d93b7`).
`raw_cross_entropy` and `weighted_cross_entropy` each have exactly `dtype`,
`shape`, `value`, `sha256`, `finite`, and `nonzero`, with dtype `<f4` and shape
`[1]`; the latter must recompute as the raw cross-entropy multiplied by the base
checkpoint weight. Both must be finite and strictly nonzero.
At the selected checkpoint the oracle wrapper emits
`sqrt(base_checkpoint_weight) * sqrt(raw_cross_entropy)`; at every unselected
checkpoint it emits exact zero. The helper's sum of squares is therefore exactly
the retained `weighted_cross_entropy`.
`h0_gradient_boundary` has exactly `capture_point`, `capture_count`,
`materialized_prefix`, `actual_gradient_start`,
`canonical_parameter_sha256`, `all_hidden_leaves_equal`, and `passed`.
`capture_point` is exactly
`after_init_etrace_state_before_first_gradient_chunk`, and `capture_count=1`.
The finite-window helper invokes one audited callback at that point: after
`brainstate.nn.init_all_states`, graph compilation, and eligibility-trace
initialization, but before its first gradient chunk. The callback restores the
stopped H0 snapshot and captures the actual live state immediately. A later
restore, a digest inferred after gradients, or an `init_state`/`reset_state`
method that is not exercised at this boundary cannot supply this evidence.

`materialized_prefix` and `actual_gradient_start` each have exactly
`hidden_paths`, `hidden_state_tree_sha256`, and `parameter_sha256`.
`hidden_paths` has these exact batch-one leaf names, dtypes, and shapes:

```text
context_memory#0     <f4 [1, 32, 32]
ff_syn/post/V#0      <f4 [1, 2048]
ff_syn/syn/g#0       <f4 [1, 2048]
memory_read#0        <f4 [1, 32]
query_encoding#0     <f4 [1, 32]
reasoning_query#0    <f4 [1, 32]
rec_syn/syn/g#0      <f4 [1, 2048]
workspace_carrier#0  <f4 [1, 2048]
```

Every leaf value has exactly `dtype`, `shape`, `data_hex`, `sha256`,
`finite_count`, and `nonfinite_count`. `data_hex` is the lowercase hexadecimal
encoding of the leaf's contiguous C-order mantissa bytes and has exactly twice
the byte length implied by its dtype and shape. The qualifier decodes those
bytes, reconstructs the array, and recomputes `sha256` with the project
dtype-and-shape framing. Both count fields must be strict integers. The
qualifier recomputes them from the decoded bytes, requires their sum to equal
the shape product, requires `finite_count` to equal that product, and requires
`nonfinite_count=0`.

The two tree digests use the existing
`example21-gate-c-hidden-state-v1` framing. For each retained key, the qualifier
splits at the final `#`, parses the suffix as a nonnegative decimal leaf index,
and sorts by `(base_path, leaf_index)`. It then appends the UTF-8 base path,
ASCII leaf index, ASCII dtype, ASCII comma-joined shape, and decoded contiguous
bytes as separate NUL-delimited fields. Hashing the literal `path#0` key as the
path field, accepting a nondecimal suffix, or omitting the separate leaf-index
field fails closed. This makes both tree digests independently recomputable
from the retained artifact.

The producer captures `materialized_prefix` directly after the authenticated
query-only model executes the pinned batch-one H0 prefix and before
stop-gradient. It captures `actual_gradient_start` only through the audited
callback above. Both snapshot parameter digests and
`canonical_parameter_sha256` must equal the regime's authenticated
`gate_c_init.initialization.<regime>.canonical_full.parameter_sha256`.
`all_hidden_leaves_equal` is recomputed from exact path, dtype, shape, byte,
leaf-digest, and tree-digest equality between the two retained snapshots.
`passed` is the conjunction of the exact capture point/count, canonical
parameter equality, complete finite snapshot schemas, and exact hidden-leaf
equality. The authenticated producer and pinned raw H0 prefix are the replay
boundary; host requalification does not regenerate GPU parameters from a seed
or silently substitute a CPU initialization. The separately compiled batch-512
operational-H0 probe remains an independent control and is not required to be
byte-identical to this batch-one evidence. `source_slice_exact`,
`schedule_cross_bound`, and every enclosing `passed` value are recomputed rather
than trusted.

For each objective, `global` has exactly `tree_paths`, `leaf_count`,
`value_count`, `l2_norm`, `sha256`, `finite`, and `nonzero`; it must cover the
exact authenticated query-only parameter tree and be finite and strictly
nonzero. `live_paths` has exactly `color_factor_head/weight`,
`readout_projection/weight`, and `rec_syn/comm/weight`. The external event on
the latent-only objective is exact zero, so `ff_syn/comm/weight` is deliberately
not a nonzero witness. Each live-path value has exactly `tree_paths`,
`leaf_count`, `leaves`, `value_count`, `l2_norm`, `sha256`, `finite`, and
`nonzero`; each must be finite and strictly nonzero. Its leaf records use the
same exact index, dtype, shape, count, finiteness, zero-count, and digest schema
as the removed paths. The qualifier must also sum every live leaf's
`zero_count`, require that sum to be strictly less than the retained total
`value_count`, and reject a claimed positive aggregate whose leaf inventory is
all zero. This makes a zero removed-path result non-vacuous.

`removed_paths` has exactly `memory_read_projection/weight` and
`workspace_query_projection/weight`. Each value has exactly `tree_paths`,
`leaf_count`, `leaves`, `value_count`, `l2_norm`, `sha256`, `exact_zero`, and
`finite`. Each `leaves` entry has exactly `index`, `dtype`, `shape`,
`value_count`, `finite_count`, `nonfinite_count`, `zero_count`, and `sha256`.
`memory_read_projection/weight` is the one-leaf tree at index zero with dtype
`<f4`, shape `[32, 2048]`, and 65,536 values;
`workspace_query_projection/weight` is the one-leaf tree at index zero with
dtype `<f4`, shape `[2048, 32]`, and 65,536 values. The qualifier independently
constructs these exact all-zero leaves and recomputes the leaf and formal
gradient-path digests from the frozen tree/path framing. It requires every
value finite and exact zero, both counts exact, `l2_norm=0`, and the reported
digest equal to the reconstructed zero digest. These selected-zero,
two-perturbation, positive-control, no-cache, and non-vacuous finite-window
results jointly define causal no-read semantics even when BrainTrace
if-conversion executes discarded arithmetic.

#### Unchanged Gate C2 mechanism oracle and stop rule

Gate C2 retains the exact finite-window oracle contract and comparison paths
above. Globally and for each required path, the thresholds remain:

```text
relative_deviation >= 1e-3
L2 difference > max(1e-8, 1e-4 * full_gradient_norm)
```

The inequality remains strict. In particular, the Gate C v1 observed
`workspace_query_projection/weight` difference `2.5744135613e-9` remains below
the unchanged `1e-8` floor and remains a failure; it does not justify lowering
the threshold for Gate C2. Any future threshold revision requires an
independently justified and preregistered protocol with a new target, control,
schema version, artifact path, and interpretation before its result is
observed. It cannot be applied to Gate C v1 or Gate C2 in place.

Gate C2 passes only if all 15 criteria recompute true in one authenticated
bundle. Any schema, prerequisite, initialization, consumed-weight, pretraining
control, H0 operational-equivalence, training, behavioral, oracle, source, or
GPU failure records the Gate C2 failing interpretation and stops. Until Gate C2
passes, Gate D remains stopped, no new ARC test is run, and the retained
exact-ARC baseline remains zero.

### Post-run authenticated Gate C2 controls result: FAIL

This is a post-run result record. It does not change the preregistered Gate C2
contract above. The `gate_c2_controls` admission ran at clean source HEAD
`555c8ee35bc349a618b3d1434240ed4f385ca564` on immutable GPU image
`sha256:54c96a66cd849673f61239f7d2cb8861f488f42ceda711dc2546ea074c08e3a0`.
The image's OCI revision was the same full source HEAD.

The authenticated controls bundle is:

| File | Repository-relative path | Bytes | SHA-256 |
|---|---|---:|---|
| result | `var/example21-causal-gate/555c8ee35bc349a618b3d1434240ed4f385ca564-gate-c2-controls.json` | 77,270,720 | `6a3ed8b9ed73792fe68d5f7a958e82850600ca581c9d71ac0b7c19aa30d36938` |
| preflight | `var/example21-causal-gate/555c8ee35bc349a618b3d1434240ed4f385ca564-gate-c2-controls.preflight.json` | 76,065 | `914eb6c5c9b4cbafd61c7446c7846efc5d7ac1ee617b2b8ab7a499b9057f19db` |
| manifest | `var/example21-causal-gate/555c8ee35bc349a618b3d1434240ed4f385ca564-gate-c2-controls.manifest.json` | 17,364 | `facb15f21c55bfe951b9448969151c12aa26a62053e0505533e8341a9f9b91e8` |

The authenticated bundle SHA-256 is
`f27d895eb42f129af3071458d563cbd738210ab00202918998544a56d1029972`.
The result is `73.691101` MiB, below the exact 192 MiB limit. It records
`311.55114891100675` seconds of internal work; the retained child-command wall
time is `335.5185390999977` seconds. The child returned zero after writing the
complete negative result. The manifest has `bundle_valid=true`,
`process_succeeded=true`, `artifact_schema_verified=true`,
`scientific_qualification_passed=false`, and `failure=null`. This is therefore
an authenticated scientific failure, not a crashed or unauthenticated run.

The fresh same-HEAD, same-image `gate_c_init` prerequisite passed. Its result,
preflight, manifest, and bundle SHA-256 values are, respectively:

```text
6bdad647ffb1bbbe7a58d246659f94cfb2e022bc9d6839e10cc786ab020c47de
17d66c160671b95e1c4cb189075f6aa72b6e4693998eaad37572a5080848ca42
e4688e991fd7164923a65427d18796cebc2f0b95b816dbecafce2e11b59d3c3e
c2ae5640fe4c3323040c576a0471c4ef1c8c643f4353b27844065d4603f013cc
```

The controls qualifier recomputed seven of nine criteria as true:

| Criterion | Recomputed result |
|---|---|
| `schema_and_control` | true |
| `exact_configuration` | true |
| `prerequisites_authenticated` | true |
| `initialization_authenticated` | true |
| `canonical_schedules_complete` | true |
| `no_behavioral_or_optimizer_updates` | true |
| `paired_h0_operational_equivalence` | **false** |
| `mechanism_oracle_complete` | **false** |
| `source_and_gpu_authenticated` | true |

The recomputed interpretation is exactly
`gate_c2_pretraining_controls_failed_stop`. Gate A passed its paired-H0 and
no-read controls. Gate B failed the shuffled same-full H0 replay: its maximum
per-example RMS difference was `1.5919210544258758e-6`, above the `1e-6`
limit. The aggregate RMS was `7.035363579363599e-8`, the maximum absolute
difference was `5.816575139760971e-6`, exactly one of 512 examples exceeded
the limit (example 505), and prediction Hamming distance remained zero. This is
a same-full operational-reproducibility miss, not evidence of a query-only
treatment effect.

The Gate B no-read failures were sparse but still exceeded the frozen
per-example tolerance. All 48 selected-read and selected-drive records were
exact zero, 18 of 24 cached-read probes passed, 39 of 48 `+11`/`-11` sentinel
probes passed, 18 of 24 query-only `-7` probes passed, and 17 of 24 query-only
`+7` probes passed. The six failed cached-read starts were intact H2, H3, H4,
H5, and H8, plus shuffled H8. Decoded predictions remained unchanged in the
cached-read probes and the query-only `+7` and `-7` probes. This statement does
not cover the full-policy positive controls. Both full-policy positive controls
were live at all 24 stream/tick locations for each sign. The opposite-sign
residual pattern and the independent same-full miss are consistent with
numerical replay instability; they do not establish cached-read or `S_K`
influence. The frozen operational criterion still fails.

The removed-path finite-window evidence itself passed in both regimes. For
both Gate A and Gate B, all 65,536 values on each of
`memory_read_projection/weight` and
`workspace_query_projection/weight` were finite and exact zero. All three live
witness paths -- `color_factor_head/weight`,
`readout_projection/weight`, and `rec_syn/comm/weight` -- were finite and
nonzero. The global gradient norms were `7.630332531466312e-5` for Gate A and
`2.0280032487405228e-5` for Gate B. Thus the stopped-H0 query-only test
successfully proves the removed paths dead while retaining live control paths;
it does not prove that the corresponding full-policy gradient meets the
separate mechanism floor.

The mechanism oracle failed that unchanged floor. The query-only global
comparison passed with relative deviation `0.08202308213429639` and L2
difference `1.994311067736179e-6`. Its required
`memory_read_projection/weight` path also passed, with relative deviation
`0.9906690906447477` and L2 difference `2.3476146125229822e-8`. The required
`workspace_query_projection/weight` path had relative deviation `1.0`, but its
L2 difference was only `2.574413770975678e-9`, below the strict `1e-8` floor.
The terminal-only global comparison also passed, with relative deviation
`7.57758473168369`. The failed workspace-query floor therefore independently
makes `mechanism_oracle_complete=false`.

The no-update audit was complete: trainer-factory calls, training-step calls,
optimizer constructors, optimizer instances, and optimizer updates were all
zero. It registered 18 distinct model roles. Every role retained
`before_parameter_sha256 == after_parameter_sha256 ==
expected_parameter_sha256`. No behavioral training occurred.

Post-run validation also found a serialization-order defect. The compact JSON
writer sorts object keys, while the no-read validator compares insertion order
through `tuple(mapping)`. The reloaded stream order is `intact`, `no_context`,
`shuffled`, but the validator expects `intact`, `shuffled`, `no_context`; the
reloaded perturbation order is `minus_7`, `plus_7`, but the validator expects
`plus_7`, `minus_7`. Both reloaded no-read validators therefore return false
before their numeric checks, including the otherwise passing raw Gate A
record. This is an implementation defect, but it does not change the stop
decision: Gate B independently fails same-full H0 replay and raw no-read
tolerances, and the unchanged mechanism oracle independently fails its
workspace-query absolute floor.

The narrow post-run implementation correction is that JSON object member order
is non-semantic. The compact writer deliberately sorts object keys, so every
validator must require the exact expected key set and validate each value by
its named key; no validator may use `tuple(mapping)` or another insertion-order
comparison as an object-schema condition. Ordered JSON arrays and other
sequence values retain their exact order requirements. Strict parsing must
continue to reject duplicate object keys. This correction does not
retroactively qualify the failed `555c8ee35bc349a618b3d1434240ed4f385ca564`
artifact, change any frozen threshold, permit a rerun of that admission, or
unlock formal Gate C2, Gate D, or ARC.

The pretraining admission did not pass, so no `formal_gate_c2` target or ten-
model training run was enabled or executed. Gate D and ARC remain stopped. No
new ARC test ran, the retained exact-ARC score remains `0`, and this result
supports no causal-mechanism conclusion.

### Gate C3: deterministic replay and terminal-H8 mechanism amendment

This is a new adaptive preregistration written after the authenticated Gate C2
controls result was observed. It does not edit, rerun, supersede, or
reinterpret either failed Gate C result. In particular, the retained Gate C2
bundle, its seven-of-nine outcome, its two failed criteria, and its stop
decision remain immutable. Gate C3 repairs two independently diagnosed control
defects: the launcher did not bind the deterministic GPU environment used by
the successful replay diagnostic, and the all-depth mechanism objective
diluted the workspace-query path because `H_0` cannot depend on that path and
earlier depths can cancel later contributions. Gate C3 changes no architecture,
parameter count, initialization algorithm, data split, episode, model seed,
training seed, optimizer, arm, training budget, behavioral threshold, H0
tolerance, no-read threshold, removed-path rule, or mechanism threshold.

Gate C3 has two new identities. The pretraining admission has launcher and
child target `gate_c3_controls`, result path
`var/example21-causal-gate/<head>-gate-c3-controls.json` with matching
`.preflight.json` and `.manifest.json` sidecars, `schema_version=1`,
`control="example21_gate_c3_pretraining_control_admission"`,
`qualification_regime="preregistered_gate_c3_pretraining_controls"`, and
`learner="pp_prop_only"`. Its passing and failing interpretations are exactly
`gate_c3_pretraining_controls_passed` and
`gate_c3_pretraining_controls_failed_stop`; invalid evidence instead uses
`gate_c3_pretraining_controls_invalid_stop`. The later formal experiment has
launcher and child target `formal_gate_c3`, result path
`var/example21-causal-gate/<head>-formal-gate-c3.json` with matching sidecars,
`schema_version=3`,
`control="example21_pp_prop_learnability_gate_c3"`,
`qualification_regime="preregistered_gate_c3_full"`, and
`learner="pp_prop_only"`. Its passing and failing interpretations are exactly
`gate_c3_passed_pp_prop_learnability_mechanism` and
`gate_c3_failed_stop_no_causal_mechanism_conclusion`.

Each controls or formal bundle must run at its own exact clean source HEAD in a
newly built immutable GPU image whose OCI revision is that exact HEAD. Before
either target, a fresh schema-1 `gate_c_init` must pass at that same HEAD and
image; no earlier initialization admission is reusable. The historical
`gate_c_init` target, fixed argv, schema, and identity remain unchanged. It only
materializes and authenticates initial parameter trees, compiler topology, and
zero optimizer
state; it does not authenticate model hidden-state bytes. It is not required
to execute H0 evidence or use the C3 deterministic environment. The controls
admission and formal target must authenticate that result, preflight, manifest,
bundle formula, source HEAD, and image before work and again before signing.
Every audited C3 model role must receive an exact copy of the authenticated
canonical parameters. The C3 controls independently materialize fresh hidden
states, require exact geometry and byte equality across the paired roles before
execution, and retain those runtime snapshot digests in the C3 artifact. All
model random values must be produced through `brainstate.random`; direct
`jax.random` use is not admissible. The controls admission performs no training,
optimizer construction, optimizer update, parameter mutation, or warm start.
If enabled after controls pass, all ten formal arms start from the newly
authenticated canonical parameters and fresh zero-valued Adam states, never
from a Gate C or Gate C2 trained state.

#### Authenticated deterministic execution

The launcher must set these exact environment-variable values before starting
Python in the container:

```text
XLA_FLAGS=--xla_gpu_deterministic_ops=true
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The fixed launcher command and preflight retain both names and exact values.
Docker applies them before the child Python process starts, so they are active
before JAX or BrainState imports. The result independently records the two live
values, and the manifest authenticates and revalidates both the preflight and
result records before signing. A missing value, a different value, or an
additional token appended to `XLA_FLAGS` fails authentication.
`gate_c3_controls` and `formal_gate_c3` use the same values; CPU, dense, or
`jax_raw` fallback is not admissible. No tolerance miss, gradient miss, output
value, or other observed result may trigger another compilation, replay, seed,
episode, stream, or batch selection.

In this controls-only stage, the two variables apply only to the
`gate_c3_controls` container. The historical `gate_c_init` target and every
legacy target retain their exact prior environment contract. The C3 result
retains the C2 controls top-level key set exactly:

```text
schema_version, control, qualification_regime, learner, prerequisites,
regimes, mechanism_oracle, source_start, source_end, source_files,
environment, qualification, total_wall_seconds
```

`prerequisites` has exactly `gate_a`, `gate_b`, and
`gate_c_initialization`. `environment` has exactly the six C2 keys `backend`,
`devices`, `image_digest`, `jax`, `python`, and
`execution_and_update_evidence`, plus `deterministic_environment`. That last
mapping has exactly these values:

```json
{
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
  "XLA_FLAGS": "--xla_gpu_deterministic_ops=true"
}
```

#### Gate C3 operational and no-read controls

Gate C3 retains the complete Gate C2 operational-H0 record schemas, fixed
`canonical_production_sparse` backend, fixed boundary, hidden paths, endpoint
geometries, comparison roles, and `rms_tolerance=1e-6`. It retains all 512
examples for every Gate B H0 control and every `intact`, `shuffled`, and
`no_context` stream; subsampling an example, stream, hidden leaf, or comparison
is forbidden. Every compact output and hidden leaf must be finite, must have
the exact frozen dtype and shape, and must independently satisfy maximum
per-example RMS difference `<= 1e-6`. Decoded predictions must have exact zero
Hamming distance. Endpoint byte digests are retained for audit, but byte
identity after GPU execution is explicitly not a qualification requirement:
the deterministic diagnostic still exhibited numerical differences on the
order of `1e-8`. The tolerance is not rescaled from that observation.

The same-full replay, copied-full separate-compilation replay, and copied-full
versus query-only comparison all remain required. Query-only intervention still
begins only after the common ordinary-query H0 boundary. Gate A controls remain
part of the admission under their unchanged C2 contract. Every Gate C2
query-only no-read control, selected read/drive check, cached-read probe,
`+11`/`-11` sentinel, `+7`/`-7` query perturbation, full-policy positive
control, local boundary restore, driver-call bound, and exact threshold remains
blocking without modification. The stopped-H0 finite-window removed-path test
also remains unchanged: both removed projection paths must be finite exact-zero
trees, and all three frozen live witness paths must remain finite and nonzero.

#### Gate C3 terminal-H8 mechanism oracle

The new blocking oracle fixes one objective before observing a Gate C3 result.
It uses the canonical Gate B validation episode index zero, intact stream,
effort `R=8`, batch one, mapping ID `232423`, and float32 event digest
`36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec`.
It uses the authenticated fresh full initialization, checkpoint target `H_8`
at sequence index `18`, and a float32 length-19 loss-weight vector whose values
are exact zero except index 18, which is exactly `1.0`. Its total loss weight is
exactly `1.0`. Both the source vector and the tensor consumed by the gradient
helper's loss-weight column `packed_inputs[:, 0, -1]` must independently equal
this exact report and must be byte-identical to each other:

```json
{
  "dtype": "<f4",
  "shape": [19],
  "sha256": "07fecad3bfcbd816df57ab71c500db391cbf3b581a99376678d0e5f9da8e6693"
}
```

The digest uses the existing Gate C raw-array framing: dtype string, Python
shape string, and contiguous C-order bytes. The complete consumed targets are
int32 shape `[19]`, values
`[0,0,0,0,0,0,0,0,0,0,0,6,8,1,7,9,3,2,5]`, SHA-256
`c4af41cac4f5eb682df15e7d6cf92b0c134b943fae1abfe99b0bfc4c2ddb27e0`.
Advances are float32 shape `[19]`, all `1.0`, SHA-256
`d69cc2400af318c684ba7c8ba0d66204f25264b3bcbba9d8d96d999bdefc4a07`.
The packed float32 shape `[19,1,50]` helper argument has SHA-256
`ef1c75296133458d90de3d5d9c204890127f83238148bd11bc2736bae6a205e1`.
Any episode, mapping, event, target, index, dtype, shape, weight, or digest
mismatch stops before gradient evaluation.

The only blocking arm comparison is the matched `full_read_h8` versus
`query_only_h8` policy pair. Both arms start from deep copies of the same
authenticated parameters and the same freshly materialized, pre-execution
state snapshot. `query_only_h8` shares the full
ordinary-query H0 execution and then removes latent reads from H1 through H8.
Both use `chunked_online_param_gradients` with `chunk_size=1`; a whole-sequence
VJP is inadmissible. The C3 call sets `compiled_scan=True`, so the 19 repeated
finite-window steps run inside the helper's BrainState scan rather than a bare
Python model loop. This opt-in preserves the chunk-one pp-prop boundary while
leaving legacy callers unchanged. Each wrapper emits exact zero at indices 0
through 17 and emits `sqrt(classification_cross_entropy)` at index 18, so the
helper's sum-of-squares loss represents the terminal-H8 cross-entropy with unit
weight by construction; byte equality between the floating square and the raw
cross-entropy is not claimed. The earlier terminal-H8 norm was measured through
the legacy host-loop mode and motivates this fixed objective, but it does not
predict or qualify the C3 compiled-scan result.

Run exactly two technical replays. Each replay materializes fresh model and
state copies, separately constructs each model and policy driver, and
independently invokes the `compiled_scan=True` path under the authenticated
environment above. Compiler-cache reuse is allowed and is not evidence. Retain
canonical per-path and global
gradient digests, norms, L2 differences, relative deviations, cosines, finite
counts, dtype, shape, and leaf order using the existing Gate C digest framing.
Before execution, both replays must have byte-exact authenticated parameter and
hidden-state snapshot digests and exact episode, event, decoded-target, and
policy identities. Post-execution gradient-byte equality across separately
constructed and invoked sparse GPU paths is explicitly not required; retain
both hashes and raw records even when same-arm hashes differ. Each of the two
full-versus-query-only comparisons must independently pass every unchanged
blocking threshold:

```text
relative_deviation >= 1e-3
L2 difference > max(1e-8, 1e-4 * full_gradient_norm)
```

The global gradient and each of
`memory_read_projection/weight` and
`workspace_query_projection/weight` must have a finite nonzero full-arm norm
and independently pass both inequalities in each replay. The query-only norm
may be exact zero, with null cosine under the existing defined/null rules. A
pass in one replay cannot compensate for a failure in the other, and averaging
the replays is forbidden.

The old Gate C2 all-depth finite-window oracle remains only in its immutable
authenticated C2 bundle and in the post-run record above. It is not copied into
the C3 result, is not a C3 prerequisite or field, and is not executed again.
Its all-depth workspace-query result may not qualify or disqualify Gate C3, and
it may not be substituted for the terminal-H8 oracle. This change is an
explicit adaptive follow-up based on the diagnosed objective geometry, not a
retroactive change to either earlier gate.

The no-update audit has exactly 20 materialized model roles: the exact 12 Gate
C2 roles whose names do not contain `mechanism_oracle`, plus these eight C3
roles:

```text
gate_b:mechanism_oracle:terminal_h8:replay_1:full_read_h8:reference
gate_b:mechanism_oracle:terminal_h8:replay_1:full_read_h8:finite_window
gate_b:mechanism_oracle:terminal_h8:replay_1:query_only_h8:reference
gate_b:mechanism_oracle:terminal_h8:replay_1:query_only_h8:finite_window
gate_b:mechanism_oracle:terminal_h8:replay_2:full_read_h8:reference
gate_b:mechanism_oracle:terminal_h8:replay_2:full_read_h8:finite_window
gate_b:mechanism_oracle:terminal_h8:replay_2:query_only_h8:reference
gate_b:mechanism_oracle:terminal_h8:replay_2:query_only_h8:finite_window
```

Every role is constructed exactly once, starts from the expected canonical
parameter digest, finishes with that same digest, and creates no trainer or
optimizer. The retained model-factory and model-constructor call arrays are the
20 exact role names in lexicographic order, matching the existing C2 audit
serialization rule. Missing, duplicate, extra, or reordered audit call records
fail.

#### Qualification and stop semantics

The Gate C3 controls admission has exactly these 11 qualification criteria:

1. `schema_and_control`;
2. `exact_configuration`;
3. `prerequisites_authenticated`;
4. `initialization_authenticated`;
5. `deterministic_environment_authenticated`;
6. `canonical_schedules_complete`;
7. `no_behavioral_or_optimizer_updates`;
8. `paired_h0_operational_equivalence`;
9. `no_read_and_removed_path_complete`;
10. `mechanism_oracle_complete`; and
11. `source_and_gpu_authenticated`.

Its `qualification` object has exactly `valid`, `passed`, `criteria`,
`failures`, and `interpretation`. Embedded booleans are never trusted. The
qualifier recomputes all 11 criteria from retained evidence. Gradient arrays
are not serialized; their authenticated producer summaries retain exact
geometry, finite counts, hashes, norms, differences, cosines, and algebraic
cross-checks under the existing Gate C trust boundary. `valid` is true only
when the complete artifact has strict valid schema, finite evidence,
authenticated provenance, and a successful serialized reload; it can remain
true when a scientifically complete run fails one or more controls. `passed`
is true exactly when `valid` is true and all 11 recomputed criteria are true.
`failures` is the lexicographically sorted list of every recomputed false
criterion and is empty exactly on pass. `interpretation` is the invalid-stop
value when `valid=false`; otherwise it is selected from `passed`. An invalid or
incomplete artifact cannot be called a scientific pass or failure and cannot
admit formal training.

`valid=false` whenever schema/control, exact configuration, prerequisite,
initialization, deterministic-environment, schedule, no-update audit, or
source/GPU authentication is incomplete or false. A structurally complete and
authenticated H0, no-read, removed-path, or mechanism record that misses its
frozen scientific threshold remains `valid=true`, `passed=false`, and uses the
scientific failed-stop interpretation.

The compact result writer may sort JSON object keys. Before signing, the
producer must reload the exact serialized bytes with the strict duplicate-key,
NaN, and infinity rejection path and rerun complete qualification on that
reloaded object. JSON mapping order is non-semantic: validators require exact
key sets and resolve values by key, never by insertion order or
`tuple(mapping)`. Arrays and all other semantic sequences retain their frozen
order. Missing or extra mapping keys, duplicate keys, reordered semantic
arrays, reload disagreement, or a qualification object inconsistent with the
recomputation fails validation.

The `formal_gate_c3` target is deliberately absent from this controls-only
implementation stage. A passing controls artifact admits implementation of the
formal stage; it does not admit training from a different source identity.
Adding the formal target changes source HEAD and image, so that later source
must generate a fresh `gate_c_init` and must repeat the unchanged C3 controls
once at its own exact HEAD and image before constructing any model trainer.
This staged repetition is provenance reauthentication, not an outcome-triggered
retry: the first controls artifact remains immutable, its result cannot select
seeds, episodes, thresholds, or code paths, and a failed first controls result
stops without formal implementation. The later formal target then runs the ten
fresh C2-contract trainings with the C3 deterministic environment,
terminal-H8 blocking oracle, and same-HEAD `gate_c3_controls` prerequisite.
Formal Gate C3 otherwise retains the Gate C2 arms, regimes, configuration,
consumed training weights, behavioral margins, frozen-write characterization,
source authentication, and stop rules. A formal failure stops Gate D and ARC.
Until a complete formal Gate C3 bundle passes, no Gate D qualification or new
ARC test may run, the retained exact-ARC score remains `0`, and no causal
latent-reasoning conclusion is supported.

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
`dafa64a8b4c3848241baa117affa55b632518a8e`; and the initialization-only Gate C
admission passed at commit `c2eb27b4d51c07e4b68bd29d81101bbfff0351b8`.
The formal Gate C mechanism ablations then ran at commit
`59b27d7be5cc9c37845da7bb2c81ae7203935338` and failed as recorded above.
The post-failure Gate C2 controls implementation and a fresh `gate_c_init`
admission were completed at
`555c8ee35bc349a618b3d1434240ed4f385ca564`. The fresh initialization passed,
but the authenticated `gate_c2_controls` run failed as recorded above. No
`formal_gate_c2` target, ten-model training run, Gate D qualification, or new
ARC test was run. The retained exact-ARC score remains zero.

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
