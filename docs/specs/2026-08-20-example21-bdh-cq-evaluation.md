# Example 21 — critical evaluation against BDH-CQ and BrainScale

Status: review; no code or configuration changed by this document
Date: 2026-08-20
Branch: `worktree-example21-bdh-cq-evaluation`

Subjects of the evaluation:

- **BDH-CQ**: Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent
  Reasoning*, arXiv:2608.09888. The target contract.
- **BrainScale**: Wang, Dong, Jiang, Ji, Liu, Wu, *BrainScale: Enabling Scalable
  Online Learning in Spiking Neural Networks*, bioRxiv 2024.09.24.614728v2
  (`docs/2024.09.24.614728v2.full.pdf`). The substrate whose online-learning
  algorithm Example 21 runs on — `pp_prop` is BrainScale's ES-D-RTRL.
- **Example 21**: `examples/pp_prop/21-latent-reasoning-in-context.py` and the
  `latent_workspace_*` modules, plus the results recorded in
  `docs/specs/2026-08-19-example21-batched-training-results.md` §5 and the
  artifacts under `var/example21-shared-*`.

Everything below is measured from artifacts already on disk or read from source.
No new run was required.

---

## 1. What Example 21 claims, and the one axis its claim boundary is silent on

The emitted claim boundary is unusually disciplined and should be preserved:

> Example 21 instantiates the paper's public ARC task, ranked-candidate, and
> variable-effort contract with BrainPy LIF neurons, BrainTrace sparse synapses,
> and pp-prop. The paper's private data, model dimensions, internal update
> rules, and training recipe were unavailable. This is not a reproduction, makes
> no paper-score or inference-cost claim, and asserts no agreement between
> pp-prop and a BPTT gradient oracle.

It disclaims score, cost, data, dimensions, recipe, and gradient agreement. It
does **not** disclaim the property that defines BDH-CQ as a result: that the
reported accuracy is obtained **without any weight update at inference**. That
is precisely where Example 21's measured score comes from (§3). The claim
boundary needs one more clause.

## 2. Contract fidelity, item by item

BDH-CQ discloses only a system-level interface:

```text
S_t   = U_θ(S_{t-1}, D_t)      # demonstration ingestion into contextual memory
H_0   = E_θ(x*, S_K)           # query-conditioned workspace initialisation
H_r+1 = F_θ(H_r, S_K)          # recurrent latent reasoning, r = 0..R-1
ŷ     = G_θ(H_R)               # decode
```

| Contract element | Example 21 | Verdict |
|---|---|---|
| `S_t` separate contextual memory | `context_memory` HiddenState, `(batch, 32, 32)`, written only on gated demonstration rows | **met** |
| `S_K` frozen during query/latent phase | write gate requires `demonstration_phase`; digest asserted stable | **met** |
| `H_r` re-reads `S_K` each latent tick | `memory_read_policy="full"` path reads on query and latent ticks | **met** |
| Shared decoder at every depth | `answer_row_head` / `answer_shape_head` at every refinement tick | **met** |
| Ranked candidates, pass@2 | candidate 1 = latest sweep logits, candidate 2 = runner-up | **met** |
| Variable effort `R` | 14 checkpoints declared, 0…390 | **partially measured** (§6) |
| Ordinary ARC tasks, official split | 400 eval tasks / 419 queries, `split_overlap_check: passed` | **met** |
| **In-context only — no weight update at inference** | test-time gradient adaptation on each task's own folds supplies essentially all score | **not met** |
| `U_θ` learned | keys/values are **fixed** random Fourier / random-projection bases; only a 32×32 `memory_write_scale` is trainable | **weakened** |

The last row matters more than it looks. In BDH-CQ, `U_θ` is learned end to end —
*what to store* is part of the learning problem. In Example 21 the storage code
is frozen random features and the only learnable degree of freedom in the write
path is a 32×32 elementwise gain. The system can learn how much to weight a
random binding; it cannot learn a better binding.

## 3. The central finding: everything scoreable comes from the arm the paper does not run

Complete-split numbers, from `2026-08-19-example21-batched-training-results.md`
§5.1–5.2 (400 tasks, 419 official queries, model-only candidates, rule channel
off):

| arm | shape | pixel | strict task pass@2 |
|---|---:|---:|---:|
| frozen shared model — *in contract* | 0.0310 | 0.2235 | **0.0000** |
| task-local pp-prop adaptation — *out of contract* | 0.6014 | 0.4902 | **0.0025** (1/400) |

The in-contract arm — demonstrations enter only through `S_K`, parameters
frozen, exactly BDH-CQ's setting — solves **zero** of 400 tasks and produces the
right output *shape* on 3.1% of queries. Everything Example 21 can show on ARC
is produced by 13,630 gradient updates applied at test time (1,363 leave-one-out
folds × 10 epochs, lr 3e-3), with parameters cloned per task and restored
afterwards.

Three things follow, and they should be stated together:

1. **This is not leakage.** `assert_no_evaluation_leakage` holds, the official
   query target is never read, restoration after each task is verified, and
   `split_overlap_check` passes. The adaptation consumes only the demonstrations
   that BDH-CQ also consumes. It is legitimate ARC practice.
2. **It is a different problem setting.** It is test-time training. BDH-CQ's
   entire thesis is that recurrent latent reasoning in `S_K`/`H_r` *replaces*
   both chain-of-thought and per-task fitting. Placing 1/400-with-TTT next to
   118/400-without-TTT compares two settings, not two models. The correct
   comparison class for the adapted arm is the ARC test-time-training
   literature; the correct comparison for BDH-CQ is the frozen arm, which scores
   zero.
3. **The one solve is not a signal.** 1/400 strict pass@2 has a Wilson 95%
   interval of roughly [0.04%, 1.4%] against the paper's [25.2%, 34.2%]. It is a
   single seed (2108). The results document already says the count *ties* the
   prior integrated run rather than exceeding it, and §5.3 shows the near-miss
   tail is thin enough that better candidate construction buys well under one
   additional exact answer. The honest reading is that Example 21's exact-answer
   rate on ARC-AGI-1 is not yet distinguishable from zero.

The interesting number in that table is not `pass@2`. It is shape accuracy
0.031 → 0.601 and pixel 0.2235 → 0.4902: adaptation moves the model from
"produces a grid" to "produces the right-sized grid with about half the cells
right", and then stops. Sub-exact competence is real and measurable; exactness
is not reached.

## 4. Substrate fit: is pp-prop being used inside BrainScale's guarantees?

Example 21 configures `ETraceConfig(trace_factorization="io_factorized",
recurrence_scope="diagonal", decay=α)`, which is BrainScale's **ES-D-RTRL**
(Eqs. 4–6), not D-RTRL (Eq. 3). Three observations, in decreasing order of how
solid they are.

### 4.1 The sign-consistency condition is violated on 5 of 7 tracked parameters

ES-D-RTRL's rank-one collapse — the entire `O(BN)` memory win — is licensed by an
explicitly stated property: the inputs `x^t` to each tracked operation *maintain
a consistent sign across all time steps*; in AlignPost `x^t` is the binary spike
vector, in AlignPre it is the synaptic conductance. Both are non-negative, so
`ε_x = Σ α^k x^k` accumulates rather than cancels.

The compiler report in `var/example21-shared-2048n-2048e-b32-u13-l390/result.json`
lists exactly seven ETP-tracked parameters. Tracing what each one consumes:

| tracked parameter | its `x^t` | sign-consistent? |
|---|---|---|
| `rec_syn.comm.weight` | `neu.get_spike()` — binary spikes | **yes** (canonical AlignPost) |
| `ff_syn.comm.weight` | row event: one-hots, soft masks, softmaxes, normalised sizes — all ≥ 0 | **yes** |
| `workspace_query_projection.weight` | `_unit_l2_cap(workspace_carrier)`, i.e. membrane voltage in mV | **no** — signed |
| `answer_row_head.weight` | same capped voltage carrier | **no** — signed |
| `answer_shape_head.weight` | same capped voltage carrier | **no** — signed |
| `memory_read_projection.weight` | `raw_read` from `S`, built from `cos`-based random Fourier keys and `randn` value bases | **no** — signed |
| `memory_write_scale` | key ⊗ value products from those same signed bases | **no** — signed |

The two parameters that satisfy BrainScale's condition are the two ordinary SNN
projections. Every parameter on the binding, workspace-query, and decoding path —
that is, every parameter the 2026-08-17 architecture pivot was created to train —
consumes sign-varying inputs, so its `ε_x` accumulates cancelling terms and the
rank-one factorisation loses the justification the paper gives for it.

This is a genuine mismatch, not an extrapolation: it is a condition the paper
names in words, checkable statically, and currently false for the majority of the
learning path. It does not prove the gradients are useless — cancellation in
`ε_x` degrades trace quality by an amount nobody has measured here — but it does
mean the memory-efficiency argument is being taken on credit.

**Action:** for each of the five, either (a) map the input into a non-negative
code before the tracked op (rectified/split-sign features, a `softplus` or
half-wave pair, or reading `S` through a non-negative feature map), or (b) move
that parameter to `recurrence_scope`/factorisation settings whose derivation does
not need the sign property, and pay the memory, or (c) measure the resulting
trace error and record it as an accepted approximation. Option (c) requires §4.2
anyway.

### 4.2 Gradient error at the operating point is unmeasured, where the paper measured it

The measured workspace runs at `mean_firing_rate ≈ 0.309–0.311` (every checkpoint
of the on-disk 2048-neuron artifact). This is *inside* the empirical envelope of
BrainScale Figure 3A, which sweeps the spiking fraction to 0.6 and keeps
single-step cosine similarity ≥ 0.992 — so the diagonal approximation is not
obviously invalid at this rate, and any claim that it is would be wrong. But
three qualifications stack:

- Figure 3A measures **single-step** Jacobian similarity on standard SNN stacks
  (LIF/ALIF/GIF × Delta/Expon/STD/STP). Example 21 runs 390–510-tick episodes with
  a fast-weight side-memory in the loop. Single-step similarity says nothing
  about accumulated trace quality over hundreds of ticks.
- Figure 5C reports that **ES-D-RTRL has lower gradient similarity than
  D-RTRL**. Example 21 uses ES-D-RTRL and would be borrowing D-RTRL's validation.
- Example 21's own claim boundary states that it asserts no agreement with a BPTT
  oracle.

So the correct statement is not "outside the validated regime" but "**unquantified
where the paper quantified it**". The repository already mandates the instrument:
per AGENTS.md, a learning-rule assertion must be measured through the
finite-window oracle `chunked_online_param_gradients`, because a whole-sequence
VJP returns BPTT for every algorithm and passes vacuously. One measurement —
cosine similarity between pp-prop and the chunked oracle, per tracked parameter,
at the operating firing rate and at episode length — converts the largest open
question in the substrate story into a number, on CPU, without a GPU run.

### 4.3 Credit where due: `λI` recurrence on `S` is the right adaptation

The 2026-08-17 spec reasoned correctly that a binding carried in recurrent LIF
activity is invisible to a diagonal-scope rule, and moved the binding into a
dedicated state whose self-recurrence is `λI` — the one recurrence structure the
diagonal approximation represents *exactly*. That is the correct response to
pp-prop's truncation rather than a workaround, and it is the strongest piece of
design reasoning in the example.

The realisation, though, undercuts it. `memory_decay = 1.0` means `λ = 1`: pure
accumulation, no forgetting, into a 32×32 matrix (4,096 bytes per example) whose
key space is 32-dimensional. A task with 10 demonstrations of up to 30 rows,
written on both sides, issues up to ~600 outer-product writes into ~32 usable
directions. **Prediction:** pairing accuracy will remain near chance for capacity
reasons even though the architecture is now correct, and the failure will scale
with demonstration-row count rather than with training budget.

That prediction is currently untestable from any artifact on disk, which is the
next point.

## 5. The gate that would decide this is switched off in every artifact

Both fields are in the on-disk result:

```json
"associative_memory_diagnostics": {"available": false,
                                   "reason": "evaluation_controls_disabled"}
"associative_capability_status": "associative_capability_gates_pending"
```

The whole architectural pivot exists to convert presence = 0.99 / pairing = 0.51
into a bound memory. Whether `S_K` did that is **unmeasured in every artifact
currently on disk**. The derangement control (`_derange_task`, which preserves
both multisets and changes only the pairing) is the correct instrument and it is
already written; it is simply not enabled in the runs that were kept.

The minimum experiment that would settle §4.3 and §5 together: re-run with
`evaluation_controls` enabled at two or three values of `context_memory_width`
(32 / 128 / 512) and report pairing accuracy against demonstration-row count. If
pairing rises with width, the memory is capacity-limited and the architecture is
sound. If it stays at 0.51 at width 512, the write/read code — fixed random
features with a 32×32 learned gain — is the problem, not capacity.

## 6. Variable effort is instantiated but never measured over its range

The paper's `R` is a first-class knob and its Table 3 shows sharp
context-dependent behaviour. Example 21 declares 14 checkpoints (0…390) but:

- the **complete trained run** scored only three of them — efforts 0, 30, 60 —
  where effort does help: shape 0.0453 → 0.5227 → 0.6014, pixel 0.2277 → 0.4734
  → 0.4902;
- the runs that do sweep all 14 checkpoints are the small ones. In the on-disk
  2048-neuron / 13-update artifact, pixel accuracy saturates at ≈ 0.094 by tick
  30 and shape accuracy *falls* from 0.0048 to 0.0024, with
  `converged_fraction = 0.0` at every checkpoint and
  `mean_changed_cell_fraction` decaying from 0.496 to 0.0045 — the workspace
  stops changing without ever reaching a declared fixed point.

So the mechanism that gives the paper its title has been measured over 60 of its
390 declared ticks on a trained model, and over its full range only on models too
undertrained for the measurement to mean anything. Whether latent effort keeps
paying past tick 60 is open, and it is the single most on-thesis experiment
available.

## 7. Scale, cost, and reproducibility

**Scale.** 2,985,756 parameters against BDH-CQ's 150M — a 50× gap, and ~14% of
Example 21's count (`readout_projection`, `height_head`, `width_head`,
`color_factor_head` ≈ 413k) is not in the compiled model at all under
`decoder_mode="row_refinement"`; the compiler emits four `state_mismatch`
warnings for exactly those heads. Training data is the 399 public ARC-AGI-1
training tasks against the paper's private corpus plus ARC-AGI-1, RE-ARC,
ConceptARC, ARC-Heavy and ARC-GEN100K. No result here should be read as evidence
about the architecture's ceiling; it is a different order of experiment.

The recurrent workspace is also thinner than "recurrent" suggests. The reported
run realises 1,024 neurons / 262,144 edges (256 per neuron, 25% of the policy
cap), which is reasonable — but the artifacts kept in `var/` are 2,048 neurons
with **2,048 edges**, one per neuron, 0.098% budget utilisation. A workspace with
one recurrent edge per neuron has almost no cross-neuron mixing to do the latent
reasoning with; it also makes the diagonal-Jacobian approximation nearly exact
for the trivial reason that there is nothing off-diagonal left. The project's own
structural gate agrees and fails every artifact on disk:
`"actual model is not the required 4096-neuron/1048576-edge scale"`.

**Cost.** The paper's headline is the cost-accuracy Pareto point: 0.85 H200-seconds
per task, $0.00070. Example 21's complete run spent 3,350.1 s of adaptation over
400 tasks — about 8.4 s per task, plus evaluation, on an RTX 3080 Ti Laptop. The
fair framing is that per-task device time is the *same order of magnitude* as the
paper's (both figures are peak-FLOP fictions at unknown utilisation, so no ratio
is worth quoting), and that at a comparable per-task budget the accuracy gap is
118/400 against 1/400. Example 21 is not spending wildly more per task; it is
getting roughly two orders of magnitude less for it. The refusal to make a cost
claim is correct and should stay.

**Reproducibility.** No committed configuration reproduces the reported result.
The 1/400 run used task-local adaptation **on**, lr 3e-3, 10 epochs; the committed
defaults in `21-latent-reasoning-in-context.py` are `--task-local-adaptation` off
(a `store_true` flag), `--adaptation-learning-rate 5e-5`,
`--adaptation-epochs 2`, `--latent-steps 60`, `--training-updates 96`. The newer
per-tick online configuration documented as "now the default" in
`2026-08-19-example21-online-adaptation.md` has been measured only on 86–100
held-out *training* tasks and has never been run on the complete 400-task
evaluation split. The one complete-split number in existence was produced by a
configuration that the code no longer defaults to, and the current default has no
complete-split number. That gap should close before any further architectural
work.

## 8. Where Example 21 is the better scientific artifact

This should not be lost in the criticism above.

- **It reports its own failures.** `full_structural_qualification: false`,
  `full_scientific_qualification: false`, `model_only_score_gate_passed: false`,
  with machine-readable `reasons_not_scientific`. The gates fail loudly instead
  of being quietly relaxed. §5.1 of the results document says the solve count
  "ties rather than exceeds" the prior run.
- **It is auditable end to end.** Source-tree SHA-256, parameter SHA before and
  after evaluation, frozen-parameter byte identity, split-overlap check, data
  manifest hash, determinism check, per-package versions.
- **Its controls are better than the paper's.** The presence/pairing derangement
  probe isolates *superposition without binding* — a specific, falsifiable
  mechanism claim. BDH-CQ reports strong aggregate controls (Table 3
  demonstration support, Table 4 composition) but withholds dimensions, update
  rules, and recipe; it is not reproducible by anyone outside the authors.
- **The most interesting result here is not the ARC score.** It is the shared
  failure mode. BDH-CQ's Table 2 shows an 18.5-point gap between test-pair
  accuracy (77.92%) and strict-task accuracy (59.38%), with 52/160 ConceptARC
  tasks having one or two correct test inputs but not solving — the paper's own
  words are "inconsistent rule application". Example 21 measures the same
  pathology at a lower level and names its mechanism: content is present
  (0.99) but the input–output relation is not bound (0.51), and the adapted model
  gets shape right on 60% of queries while getting the whole grid right on 0.25%.
  Two systems, three orders of magnitude apart in scale, failing on binding
  consistency rather than on perception. That is the link worth pursuing, and it
  is a claim Example 21 can support and the paper cannot.

## 9. Ranked recommendations

1. **Add the missing clause to the claim boundary**: state that the scored
   configuration performs test-time gradient adaptation on each task's own
   demonstration folds, and that BDH-CQ performs no weight update at inference.
   Report frozen and adapted arms side by side wherever a number is quoted.
2. **Run the pending associative-capability gate** with
   `evaluation_controls` enabled and a `context_memory_width` sweep. This is the
   cheapest decisive experiment on the list, and §4.3 makes a falsifiable
   prediction about its outcome.
3. **Measure pp-prop against `chunked_online_param_gradients`** per tracked
   parameter at the operating firing rate and episode length. Converts §4.2 from
   an argument into a number.
4. **Fix or accept the sign-consistency violation** on the five tracked
   parameters in §4.1 — non-negative input coding is the cheap fix and is
   testable against (3).
5. **Sweep effort past 60 on a trained model** before adding architecture. It is
   the paper's central mechanism and it is currently measured over 15% of its
   declared range.
6. **Make the reported configuration the default**, or record the reported
   configuration as a committed preset, so the 1/400 number is reproducible from
   the repository as it stands.
7. **Learn `U_θ`.** Fixed random Fourier keys with a 32×32 learned gain is the
   weakest link between the implemented interface and the paper's. Only attempt
   this after (2) says whether capacity or coding is the binding constraint.

## 10. Bottom line

Example 21 is an honest, well-instrumented instantiation of BDH-CQ's *public
interface* on a spiking substrate, and its engineering discipline — fail-closed
gates, hashed parameters, leakage assertions, controls that can falsify its own
claims — exceeds what the paper itself makes checkable. On the science it is not
yet a comparison. In the paper's actual setting, frozen weights and
demonstrations entering only through context, it solves zero ARC-AGI-1 tasks; the
single solve it does report comes from test-time training the paper does not
perform, at a 50× parameter deficit and with the mechanism the whole example
exists to test — associative binding in `S_K` — still ungated and its variable-
effort knob measured over 15% of its range. The productive next steps are three
measurements, not another architecture.
