# Example 21 — Learned memory coding (U_θ, recommendation #7)

Status: spec; implementation on `investigate/ex21-learned-memory-keys`
Date: 2026-08-20
Branch: `investigate/ex21-learned-memory-keys` (worktree `.worktrees/ex21-learned-keys`)

## Question

BDH-CQ's demonstration-ingestion operator `S_t = U_θ(S_{t-1}, D_t)` is learned
end to end — *what to store* is part of the learning problem. Example 21
freezes the storage coding: keys are fixed random Fourier features
(`scale · cos(γ·(features @ B_k) + b)`, seeds +101/+102), values are a fixed
`tanh(features @ B_v)` projection (seed +103), and the only trainable
write-path parameter is the 32×32 elementwise `memory_write_scale`. The
system can learn how much to weight a random binding; it cannot learn a
better binding.

Both capacity axes are now exhausted
(`2026-08-20-example21-context-width-sweep.md`,
`2026-08-20-example21-binding-load-sweep.md`): binding is perfect and
width-invariant (32/512) and load-invariant (K = 4/6/8) on the clean
curriculum, while ARC binding stays deficient under hundreds of heterogeneous
correlated row-event writes. The remaining suspect is the addressing metric:
random-projection similarity collides rows that should address different
slots and orthogonalizes rows that should co-address, and no elementwise gain
can rotate the code. **Does making the key (and value) encoders trainable
improve ARC binding?**

## Design (revised during implementation — see "Structural finding")

New `ModelConfig.memory_coding` field (and matching `ExperimentConfig` field
+ `--memory-coding` CLI flag), literal values:

| value | key map | value map |
|---|---|---|
| `"frozen"` (default) | fixed RFF cosine (today, bit-exact) | fixed tanh projection (today) |
| `"learned_keys"` | trainable ETP `Linear` + cosine, trained via retrieval path | fixed tanh projection |

### Structural finding: pp-prop cannot differentiate the memory write

The original three-arm design (`learned_keys_values` as arm C) is
**structurally impossible under pp-prop**. The algorithm requires a
position-preserving (elementwise) path from every ETP primitive's output to
each hidden state it feeds; the outer-product write
`einsum("bi,bj->bij", key, value)` into `context_memory` mixes positions, so
`compile_pp_prop` raises `NotSupportedError` for any trainable operator
upstream of the write. Consequences:

- **Keys** have a second, elementwise path to hidden state — the query
  encoding / read side — so the key projection *can* train, but only through
  retrieval. The write-side key is wrapped in `stop_gradient` (forward values
  unchanged; the written code still co-adapts because write and query share
  the same weights). The gradient is approximate: it ignores the dependence
  of the stored memory on the key parameters.
- **Values** reach hidden state *only* through the mixing write. There is no
  gradient path at all; `learned_keys_values` is removed rather than shipped
  as a silently-untrained arm. Training the value coding requires a new ETP
  outer-product primitive with its own trace rule (Layer 1–3 work) — recorded
  as follow-up, not attempted here.

### Compiler change (braintrace Layer 2)

`_forward_reachable_hidden_vars` in `braintrace/_compiler/hid_param_op.py`
expanded through `stop_gradient`, so the zero-gradient write path still
connected the key projection to `context_memory` and pp-prop rejected the
tail. `stop_gradient` is now a barrier in that reachability walk (both the
direct-group discovery and the full collection), consistent with the
hidden-group tracer which already treated it as one. A hidden state
reachable only through `stop_gradient` receives exactly zero gradient, so
excluding it changes no gradient value — it only stops algorithms from
rejecting paths that carry nothing. Covered by
`hid_param_op_test.py::TestStopGradientBarrier`.

Initialization is **function-identical to frozen at step 0**:

- Key `Linear(key_width → memory_width)` with `w_init = γ · B_k` and
  `b_init = b` (the +102 phase vector). Then
  `phase = linear(features)`, `code = √(2/W) · cos(phase)` — identical to
  `γ·(f @ B_k) + b` since `γ·(f@B) = f@(γB)`. γ is absorbed into the learned
  weight; the phase bias trains as the Linear bias (routed through the same
  ETP `matmul` primitive, same ParamState).
- Value `Linear(value_width → memory_width)` with `w_init = B_v`,
  `b_init=None`; `code = tanh(linear(features))`.

Side-validity zero-masking, write gate, decay, `memory_write_scale`, query
and read paths are untouched. `AssociativeMemoryReport` reports
`key_map="learned_rff_cosine"` / `value_map="learned_tanh_projection"` and
component types `braintrace.nn.Linear` for learned arms; sha256 fields keep
hashing the *initial* bases so provenance stays checkable.

### Why ETP `Linear` and not a bare `ParamState`

Parameters train only through ETP primitives (established in the
gated-carrier round): a `ParamState` used in plain `jnp` math silently never
receives gradients. `braintrace.nn.Linear.update` routes through the ETP
`matmul` primitive, which is exactly what `workspace_query_projection`
already does upstream of a HiddenState update. The novelty is a `Linear`
upstream of `cos`/`tanh` and the outer-product write into `context_memory` —
this is the thing the smoke stage must confirm via compiler diagnostics
before any training curve is believed.

## Arms and decision rule

ARC configuration, seed 2108, `--evaluation-controls`, 100-task evaluation
cap (pilot scale; non-scientific by construction), image rebuilt from this
worktree with its own tag.

| arm | `--memory-coding` |
|---|---|
| A baseline | `frozen` (reuse existing width-sweep w32 pilot artifact where comparable) |
| B | `learned_keys` |

(Arm C `learned_keys_values` removed — structurally untrainable under
pp-prop; see "Structural finding".)

Primary readouts are the pairing-sensitive controls (shuffled-demonstrations
deviation, slot ablation), then shape/pixel. Hypothesis: learned coding
improves binding specifically (shuffled-demonstrations arm degrades more
relative to intact than in frozen), with shape/pixel following.

- B > A on pairing-sensitive metrics → coding confirmed as the binder;
  next steps are scaling the learned-keys run and the outer-product ETP
  primitive that would unlock value learning.
- B ≈ A with key weights verifiably moving → retrieval-path key learning is
  insufficient at this scale; remaining hypotheses are the write-path
  gradient (needs the outer-product primitive) or interference mechanisms
  the encoder cannot fix.
- Key weights do NOT move (digest unchanged) → compiler/plumbing failure,
  not evidence; fix before interpreting.

Trainability check: compare the key-projection parameter digest before/after
training (entry point already checkpoints parameters); frozen arm must keep
today's digests bit-exact.

Fallback arm (only if B stalls with weights moving): learned linear key map
*without* the cosine, in case training through `cos` is the obstacle.

## Tests (co-located, `latent_workspace_model_test.py` / entry-point tests)

1. `memory_coding` validation: default `"frozen"`; rejects unknown strings,
   the removed `"learned_keys_values"`, non-str, and learned coding without
   memory. (`test_memory_coding_defaults_to_frozen_and_validates`)
2. Init equivalence: `learned_keys` key codes allclose (fp32-tight) to
   frozen, value codes byte-identical, invalid-side rows exactly zero.
   (`test_learned_memory_coding_matches_frozen_codes_at_initialization`)
3. Report: learned arm names `learned_rff_cosine_retrieval_path`, keeps
   basis hashes. (`test_learned_memory_coding_report_names_components...`)
4. Trainability: pp-prop compiles; the key path is an etrace weight,
   `all_direct`, its relation excludes `context_memory` and includes
   `query_encoding`; finite-window pp-prop gradient is nonzero; frozen arm
   has no such path. (`test_learned_key_coding_trains_through_retrieval...`)
5. Compiler barrier: `stop_gradient` severs relation reachability.
   (`hid_param_op_test.py::TestStopGradientBarrier`)
6. Entry point: `--memory-coding` default/choices, ExperimentConfig →
   ModelConfig wiring incl. smoke path.
   (`test_memory_coding_flag_wires_into_model_config`)

## Execution plan

1. Implement + tests; full 4-file suite in Docker **without** `--gpus all`.
2. Smoke: `--smoke --memory-coding learned_keys` in-container; inspect
   compiler diagnostics.
3. GPU pilots (arms B, C): 100-task cap, evaluation controls, measured wall
   clock, sub-hour by construction (prior pilots of this shape measured
   233 s at width 32).
4. Record results here; propose scale-up separately.

## Results

(to be filled after runs)
