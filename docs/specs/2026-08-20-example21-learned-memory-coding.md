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

## Design

New `ModelConfig.memory_coding` field (and matching `ExperimentConfig` field
+ `--memory-coding` CLI flag), literal values:

| value | key map | value map |
|---|---|---|
| `"frozen"` (default) | fixed RFF cosine (today, bit-exact) | fixed tanh projection (today) |
| `"learned_keys"` | trainable ETP `Linear` + cosine | fixed tanh projection |
| `"learned_keys_values"` | trainable ETP `Linear` + cosine | trainable ETP `Linear` + tanh |

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
| C | `learned_keys_values` |

Primary readouts are the pairing-sensitive controls (shuffled-demonstrations
deviation, slot ablation), then shape/pixel. Hypothesis: learned coding
improves binding specifically (shuffled-demonstrations arm degrades more
relative to intact than in frozen), with shape/pixel following.

- B/C > A on pairing-sensitive metrics → coding confirmed as the binder;
  next step is scaling the learned-coding run.
- B/C ≈ A with key weights verifiably moving → coding refuted at this
  scale; the deficit hypothesis moves to interference mechanisms the
  encoder cannot fix (e.g. single-trace superposition itself).
- Key weights do NOT move (digest unchanged) → compiler/plumbing failure,
  not evidence; fix before interpreting.

Trainability check: compare the key-projection parameter digest before/after
training (entry point already checkpoints parameters); frozen arm must keep
today's digests bit-exact.

Fallback arm (only if B stalls with weights moving): learned linear key map
*without* the cosine, in case training through `cos` is the obstacle.

## Tests (co-located, `latent_workspace_model_test.py` / entry-point tests)

1. `memory_coding` validation: default `"frozen"`; rejects unknown strings;
   rejects non-str.
2. Frozen arm bit-exactness: model with `memory_coding="frozen"` produces
   byte-identical key/value codes to pre-change behavior (guarded by the
   existing chunked-vs-unchunked bitwise digest test staying green).
3. Init equivalence: at initialization, `learned_keys` /
   `learned_keys_values` produce key/value codes allclose (fp32-tight) to
   frozen for random events, including zero-masking of invalid sides.
4. Trainability: a short pp-prop training loop on a tiny config moves the
   key `Linear` weights in `learned_keys` (and value weights in
   `learned_keys_values`) while `frozen` has no such parameters; compiler
   diagnostics report no rejected/moved required parameters.
5. Report: `associative_memory_report()` names the learned maps and keeps
   basis hashes.

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
