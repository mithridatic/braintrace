# Example 21 — ARC pilot: exact-trace (D-RTRL) vs pp-prop write training

Status: approved 2026-08-21 ("You can try the arc run … do like 30 or something
real quick"); in progress
Date: 2026-08-21
Branch: `investigate/ex21-learned-memory-keys` (worktree `.worktrees/ex21-learned-keys`)
Depends on: `2026-08-21-etp-outer-write-drtrl-trace.md` (gradient verdict)

## Question

The gradient panel established that pp-prop delivers a corrupted training
signal to the write projections (wrong pairing direction, 2.5–11× too small)
and that an exact D-RTRL trace restores it (alignment 0.994–0.999, response
within 2%). Does a faithful write gradient convert into ARC behavior — above
all, does the **shuffled-demonstrations deviation** finally move off zero?

## Design

Two identical training runs differing only in the eligibility-trace engine:

| arm | `--trace-engine` | trace for the write projections |
|---|---|---|
| control | `pp_prop` | IO-factorized, rank-1 collapsed (yesterday's arm) |
| treatment | `d_rtrl` | per-parameter exact, position-retaining |

Both at `--memory-coding learned_write`. J sized the evaluation:
`--evaluation-task-limit 30` (not 100/419) for speed. All other knobs match
the recorded 2026-08-21 3-arm pilot exactly: seed 2108, 4096n/4096e, width 32,
memory decay 1.0, batch 32, 260 updates, chunk 5, lr 1e-3, latent 60,
row_refinement, `--evaluation-controls` on.

Implementation surface (this spec's code change): a `trace_engine` knob
threaded CLI → `ExperimentConfig` → `ModelConfig` → `etrace_config()`, where
`"d_rtrl"` returns the `per_param`/`diagonal` coordinate (no decay — the
exact trace has no filter). `braintrace.compile` picks the engine from the
coordinate; nothing else in the pipeline changes. TDD: 4 tests RED
(missing field/flag) → GREEN; the pp coordinate tests unchanged and green.

## Reading the result (30 tasks — coarse by design)

At 30 tasks each exact-match count is ±1 task ≈ ±0.033; treat anything inside
that as noise. The informative readouts, in order:

1. `shuffled_demonstrations` deviation (evaluation controls): the pairing
   readout. Yesterday: ≈0 in every arm. Any clearly negative deviation
   (shuffling *hurts*) in the d_rtrl arm is the first behavioral evidence of
   binding.
2. Write-projection `l2_delta` + `memory_coding_divergence`: did the write
   actually train, and did write/retrieval encoders stay coherent?
3. Shape/pixel exact-match vs the pp arm, same 30 tasks (both arms score the
   identical task set — seed-pinned).
4. Wall clock and peak memory of the d_rtrl arm on the 3080 Ti (16 GB,
   XLA fraction 0.80): this is also the feasibility measurement for any
   longer run. OOM or WDDM context loss is itself a result — record and fall
   back (batch 16, then 2048n) rather than tune silently.

A null (deviation still ≈0 under a verified-faithful gradient) is a real
answer: it moves the suspect from the learning rule to optimization/capacity
or the readout, and ends the learning-rule branch of the elimination chain.

## Scope

- 30-task evaluation, two runs, no third arm, no sweep.
- No task-local adaptation (banned); shared-model training only.
- Runs record `trace_engine` in their config dump automatically.
- Driver in scratchpad; results recorded here.

## Results — attempt 1 (full scale, 4096n / batch 32): d_rtrl infeasible

- **pp_prop arm**: completed clean in 229 s wall (training 52/52 windows in
  ~130 s, evaluation 37 s). Peak GPU memory 627 MB. Outputs in
  `var/example21-drtrl30-pp_prop/`. Topline at 30 tasks: 0 exact tasks at
  every effort, shape accuracy 0.613 @ effort 30/60, valid-cell pixel
  accuracy 0.440. Write projections trained (l2_delta: write_key_weight
  1.80, write_value_weight 1.93, write_key_bias 0.11, memory_write_scale
  0.58).
- **d_rtrl arm**: training rate measured at **~102 s per 5-update window**,
  steady across windows 2–6 (elapsed 610 s at 6/52, ETA ~78 min for
  training alone) vs ~2.5 s/window for pp — a **~40× slowdown**. The
  52-window run cannot finish inside the 45-min kill, so it was stopped at
  window 6 rather than left to die without an evaluation. Memory was not
  the binding constraint (pp peak 0.63 GB; no OOM observed before the
  kill). Suspected mechanism: `etp_outer_write`'s D-RTRL rules disable
  chunked execution (`_chunk_supported` → False), forcing the per-step
  scan fallback, on top of the position-retaining trace's extra einsums.

**Fallback decision** (deviation from the written ladder, with reasoning):
the ladder "batch 16, then 2048n" was written for OOM; the observed failure
is speed. At the measured rate, batch 16 alone predicts ≈ 2650 s of
training plus compile/eval — still over the 2700 s cap — so the combined
rung (**2048n + batch 16**) was taken directly. Both arms rerun at the
fallback config so the comparison stays matched; fallback outputs in
`var/example21-drtrl30fb-{pp_prop,d_rtrl}/`. Full-scale-vs-fallback scores
are not comparable across attempts; only within-attempt arm deltas count.

## Results — attempt 2 (fallback scale, 2048n / batch 16): NULL

Both arms completed. Wall clock: pp_prop 160 s; d_rtrl 1805 s (30.1 min,
~35 s/window — the fallback bought ~2.9×; still ~11× slower than pp).
Peak GPU memory: pp 0.53 GB, d_rtrl 2.92 GB. (d_rtrl's docker client
exited 125 at teardown — Docker Desktop noise; the run itself logged
`run 1/1` and wrote all artifacts.)

**1. Shuffled-demonstrations deviation (the headline readout): still ≈0
in both arms.** Shuffled-minus-intact, frozen_no_adaptation:

| effort | pp shape dev | pp pixel dev | d_rtrl shape dev | d_rtrl pixel dev |
|---|---|---|---|---|
| 0 | −0.032 | −0.073 | −0.032 | −0.044 |
| 30 | −0.032 | −0.017 | −0.032 | −0.013 |
| 60 | −0.032 | −0.008 | +0.000 | −0.017 |

Every entry is inside the ±0.033 one-task band (effort-0 pixel dips
appear in both arms identically — generic context disruption, not
binding). The faithful gradient did not create pairing sensitivity.

**2. Write projections trained comparably in both arms** (l2_delta,
pp vs d_rtrl): write_key_weight 1.80/2.32, write_value_weight 2.05/2.12,
write_key_bias 0.11/0.12, memory_write_scale 0.56/0.63. The exact trace
moved the write slightly more, to no behavioral effect.

**3. Task scores are arm-identical within noise**: 0 exact tasks in both
arms at every effort; intact shape accuracy @60: pp 0.613, d_rtrl 0.645
(one task apart); pixel 0.498 vs 0.488.

## Verdict

Attribution trap checked before interpreting: `write_retrieval_key_cosine`
= 0.99991 (d_rtrl) / 0.99993 (pp) — the untied write/retrieval encoders
did not drift, so the null is not a read-degradation artifact.

**Null under a verified-faithful gradient.** The gradient panel proved
the write projections now receive the true pairing gradient (alignment
0.994–0.999), the write demonstrably trains, and ARC binding behavior
still does not appear. Per the pre-registered reading: this **closes the
learning-rule branch of the elimination chain**. The corrupted pp-prop
write gradient was real (confirmed at the gradient level) but was NOT the
binding blocker at this scale/regime. Remaining suspects: optimization
regime (260 updates / lr), capacity–architecture (memory width 32,
single write per demonstration), and the readout path (whether the
decoder can exploit memory reads at all). A pp-compatible write
reparameterization (sign-consistency fix) is NOT warranted by this
result — it would fix a gradient that fixing didn't change behavior.

d_rtrl retires to what it always was here: a measuring instrument.
