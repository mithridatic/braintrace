# Example 21 — per-tick online adaptation

Status: pre-registered; sweep running
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Mechanism: `2026-08-19-online-update-driver.md`.
Replaces the direction in `2026-08-19-example21-episode-scaling-curve.md`, which
was stopped by decision rather than by measurement.

## 1. Why this instead of more pretraining

The episode-scaling curve was the best-evidenced *pretraining* lever, and it was
stopped deliberately: more offline pretraining is not the direction being
pursued. What replaces it is the mechanism the pretraining path was throwing
away.

`etrace_grad` computes one gradient term per tick and accumulates them, so an
episode of 180 ticks produces exactly one optimizer step. The eligibility trace
makes every one of those terms complete at its own timestep, so the accumulation
is a choice. `etrace_online` applies them as they are produced, and an episode
whose supervised window is 60 ticks becomes 60 updates instead of one.

This matters most where the measurements say the model is actually bound.
Task-local adaptation is causally responsible for essentially the whole score —
shape accuracy 0.031 frozen against 0.601 adapted on the complete split — and it
is starved of data: exact `pass@2` was flat across 40, 120 and 300 adaptation
steps and across three learning rates, because a task supplies only a handful of
leave-one-out folds. Per-tick updates extract more learning from those same
folds without inventing any new data, which is a different axis from the step
count that was already measured flat.

## 2. Arms

All arms score the same 100 held-out ARC *training* tasks, adapt only on each
task's own folds, and never read an official query target. The evaluation split
is untouched.

| arm | start | updates per episode |
|---|---|---|
| control | pretrained checkpoint | one, accumulated |
| online | pretrained checkpoint | one per supervised tick |
| scratch | fresh initialization | one per supervised tick |

`control` is the retained path and reproduces the existing adaptation probe.
`online` isolates the mechanism: same data, same start, same episode, only the
update schedule differs. `scratch` is the arm that answers whether pretraining
can be removed rather than merely shortened.

The learning rate is swept for the online arms before the comparison, because
the two schedules are not comparable at a shared rate: 60 updates of size `L`
move roughly as far as one update of size `60L`, and Adam's normalization makes
the step size track `L` rather than the gradient magnitude.

### 2.1 The mask is binarized in the online arm

`etrace_grad` divides an accumulated gradient by the total mask weight, so
Example 21's `1/effort` weights are part of a reduction, not a per-step scale.
Carrying them into a per-step update would shrink every update by the number of
supervised ticks and silently confound the schedule with the rate. The online
arm therefore supervises the same ticks at weight one.

## 3. Decision rule, fixed before the result

- **`online` beats `control` on exact `pass@2`** — the update schedule is a real
  lever, and it applies to the reported evaluation path directly.
- **`online` matches `control`** — the mechanism is neutral here; report it as
  such rather than as an improvement, and the interesting arm is `scratch`.
- **`scratch` reaches a non-trivial exact score** — pretraining is removable and
  the whole offline stage can go.
- **`scratch` scores zero** — online updates are not by themselves a substitute
  for a shared prior, and the honest next question is whether that prior can be
  acquired online across a single pass over tasks rather than offline.

At 100 tasks a single exact answer is one percentage point, which is the
resolution these arms are separated at. A difference of one task is not a
result, and will not be reported as one.

## 4. What is not claimed

The driver adds an expressible regime. Nothing about it guarantees a better
score, and §3's second and fourth outcomes are both live. No ARC-score claim is
made until the arms are measured.
