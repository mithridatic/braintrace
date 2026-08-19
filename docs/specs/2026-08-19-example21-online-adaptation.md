# Example 21 — per-tick online adaptation

Status: control arm measured and found mis-tuned; comparison incomplete
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

## 5. Learning-rate sweep

23 held-out training tasks, one pretrained checkpoint (96,000 episodes,
1,024 neurons, 262,144 edges), 40 adaptation updates at adaptation batch 4.
Frozen accuracy is identical across every row because every arm starts from the
same parameters.

| schedule | rate | shape | pixel | exact pass@2 | seconds |
|---|---:|---|---|---:|---:|
| accumulated | 3e-3 | 0.5217 -> 0.5217 | 0.4809 -> 0.5253 | 0.0435 | 224 |
| per tick | 5e-5 | 0.5217 -> **0.6957** | 0.4809 -> 0.5365 | 0.0435 | 449 |
| per tick | 2e-4 | 0.5217 -> 0.5652 | 0.4809 -> 0.5461 | 0.0435 | 485 |
| per tick | 1e-3 | 0.5217 -> 0.4783 | 0.4809 -> 0.5171 | 0.0435 | 463 |

**Shape separates the schedules; exact does not resolve at this width.** The
accumulating path does not move shape accuracy at all over 40 updates. Per-tick
updates at 5e-5 move it by 0.174 on the same folds, from the same parameters,
with the same loss. That matters because wrong shape is the largest failure
bucket on the complete split, 40% of 419 queries, and shape is only 60 logits.

Every arm scores exactly one exact answer of 23, which is the resolution floor
this sweep was never able to see past. Nothing about exactness is claimed from
it; that is what the 100-task arms in §2 are for.

The rate ordering is monotone over the range measured — 5e-5 beats 2e-4 beats
1e-3 — so 5e-5 is a boundary rather than an interior optimum, and a lower rate
may be better still. It is carried into the full arms as the best *measured*
value, not as a tuned one.

Per-tick adaptation costs roughly twice the wall clock, which is the honest
price of 2,400 optimizer updates per task against 40.

## 6. Cost, and one correction

Building the jitted runner, the optimizer, and the compiled learner inside the
per-task loop made every task rebuild its program. Hoisting them out, and
restoring a snapshot of untouched Adam state between tasks so each still starts
clean, took the probe from 19.5 to 13.3 seconds per task.

The remaining 11.2 seconds per task is real work rather than overhead: 40
episodes of 180 ticks is 7,200 sequential recurrent steps with a pp-prop
gradient at each, and the online arm applies 2,400 Adam updates over 1.8 million
parameters.

An intermediate reading claimed the probe was host-bound, on eight consecutive
`nvidia-smi` samples reading 0% utilization. Sampling across a whole task
instead shows the compiled adaptation saturating the device at 96-100% and
148 W, with the zeros falling in the host gaps between stages. The claim was an
artifact of when the samples were taken, and is withdrawn.

## 7. The control arm is mis-tuned, and the comparison is not yet fair

At full probe width the control arm makes shape accuracy **worse**:

| arm | shape | pixel | exact pass@1 | exact pass@2 |
|---|---|---|---:|---:|
| frozen | 0.6163 | 0.5140 | 0.0116 | 0.0116 |
| control, accumulated at 3e-3 | 0.5465 | 0.5204 | 0.0000 | 0.0116 |

86 tasks scored of 100 offered; the rest carry fewer than three
demonstrations. Adaptation helped shape on 14 tasks and hurt it on 20, and it
destroyed the one exact `pass@1` the frozen model had.

That is a mis-tuned optimizer, not a property of accumulation. The rate 3e-3 is
inherited from the retained production configuration, which runs a different
fold schedule — capacity 70 over 10 epochs — rather than this probe's 40 updates
at batch 4.

**So the sweep in §5 is not a fair comparison and must not be reported as one.**
It swept the online rate over three values and left the accumulating arm at one
inherited value. "Online at its best measured rate beats accumulation at an
untuned rate" is not the claim this specification set out to test, and the
degradation above is evidence the control's rate is genuinely wrong rather than
merely unoptimized.

The comparison is therefore incomplete until the accumulating arm gets the same
treatment: a rate sweep on the same tasks, with its best measured rate carried
into the full-width comparison. Until that runs, §5's shape result stands only
as evidence that per-tick updates *can* improve shape where one particular
accumulating configuration degrades it — not that the schedule is the cause.

This was found by looking at the per-task helped/hurt split rather than the
aggregate, which is the same discipline that resolved the near-miss tail in the
earlier results document: an aggregate that moves in the expected direction can
still be hiding the opposite mechanism.
