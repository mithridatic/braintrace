# Example 21 — per-tick online adaptation

Status: complete; schedule reported neutral at matched tuning
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

## 8. The three arms at full width

86 tasks scored of 100 offered; the rest carry fewer than three demonstrations.
Each arm adapts only on a task's own leave-one-out folds and never reads an
official query target.

| arm | shape | pixel | exact pass@2 | shape helped / hurt | fold loss | seconds |
|---|---|---|---:|---:|---|---:|
| control, accumulated 3e-3 | 0.616 -> 0.546 | 0.514 -> 0.520 | 1 -> 1 | 14 / 20 | 0.468 -> 0.211 | 825 |
| online, per tick 5e-5 | 0.616 -> **0.663** | 0.514 -> **0.551** | 1 -> 2 | 15 / 11 | 0.464 -> 0.250 | 1,820 |
| scratch, per tick 5e-5 | 0.151 -> 0.372 | 0.355 -> 0.493 | 0 -> 1 | **21 / 2** | 0.811 -> 0.374 | 1,879 |

### 8.1 Lower fold loss, worse answers

The control reaches the lowest fold loss of the three, 0.211, and gives the
worst held-out result: it is the only arm that *reduces* shape accuracy, and it
destroys the frozen model's one exact `pass@1`. It overfits 40 large steps into
a handful of repeated folds.

Per-tick updates at 5e-5 end at a **higher** fold loss, 0.250, and generalize
better on every axis. Many small steps are implicitly regularized where 40 large
ones are not. Fold loss is therefore not a usable model-selection signal for
this adaptation stage, which is worth stating because it is the quantity the
adaptation reports.

### 8.2 Adaptation without any pretraining

The scratch arm starts from a fresh initialization. No pretrained parameters
exist anywhere in it, and its only data is each task's own demonstrations.

It moves shape 0.151 -> 0.372 and pixel 0.355 -> 0.493, and it has the cleanest
per-task profile of any arm: shape improves on 21 tasks and degrades on 2,
against the control's 14 and 20. It reaches one exact answer, matching the
pretrained control.

**This meets the third outcome in §3, at the resolution floor.** One exact
answer of 86 is the same single task that §3 says is not a result, so the exact
number establishes nothing on its own. What is established is that per-task
online adaptation learns a large amount from demonstrations alone: a 0.22 shape
gain and a 0.14 pixel gain from random initialization, on tasks it has never
seen, with no offline stage of any kind.

It remains below the pretrained online arm on every axis, so a shared prior
still helps. The open question is no longer whether the prior must be acquired
offline, but whether it can be acquired across a single online pass over tasks.

### 8.3 Still unresolved

§7's fairness problem is not fixed by these numbers. The control ran at one
inherited rate and is visibly mis-tuned. The rate sweep for the accumulating arm
decides whether the per-tick schedule is the lever, or whether only the
effective step size ever mattered. No schedule claim is made until it lands.

## 9. Matched comparison: the schedule is close to neutral

The accumulating arm got the rate sweep §7 said it was owed. Its best measured
rate is 1e-3; 3e-3 was too high and 3e-4 and below stop buying anything. Rerun
at full width against the same 86 tasks:

| arm | shape | pixel | exact pass@2 | shape helped / hurt | fold loss | seconds |
|---|---|---|---:|---:|---|---:|
| accumulated 3e-3 | 0.616 -> 0.546 | 0.514 -> 0.520 | 1 | 14 / 20 | 0.468 -> 0.211 | 825 |
| accumulated 1e-3, tuned | 0.616 -> 0.628 | 0.514 -> 0.544 | 1 | 15 / 14 | 0.468 -> 0.241 | 804 |
| per tick 5e-5 | 0.616 -> 0.663 | 0.514 -> 0.551 | 2 | 15 / 11 | 0.464 -> 0.250 | 1,820 |

**Most of the apparent advantage was tuning, not schedule.** Against a tuned
accumulating arm the per-tick gap narrows to 0.035 shape and 0.007 pixel. At 86
tasks 0.035 shape is three tasks, and the exact difference is one. Neither
separates.

This is §3's second outcome and it is reported as such: **the update schedule is
close to neutral for ARC task-local adaptation at this scale, and no improvement
is claimed for it.** It costs 2.3x the wall clock, so on this evidence there is
no reason to prefer it here.

Sections 5 and 8 compared per-tick at a swept rate against accumulation at an
inherited one, and their shape numbers should be read only through this section.

### 9.1 What does survive

Fold loss is anti-correlated with held-out quality across all three pretrained
arms — final loss 0.211, 0.241, 0.250 orders exactly opposite to shape accuracy
0.546, 0.628, 0.663. The adaptation stage reports fold loss, and it must not be
used to select a configuration.

The driver itself stands on its own: `etrace_online` makes a regime expressible
that the library could not express, and the accompanying tests pin its contract.
That is worth having independently of whether it wins on this task.

### 9.2 The result that matters is §8.2

The scratch arm carries no pretrained parameters and still improves shape on 21
tasks against 2, from demonstrations alone. That is much larger and much cleaner
than any schedule effect measured here, and it is the direction worth pursuing:
whether the prior the pretrained arms enjoy can be acquired across a single
online pass over tasks rather than an offline stage.
