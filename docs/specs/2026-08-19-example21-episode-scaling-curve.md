# Example 21 — episode-scaling curve

Status: pre-registered; running
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Follows `2026-08-19-example21-batched-training-results.md` §6 item 1.
Infrastructure: `2026-08-19-example21-resumable-pretraining.md`.

## 1. The question

Does exact `pass@2` move at all when pretraining episodes increase four-fold?

Every other lever measured in this session was eliminated: the adaptation budget
saturates past 40 steps, candidate construction resolves cell by cell to under
one additional exact answer, augmented adaptation folds failed their gate, and
neuron and edge scale was already measured. Only episodes survived, and the
evidence for it is a loss curve still descending at 96,000 episodes.

## 2. What is measured, and on what

Four segments of 6,000 updates at batch 16, chained through
`initial_checkpoint`, giving cumulative 96,000 / 192,000 / 288,000 / 384,000
episodes on one set of parameters. The first point is matched by construction to
the retained 96,000-episode run, so it is a control on this holdout rather than a
new claim.

Configuration: 1,024 neurons, 262,144 recurrent edges, learning rate 3e-3,
clip 1.0, `training_chunk_size` 50, augmentation on, seed 2108.

Every segment sets `training_holdout_tasks 100`, so the last 100 ARC-AGI-1
*training* tasks enter no gradient. `var/qual/adapt_probe.py --probe-tasks 100`
scores exactly those tasks under the production protocol: clone the checkpoint
per task, adapt with pp-prop on that task's own leave-one-out folds, score a
fold whose target never entered adaptation.

**The ARC evaluation split is not touched by any curve decision.** That is the
point of the holdout. A curve scored on the evaluation split would make the
final reported number a selected one.

**The probe is 100 tasks and not 16 for one reason.** The retained probe scored
15 usable tasks at exact `pass@2` 0.0667 — one task. It cannot distinguish one
exact answer from two, which is exactly why the augmentation gate in §5.6 of the
results document was uninformative at 1 versus 0. Shape and pixel accuracy are
stable at 32 tasks; an exact answer is a rare event and needs an order of
magnitude more.

## 3. Decision rule, fixed before the result

- **Exact `pass@2` rises monotonically or near-monotonically across the four
  points** — episodes convert. Extend the curve until it flattens, then run the
  complete 400-task evaluation at the best point.
- **Shape and pixel accuracy rise while exact `pass@2` stays flat** — episodes
  do not convert at this accuracy. Report that, and stop extending the run on
  the strength of pixel accuracy. §5.3 of the results document is why this
  outcome is live: only 11 of 419 queries sit within three cells of exact and
  40% have the wrong shape, so a large pixel gain can buy no exact answers.
- **Nothing moves** — the lever is dead and the binding constraint is elsewhere.

## 4. Expected magnitude, stated before the result

On the §5.3 distribution this lever plausibly moves the complete-split score
from 1 of 400 into the low single digits. It is the best-evidenced move
available and it is worth the GPU time, but it is not a route to a high ARC
score on its own, and no such claim is made here.

## 5. Cost

Measured at this configuration: 48.8 episodes/s, so a 96,000-episode segment is
about 33 minutes and a 100-task probe about 16 minutes. The four-point curve is
therefore roughly 3.3 hours. Peak device memory is 2.1 GiB against a 12.8 GiB
fail-closed limit, 16%.

Segments checkpoint every 10 chunks, so the WDDM fault that cost a 64-minute
stage earlier in this session can now cost at most 500 updates.
