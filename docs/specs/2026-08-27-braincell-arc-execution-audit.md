# BrainCell ARC execution audit

Date: 2026-08-27
Scope: `replace-example21-with-braincell-arc`

## Authority and state

The active execution authority is the OpenSpec change
`replace-example21-with-braincell-arc`. Its 19 requirements map to 42 unchecked
tasks in `tasks.md`. The older `add-latent-reasoning-example` change is a
completed implementation ledger. It does not define the replacement acceptance
criteria.

The replacement change is implementation-ready. The current project checkout
does not yet contain its new entry point or sibling test because the accepted
work remains in the standard gated delivery queue. Paperclip reports that this
audit workspace resolves to `git_worktree`, `origin/main`, and a non-primary
checkout. The recovery owner conflict is resolved.
[BRA-79](/BRA/issues/BRA-79) is in progress with
[Rainbow](/BRA/agents/rainbow), which is the live continuation path for Gate 5.
[BRA-49](/BRA/issues/BRA-49) remains blocked by that integration issue until
Rainbow records the required workspace, integration, and containment evidence.
Do not create a duplicate integration stream.

## Requirement coverage

| Requirement group | OpenSpec tasks | Measurable gate |
| --- | --- | --- |
| ARC boundary and events | 2.1–2.3 | Direct fixed-file loading; exact 705 by 441 event round trip; no target or evaluation leakage |
| BrainCell baseline and event step | 1.1–1.3, 4.1–4.5 | BrainCell 0.1.0 compatibility; exact sparse counts; finite state; delayed recurrence; frozen padding state |
| PP-Prop learning and schedule | 5.1–5.3 | Complete temporal relations; finite direct gradients; fixed 8- and 64-update schedules; no BPTT |
| Loss, prediction, scoring, and artifacts | 3.1–3.5 | Complete strict loss; direct integer output; zero-tolerance scoring; bounded result and checkpoint files |
| Backend, runtime, and proof | 6.1–6.4 | Measured backend selection; decoder at or below 100 ms; proof at or below 180 seconds; focused tests at or below 60 seconds and above 90% coverage |
| Structural stages | 7.1–7.6 | Evidence-ranked five-percent prune/add arms; immutable parent; fixed updates; false-to-true gain with no regression |
| Dale and deferred biology | 8.1–8.4 | Measured typed candidates; enforced signs; no default optional mechanism; reject slow or strict-flat future arms |
| Plot, documents, and retirement | 9.1–10.3 | Prediction-invariant plot; executed-system documents; no active obsolete entry point, module, command, or import |
| Final validation | 11.1–11.3 | Literal runtime and coverage gates; strict OpenSpec validation; clean diff; public-interface and language review |

No requirement lacks a task. The main planning risk is incorrect concurrency,
not missing acceptance coverage: most implementation tasks modify the same new
entry point and sibling test, so they form one work stream.

## Required execution order

1. Rainbow completes [BRA-79](/BRA/issues/BRA-79): record that its execution
   workspace resolves to `git_worktree`, `origin/main`, and a non-primary
   checkout; integrate the accepted Gate 5 commit; and record containment
   evidence.
2. One implementation specialist owns tasks 1 through 5. Compatibility tasks
   1.1–1.3 are the first gate. Data/artifact helpers may proceed in parallel
   only when their files do not overlap the model and driver files.
3. The same implementation specialist completes tasks 6.1–6.4 and records an
   accepted baseline proof. A failed proof stops the stream; it does not expand
   updates or change the task set.
4. After an accepted immutable checkpoint exists, the specialist completes
   structural tasks 7.1–7.6, then Dale tasks 8.1–8.4. Do not start these arms
   from an unaccepted baseline.
5. Complete tasks 9 and 10 from executed evidence, then run tasks 11.1–11.3.
6. Route the implementation to an independent Revy review. Rejected work
   returns to the implementation specialist. Approved work advances to Rainbow
   for release approval. Preserve `maxReviewRounds: 50` and Rainbow participant
   `872b1fed-7368-42f0-8233-ebc1df38e158`.

## Architecture-review dispositions

| Finding | Severity and value | Disposition | Rationale |
| --- | --- | --- | --- |
| Replace the legacy latent-workspace answer path with direct BrainCell output | Critical | Accepted in tasks 1–6 and 10 | It removes answer shortcuts and makes biological state and PP-Prop observable in the executed path. |
| Measure compatibility, causality, and runtime before structural claims | Critical | Accepted in tasks 1, 4, and 6 | Structural and Dale claims are invalid without a finite, trainable, time-bounded baseline. |
| Keep recurrent storage sparse and bound biological connections | High | Accepted in tasks 4.1–4.2 and 7 | It protects the efficiency mission and prevents hidden dense allocation. |
| Promote structural and Dale changes only from measured evidence | High | Accepted in tasks 7 and 8 | It prevents random or post-hoc biological claims and preserves the immutable parent checkpoint. |
| Add more optional biological mechanisms now | Medium value, high schedule risk | Deferred by task 8.4 | The approved change requires a minimal baseline. A later one-feature experiment must pass the same strict and runtime gates. |
| Keep the legacy system as a fallback answer path | Critical conflict | Rejected by tasks 10.1–10.3 | It conflicts with the direct-prediction requirement and would preserve unreviewed shortcuts. |

## Queue disposition

The standard Paperclip queue already covers OpenSpec tasks 1.1 through 11.3.
Gates 1 through 4 are complete. Gate 5 follows
[BRA-79](/BRA/issues/BRA-79) → [BRA-49](/BRA/issues/BRA-49) →
[BRA-46](/BRA/issues/BRA-46) → [BRA-9](/BRA/issues/BRA-9). Later issues keep
explicit blocker edges through final validation and release. Every engineering
issue in the active sequence has independent Revy review, Rainbow approval,
`maxReviewRounds: 50`, and the declared implementation return assignee.

No requirement gap or second queue is warranted. The recovery owner conflict is
closed through [BRA-110](/BRA/issues/BRA-110). Rainbow now owns the existing
[BRA-79](/BRA/issues/BRA-79) integration stream; no further coordination issue
is required by this audit.
