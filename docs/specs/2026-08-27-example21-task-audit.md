# Example 21 task audit

Date: 2026-08-27
Status: planning audit; the OpenSpec change remains the execution authority

## Authority

- Active change: `openspec/changes/replace-example21-with-braincell-arc/`
- Binding requirements:
  `openspec/changes/replace-example21-with-braincell-arc/specs/pp-prop-braincell-arc/spec.md`
- Executable checklist:
  `openspec/changes/replace-example21-with-braincell-arc/tasks.md`
- Approved architecture record:
  `docs/specs/2026-08-24-example21-architecture-recommendations.md`

This audit does not create a second task queue. Paperclip issues must link to
the OpenSpec task identifiers that they execute.

## Requirement coverage

| Binding requirement | OpenSpec tasks | Disposition |
| --- | --- | --- |
| Real ARC task boundary | 2.1, 2.3, 5.3, 6.3 | Covered by loader, fixed screens, proof isolation, and validation-only execution checks. |
| Lossless temporal ARC events | 2.2, 4.3 | Covered by event round trips, byte stability, fixed capacity, and false-advance state checks. |
| Minimal untyped BrainCell baseline | 1.2, 4.1, 4.2, 4.4 | Covered by compatibility fixtures, exact sparse counts, cell construction, and the matched timestep gate. |
| One biological event step | 4.2, 4.3 | Covered by bounded current-density paths, prior-spike timing, and padding-state identity. |
| Compiled BrainTrace PP-Prop learning path | 1.2, 1.3, 4.3, 5.1 | Covered by compiler relations, independent derivative fixtures, compiled loops, and temporal-parameter checks. |
| Fixed bounded training schedule | 4.5, 5.2, 5.3 | Covered by reset behavior, one update per episode, exact update counts, and fail-closed scheduling. |
| Strict-aligned request loss | 3.1, 6.3 | Covered by request-only supervision and direct pretraining and post-training loss evidence. |
| Direct integer prediction | 3.2, 4.5, 6.3 | Covered by direct voltage readout, target-free integer decoding, and state-intervention evidence. |
| Zero-tolerance strict task scoring | 3.3 | Covered by query and task exactness fixtures with no partial score. |
| Small direct result | 3.4 | Covered by the exact schema, atomic write, extra-field rejection, and 256 KiB limit. |
| Compact immutable checkpoint | 3.5 | Covered by array-only round trips, validation, reset, size, atomicity, and parent preservation. |
| Measured backend and runtime limits | 6.1, 6.2, 6.3, 6.4, 7.6, 11.1 | Covered by literal backend, decoder, proof, focused-test, structural-arm, and ordinary-run gates. |
| Temporary real-data proof | 5.3, 6.3 | Covered by fixed training and validation roles, direct mechanism observations, changed behavior, and the three-minute gate. |
| Observed structural stages | 7.1-7.6 | Covered by measured attribution, exact pruning and addition counts, physical compaction, fixed updates, and strict promotion gates. |
| Observed Dale-type stages | 8.1-8.3 | Covered by evidence-based separate candidates, differentiable sign enforcement, and strict promotion gates. |
| Deferred biological detail is evidence gated | 8.4 | Covered by a default-off guard and explicit later-change conditions. |
| Optional structural visualization | 9.1 | Covered by checkpoint-derived topology and state-neutral plotting. |
| Minimal implementation and clear documentation | 6.4, 9.2, 9.3, 11.3 | Covered by co-located meaningful coverage, implementation-truth documents, docstrings, and language review. |
| Obsolete Example 21 path is retired | 10.1-10.3, 11.2 | Covered by command replacement, removal, reference scans, public-API preservation, and strict validation. |

No binding requirement lacks an OpenSpec task. The prior
`add-latent-reasoning-example` checklist is complete and is superseded for
active Example 21 execution. It must not receive new implementation work.

## Roadmap state

### Active

- [ ] **BRA-126: reconcile the task queue with the active change.**
  Dependency: the active OpenSpec change and approved architecture record.
  Success: every queued Example 21 issue points to one E21 item below, has one
  owner, has no overlap with another in-flight issue, and uses the standard
  Revy-to-Rainbow execution policy. Assigned specialist: Project Manager.

No implementation item is active. The architecture record says implementation
needs separate approval. Do not dispatch E21-1 through E21-10 until that approval
is present in the standard planning system.

### Upcoming

- [ ] **E21-1 through E21-10.** Execute in the dependency order in the next
  section after implementation approval. Assign each item before it becomes
  active. Do not use a board user as an execution-policy participant.

### Completed

- None for the replacement change. Move an E21 item here only after its tests,
  independent review, and Rainbow release approval pass. Record the completion
  date at that time.

### Backlog

- Chemical synapses, extra channels, morphology, neuromodulation, persistent
  memory, and multi-layer recurrence. Each needs a separate measured change
  after the direct BrainCell baseline passes its promotion gate.

## Execution order

| Item | Dependencies | Measurable exit | Specialist |
| --- | --- | --- | --- |
| E21-1: OpenSpec section 1 | Approved change | Clean imports; finite compatibility and derivative fixtures | Dependency and numerical-compatibility implementer |
| E21-2: sections 2-3 | E21-1 for installed runtime | Loader, event, loss, decoder, scorer, result, and checkpoint tests pass | Example 21 implementation owner |
| E21-3: section 4 | E21-1 and E21-2 | Exact sparse counts; finite state; compiled padding and reset gates pass | Same Example 21 implementation owner |
| E21-4: section 5 | E21-3 | Compiler relations and fixed 8/64-update schedules pass without BPTT | Same Example 21 implementation owner |
| E21-5: section 6 | E21-4 | Temporary proof, decoder, focused-test, and backend gates meet literal limits | Same Example 21 implementation owner |
| E21-6: sections 7-8 | E21-5 | Structural and Dale candidates obey exact count, causal, sign, and strict-result gates | Same Example 21 implementation owner |
| E21-7: section 9 | E21-5; accepted checkpoints from E21-6 when used | Plot is state-neutral; both documents describe executed evidence only | Documentation specialist, with measured artifacts from the implementation owner |
| E21-8: section 10 | E21-5 through E21-7 | One active Example 21 command remains; reference scan passes; public API is unchanged | Example 21 implementation owner |
| E21-9: section 11 | E21-8 | Focused gates, strict OpenSpec validation, `git diff --check`, docstring, and ASD-STE100 checks pass | Example 21 implementation owner, then independent Revy review |
| E21-10: release approval | E21-9 and approved Revy review | Rainbow completes the repository release checks and finalizes the release | Rainbow (`872b1fed-7368-42f0-8233-ebc1df38e158`) |

The specialist column defines the required role until a named agent is assigned
in Paperclip. An unassigned item stays Upcoming; it must not be treated as an
active work stream.

## Dispatch constraints

- Sections 2-8 and 10-11 affect
  `examples/pp_prop/21-braincell-arc.py` and its sibling test. They are one
  work stream. Do not assign them to two agents at the same time.
- Section 1 can run independently only while it changes dependency and
  compatibility files that do not overlap the executable stream.
- Section 9 can run independently only after the implementation owner supplies
  the measured artifacts. The documentation specialist must not invent results.
- Every engineering issue must use the standard execution policy: implementation
  owner, independent Revy review with `maxReviewRounds: 50`, then Rainbow release
  approval. The implementation owner is the review return assignee.
- Before dispatch, the ARC-AGI workspace policy must resolve to `git_worktree`,
  `origin/main`, and a non-primary checkout. Workspace repair and integration
  belong to Rainbow.

## Architecture-review dispositions

- High value: keep the corrected CSR source/target semantics in tasks 4.1,
  4.2, 7.1, and 7.5. Rationale: the approved review found that the earlier
  Example 20 contribution path used the wrong source endpoint.
- High value: keep the one-step finite-difference and spike-path fixtures in
  tasks 1.2-1.3. Rationale: PP-Prop is approximate over time, so a BPTT equality
  claim would be invalid.
- High value: keep the temporary real-data proof before structural and Dale
  work. Rationale: later stages need observed direct behavior from an accepted
  baseline.
- Deferred: chemical synapses, extra channels, morphology, neuromodulation,
  persistent memory, and multi-layer recurrence. Rationale: the approved change
  requires one measured feature per later experiment. These items remain outside
  the active queue until the baseline and promotion gate pass.

## Queue gardening rules

- Treat queued and running engineering issues as In Progress in the roadmap.
- Move an item to Completed only after tests, independent review, and release
  approval pass. Record the completion date then.
- Put implementation discoveries that are outside this OpenSpec change in the
  Backlog. Do not expand the active change silently.
- Only Rainbow can finalize the release.
