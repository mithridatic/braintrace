# Example 21 BrainCell ARC delivery plan

Status: Proposed for implementation approval
Planning issue: BRA-2
Execution authority: `openspec/changes/replace-example21-with-braincell-arc/`

## Scope and controls

This plan sequences the approved requirements. The OpenSpec proposal, design,
capability specification, and task list remain the execution authority. This
plan does not change a requirement or a frozen interface.

Implementation must not start until the board approves this plan. Work must use
a worktree branch. Cody owns the only implementation stream. Revy reviews each
release gate. The Project Manager owns roadmap and queue state. A Release
Manager must own release checks and finalization. No Release Manager is
currently available, so the Chief of Staff must provision or designate one
before the release gate.

The central source and test files are shared by most phases. Those phases run
in sequence. Documentation-only work can run in parallel only when it does not
describe code that has not passed its implementation gate.

## Active

- [ ] **BRA-2-G0 — Approve and queue the delivery plan**
  - Dependencies: OpenSpec change and architecture recommendations.
  - Assigned specialist: Project Manager. Approval owner: Chief of Staff or
    board.
  - Success criteria: The plan is approved; the child issues preserve the task
    order below; each child issue names its OpenSpec task range, dependencies,
    success criteria, assignee, reviewer, and worktree requirement; a Release
    Manager is designated before the release gate.

## Upcoming implementation sequence

- [ ] **BRA-2-I1 — Establish BrainCell compatibility**
  - OpenSpec tasks: 1.1-1.3.
  - Dependencies: BRA-2-G0.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: BrainCell 0.1.0 is pinned in every declared environment;
    import, Hodgkin-Huxley, PP-Prop relation, finite-difference, and spike-path
    fixtures pass without BPTT; Review Gate 1 passes.

- [ ] **BRA-2-I2 — Build ARC data, event, loss, result, and checkpoint contracts**
  - OpenSpec tasks: 2.1-3.5.
  - Dependencies: BRA-2-I1.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: Raw practice data loads directly; evaluation data is
    rejected in ordinary work; the 705 by 441 encoding round-trips exactly;
    target mutation cannot change inference input; strict loss, direct decoder,
    exact scoring, the 256 KiB result, and the 32 MiB immutable checkpoint meet
    every stated edge case; Review Gate 2 passes.

- [ ] **BRA-2-I3 — Implement the sparse BrainCell model and PP-Prop training**
  - OpenSpec tasks: 4.1-5.3.
  - Dependencies: BRA-2-I2.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: The exact 2,048-neuron topology and cell values execute;
    CSR rows are sources; input and recurrent storage remains sparse; false
    advances preserve biological and eligibility state bitwise; repeated work
    uses BrainState transforms; compiler relations are complete; episode reset,
    loss accumulation, clipping, Adam rates, and proof and ordinary update
    counts match the specification; Review Gate 3 passes.

- [ ] **BRA-2-I4 — Prove backend, timing, and direct behavior**
  - OpenSpec tasks: 6.1-6.4.
  - Dependencies: BRA-2-I3.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: Matched CPU and GPU probes select the literal lower valid
    median; decoder calls meet 100 ms; the real-data proof completes in 180
    seconds or less; only `d631b094` trains; `46f33fce` remains forward-only;
    weights and direct prediction change; interventions and direct strict data
    are recorded; focused coverage exceeds 90% and finishes in 60 seconds or
    less; Review Gate 4 passes.

- [ ] **BRA-2-I5 — Add bounded structural adaptation**
  - OpenSpec tasks: 7.1-7.6.
  - Dependencies: BRA-2-I4 passes the temporary proof.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: Contribution, ownership, twins, pruning, compaction,
    neuron addition, and connection addition use direct measured evidence and
    stable ties; no dense pair array is created; optimizer data remaps as
    specified; each candidate has one arm, stays within 300 seconds, gains at
    least one strict Boolean, and causes no strict regression; Review Gate 5
    passes.

- [ ] **BRA-2-I6 — Add measured Dale candidates and deferred-feature guards**
  - OpenSpec tasks: 8.1-8.4.
  - Dependencies: BRA-2-I5 and an accepted untyped parent checkpoint.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: Excitatory and inhibitory candidates start from the same
    parent; selection is measured and stable; the local sparse `weight_fn`
    preserves accepted signs through training and structural work; the strict
    gate accepts no regression; optional biological mechanisms remain disabled;
    Review Gate 6 passes.

- [ ] **BRA-2-I7 — Add the plot and implementation-truth documents**
  - OpenSpec tasks: 9.1-9.3.
  - Dependencies: BRA-2-I4 for baseline truth; BRA-2-I6 for structural and Dale
    truth.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: The explicit plot matches checkpoint counts and changes
    no prediction bytes; the causal explanation separates observations from
    inferences and shows direct data; the system model describes only executed
    code; all user-facing text follows ASD-STE100; Review Gate 7 passes.

- [ ] **BRA-2-I8 — Retire the obsolete Example 21 path**
  - OpenSpec tasks: 10.1-10.3.
  - Dependencies: BRA-2-I4 and a validated replacement path; BRA-2-I7 for the
    active documentation command.
  - Assigned specialist: Cody. Reviewer: Revy.
  - Success criteria: Only the BrainCell Example 21 path remains active; named
    latent-workspace, index, diagnostic, Docker, and README surfaces are
    removed; raw ARC files remain; scans find no removed import or forbidden
    shortcut; the public `braintrace` API is unchanged; Review Gate 8 passes.

- [ ] **BRA-2-V1 — Run final independent review and validation**
  - OpenSpec tasks: 11.1-11.3.
  - Dependencies: BRA-2-I1 through BRA-2-I8.
  - Assigned specialist: Revy. Remediation owner: Cody.
  - Success criteria: Focused tests, coverage, proof, decoder, ordinary run,
    strict OpenSpec validation, repository-wide OpenSpec validation, and
    `git diff --check` pass; the reviewer verifies docstrings, ASD-STE100,
    errors, naming, scope, and worktree cleanliness; failed review returns to
    Cody and does not advance.

- [ ] **BRA-2-R1 — Finalize the reviewed release**
  - Dependencies: BRA-2-V1 and a designated Release Manager.
  - Assigned specialist: Release Manager, currently unfilled.
  - Success criteria: The Release Manager runs the repository release process,
    confirms reviewer approval and release checks, gardens the roadmap after
    the reviewed release, and alone finalizes the release. The Project Manager
    and implementation specialist do not merge or finalize it.

## Architecture-review dispositions

| Finding | Severity and value | Disposition | Rationale |
|---|---|---|---|
| Implementation has separate approval | Critical | Gate at BRA-2-G0 | The architecture document approves documentation direction only. |
| Temporal weights need complete PP-Prop relations; BPTT is forbidden | High | BRA-2-I1, I3, and I4 | A missing relation invalidates the learning claim; local finite difference is the permitted independent check. |
| Real ARC input must be lossless, target-free, and shortcut-free | High | BRA-2-I2 and I4 | Direct evidence is invalid if target or handcrafted answer information reaches inference. |
| Repeated neural execution must be compiled and runtime bounded | High | BRA-2-I3 and I4 | The mission requires efficient learning, and the specification sets literal timing limits. |
| Example 20 used the wrong CSR source endpoint | High | BRA-2-I3 and I5 | Structural evidence must use rows as sources and columns as targets. |
| Results and checkpoints must remain small and atomic | Medium | BRA-2-I2 | Bounded artifacts support fast iteration and preserve the last valid parent. |
| Structural and Dale stages require a successful baseline proof | Medium | BRA-2-I5 and I6, sequenced after I4 | Added biology or structure has no value before the direct mechanism works. |
| The old command is breaking but not a public package API | Medium | BRA-2-I8 after replacement validation | Late retirement preserves rollback while preventing two active Example 21 paths. |
| Documentation must report implementation truth | Medium | BRA-2-I7 | Causal claims must follow direct predictions, targets, and interventions. |
| Optional biology is not part of this change | Low, future value | Backlog only | Each future mechanism needs a separate OpenSpec change and one-feature evidence gate. |

## Backlog

- Test one optional biological mechanism only after this change is reviewed and
  released. Create a separate OpenSpec change. Start with calcium-dependent
  adaptation, then consider HCN or NMDA in separate arms.
- Consider a bounded compiled coordinate structural search only if the block
  stages pass and direct evidence shows a need.
- Consider a larger real-ARC screen only with explicit approval after the fixed
  development screen shows repeatable strict improvement.

## Completed

No delivery item for this change is complete. Completed items move here only
after tests and independent review pass, with the completion date.
