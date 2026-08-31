# BrainCell ARC task audit

Date: 2026-08-27

Authority: The active OpenSpec change
[`replace-example21-with-braincell-arc`](../../openspec/changes/replace-example21-with-braincell-arc/)
defines the implementation. The Paperclip issue graph defines assignment and
release state. This document is an audit record. It is not a second task queue.

## Result

The OpenSpec checklist covers each requirement in the capability delta. The
checklist also preserves the required order: compatibility, data and output,
baseline model, proof, structural stages, Dale stages, retirement, and final
validation.

The earlier `add-latent-reasoning-example` change is a superseded historical
record. Do not dispatch work from its checklist. The replacement change is the
only execution authority for active Example 21 work.

Do not dispatch implementation groups 2 through 8 to different agents. These
groups all change `examples/pp_prop/21-braincell-arc.py` and its sibling test.
Parallel ownership would duplicate one work stream and create conflicting
changes.

The dependency and documentation work in group 1 and parts of groups 9 and 10
can run in parallel only when their file sets do not overlap the implementation
stream. The Release Manager owns workspace integration.

## Verified repository state

- The active checklist has 42 tasks and no checked task in this checkout.
- `examples/pp_prop/21-braincell-arc.py` and its sibling test do not exist.
- The obsolete Example 21 entry points and `latent_workspace*` modules still
  exist. Their removal remains task group 10 and must not happen before the
  replacement passes its reference and command scans.
- The next valid implementation batch is `1.1–1.3`. Tasks `2.1–11.3` depend on
  the compatibility boundary or later proof evidence as stated below.
- No live assignment or notification is claimed in this document. Paperclip is
  the assignment authority, and each dispatch requires a verified execution
  policy and isolated worktree.

## Continuation recovery state

The 2026-08-27 continuation could not complete live routing. The run-scoped
Paperclip API bridge refused both read attempts. The blocked-status write also
could not be confirmed and was not retried after the bounded failure limit.
Git metadata remained read-only, so this audit could not be committed on the
execution branch.

The Paperclip runtime operator owns recovery. Restore the run-scoped API bridge
and writable Git metadata for the execution worktree, then wake the Project
Manager on the existing issue. The resumed run must complete these actions in
order:

1. Read the current issue graph, agent roster, and execution-policy state.
2. Verify that ARC work resolves to `git_worktree`, base ref `origin/main`, and
   a non-primary checkout.
3. Reconcile the next `1.1–1.3` implementation assignment without splitting the
   shared tasks `2.1–8.4` across owners.
4. Verify independent Revy review, `maxReviewRounds: 50`, the implementer as
   return assignee, and Rainbow as the release approval participant.
5. Notify only the verified owner or owners through the standard issue queue.

## Requirement coverage

| Capability requirement | OpenSpec task identifiers | Measurable success gate | Specialist |
| --- | --- | --- | --- |
| Real ARC task boundary and lossless events | 2.1–2.3 | Direct named-file loads, exact round trips, 705 by 441 byte-stable events, and no target leakage | Implementation specialist |
| Minimal BrainCell baseline and event step | 1.2–1.3, 4.1–4.5 | Exact neuron and sparse-connection counts, finite declared cell state, compiled padding freeze, matched integration gate, and causal state use | Implementation specialist |
| PP-Prop learning and fixed schedule | 5.1–5.3 | Complete temporal relations, finite independent derivative fixtures, exact update counts, and no BPTT or validation update | Implementation specialist |
| Strict loss, direct prediction, and exact score | 3.1–3.3 | Every strict datum affects loss, direct integer decoding covers all dimensions and cells, and only zero-tolerance pass-at-1 is reported | Implementation specialist |
| Small result and immutable checkpoint | 3.4–3.5 | Result is at most 256 KiB; checkpoint is at most 32 MiB, array-only, atomic, and preserves its parent on failure | Implementation specialist |
| Backend, runtime, and temporary proof | 6.1–6.4 | Matched backend selection, decoder at most 100 ms, proof at most 180 seconds, focused tests at most 60 seconds, and coverage above 90 percent | Performance specialist after implementation handoff |
| Observed structural stages | 7.1–7.6 | Exact five-percent changes, no pruning at zero strict score, no strict regression, physical compaction, and bounded sparse selection | Same implementation specialist after proof passes |
| Observed Dale stages and deferred biology | 8.1–8.4 | Matched typed candidates, persistent effective signs, no chemical mechanism in the type arm, and no optional mechanism in the baseline | Same implementation specialist after proof passes |
| Plot and implementation-truth documents | 9.1–9.3 | Plot matches executed counts without changing predictions; documents separate observations from inferences and describe executed code only | Documentation specialist after accepted checkpoint evidence exists |
| Obsolete-path retirement and final validation | 10.1–11.3 | One active Example 21 remains, the public API is unchanged, strict OpenSpec checks pass, and review checks all public text and docstrings | Implementation specialist, then independent Reviewer, then Release Manager |

## Sequence and dependencies

Unchecked boxes below show required work, not verified Paperclip status. Update
them only from reviewed queue and release evidence.

- [ ] `1.1–1.3`: Establish the dependency and compatibility boundary.
- [ ] `2.1–5.3`: Build one direct implementation stream. Tasks in these groups
  depend on the compatibility boundary and share the same production and test
  files.
- [ ] `6.1–6.4`: Prove backend selection, time limits, coverage, and direct
  behavior. Structural or Dale work must not start before task 6.3 passes.
- [ ] `7.1–7.6`: Run structural work from an accepted proof checkpoint.
- [ ] `8.1–8.4`: Run Dale work from the same accepted untyped parent. Keep
  excitatory and inhibitory candidates separate.
- [ ] `9.1–9.3`: Produce evidence-bound plot and documents after an accepted
  checkpoint exists.
- [ ] `10.1–10.3`: Retire the obsolete path only after the replacement path is
  complete enough to satisfy its reference and command scans.
- [ ] `11.1–11.3`: Run final checks, then route to independent review and
  Rainbow release approval.

## Architecture-review dispositions

- **High value — accepted and sequenced:** Replace the large latent-workspace
  path with one direct BrainCell model. OpenSpec groups 1–6 implement and prove
  this baseline before optional work.
- **High value — accepted and sequenced:** Correct CSR semantics so rows are
  sources and columns are targets. Tasks 4.1, 7.1, and 7.5 carry direct checks.
- **High value — accepted as a gate:** Use exact task pass-at-1 only. Tasks
  3.1–3.4 and 6.3 keep loss evidence separate from ARC success.
- **High severity — accepted as a safety boundary:** Keep targets out of model
  events and keep validation forward-only. Tasks 2.2 and 5.3 test both rules.
- **Medium value — deferred until proof passes:** Structural pruning, growth,
  and Dale typing remain groups 7 and 8. They cannot run from an unproved
  baseline.
- **Low current value — deferred to backlog:** Chemical synapses, extra
  channels, compartments, morphology, neuromodulation, and persistent memory
  are separate future experiments. Task 8.4 implements only the guard and
  documentation.
- **Rejected for this change:** BPTT, synthetic qualification, copy or rule
  paths, candidate forests, reranking, partial scores, random regrowth, and
  random E/I ratios. Tasks 5.3 and 10.3 provide final scans and stop gates.

## Planning-hygiene finding

The repository roadmap
[`2026-07-25-algorithm-axes-roadmap.md`](2026-07-25-algorithm-axes-roadmap.md)
still says `design, awaiting review`, although phases P0 through P4 say done and
P5 says implemented out of tree. This is a medium-severity planning-state
conflict. Do not rewrite or archive that historical roadmap until the standard
queue confirms the reviewed release and Rainbow performs or authorizes release
gardening. Record a sequenced roadmap-gardening issue after that confirmation;
keep the BrainCell ARC OpenSpec change as the current execution authority.

## Dispatch and release checks

Before any Paperclip dispatch, verify that each engineering issue has one
implementation owner, an independent Revy review stage, `maxReviewRounds` set
to 50, the implementation owner as `returnAssignee`, and Rainbow
(`872b1fed-7368-42f0-8233-ebc1df38e158`) as the release approval participant.

For ARC-AGI work, also verify that the execution policy resolves to
`git_worktree`, base ref `origin/main`, and a non-primary checkout. Route
workspace repair and integration to Rainbow. Only Rainbow can finalize the
release.

## Audit verification

- The active capability delta has 19 requirements. The active OpenSpec
  checklist has 42 unchecked implementation and validation items.
- The checklist covers each requirement class and keeps the shared-file work in
  one ordered implementation stream.
- `git diff --check` passes for this audit document.
- Strict OpenSpec validation remains part of task 11.2. The `openspec` command
  is not installed in this execution workspace, so this audit does not claim
  that validation has passed.
- Live Paperclip assignment, execution-policy, and isolated-worktree checks
  must pass before dispatch. Repository documents cannot prove control-plane
  state.
