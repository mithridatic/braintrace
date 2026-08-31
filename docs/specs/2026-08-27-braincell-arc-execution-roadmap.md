# BrainCell ARC execution roadmap

Status: active

Updated: 2026-08-27

Execution authority: [`replace-example21-with-braincell-arc`](../../openspec/changes/replace-example21-with-braincell-arc/README.md)

## Scope and planning rules

This roadmap contains only active and upcoming work for the BrainCell ARC
replacement. The OpenSpec proposal, design, capability specification, and task
list define the product requirements. This roadmap defines sequence, ownership,
and handoffs. It does not change a requirement or a frozen interface.

Every engineering issue must use the standard execution policy:

1. The implementation specialist owns implementation and every review return.
2. Revy performs independent review. `maxReviewRounds` is `50`.
3. Rainbow (`872b1fed-7368-42f0-8233-ebc1df38e158`) performs release approval.
4. Rejected work returns to the implementation specialist. Approved work goes
   to Rainbow. The board is not an execution-policy participant.
5. Before dispatch, the ARC-AGI workspace policy must resolve to
   `git_worktree`, `origin/main`, and a non-primary checkout. Rainbow owns
   workspace repair and integration.

## Requirements audit

- The active replacement change has 42 open checklist items. BCA-01 through
  BCA-09 map every item. Tasks 9.2 and 9.3 have one implementation owner in
  BCA-05; BCA-08 performs their final implementation-truth verification and
  does not create a second work stream.
- The superseded latent-reasoning change has 37 completed checklist items and
  no open item. It stays historical until BCA-06 removes the obsolete active
  path.
- No parallel engineering batch is approved before BCA-DG-01 passes. The
  remaining phases are sequential because they change or validate the same
  Example 21 module, co-located test, evidence, commands, and documents.

## Dispatch gate

Dispatch status: blocked. No BrainCell ARC engineering issue may enter the
standard task queue until BCA-DG-01 passes.

- [ ] **BCA-DG-01 — Restore the required execution base**
  - Dependency: none.
  - Assigned specialist: Rainbow
    (`872b1fed-7368-42f0-8233-ebc1df38e158`).
  - Finding: the execution checkout is a non-primary `git_worktree`, but
    `refs/remotes/origin/main` does not exist. The required base therefore does
    not resolve.
  - Success: Rainbow repairs the workspace or integration state and verifies
    that the execution policy resolves to `git_worktree`, `origin/main`, and a
    non-primary checkout. Engineering dispatch can start only after this gate
    passes.

## Active sequence

- [ ] **BCA-01 — Runtime and compatibility foundation**
  - OpenSpec tasks: 1.1–1.3.
  - Dependencies: BCA-DG-01.
  - Assigned specialist: BrainCell compatibility implementation specialist.
  - Success: BrainCell 0.1.0 is pinned in every declared environment; clean
    development and image imports pass; the Hodgkin-Huxley, unit-boundary,
    PP-Prop relation, centered finite-difference, direct-readout, and spike-path
    fixtures pass without BPTT.

- [ ] **BCA-02 — Direct ARC data and output contracts**
  - OpenSpec tasks: 2.1–2.3 and 3.1–3.5.
  - Dependencies: BCA-01.
  - Assigned specialist: Example 21 implementation specialist.
  - Success: the named raw ARC files load without an index; the 705 by 441
    target-free event contract round-trips exactly; request losses, the direct
    360-value decoder, zero-tolerance strict scoring, the 256 KiB atomic result,
    and the 32 MiB immutable checkpoint contract pass their co-located tests.

- [ ] **BCA-03 — Sparse BrainCell model and PP-Prop training**
  - OpenSpec tasks: 4.1–4.5 and 5.1–5.3.
  - Dependencies: BCA-02.
  - Assigned specialist: Example 21 implementation specialist.
  - Success: the model has 2,048 neurons, 14,112 sparse input connections, and
    16,384 sparse non-self recurrent connections; repeated neural work uses
    BrainState transforms; reset, false-advance, integration, compiler-relation,
    optimizer, proof-schedule, and no-target gates pass.

- [ ] **BCA-04 — Bounded baseline proof**
  - OpenSpec tasks: 6.1–6.4.
  - Dependencies: BCA-03.
  - Assigned specialist: performance and evidence specialist.
  - Success: matched backend selection is recorded; every warmed decoder call is
    at most 100 ms; the real-data proof completes in at most 180 seconds and
    changes a recurrent weight and a direct prediction; the co-located test
    module exceeds 90% meaningful coverage and completes in at most 60 seconds.
  - Gate: structural, Dale, documentation-truth, and retirement work does not
    claim an accepted baseline until this issue passes review.

## Upcoming work after the baseline proof

The remaining work is sequential. The approved migration plan first records
the executed baseline in the causal and system-model documents. It then retires
the old path before structural and Dale stages start. The structural stream
changes the new Example 21 module and sibling test. The final plot and document
review follow the structural evidence, so no accepted claim runs ahead of
executed code.

- [ ] **BCA-05 — Record the executed baseline**
  - OpenSpec tasks: 9.2–9.3.
  - Dependencies: BCA-04.
  - Assigned specialist: technical documentation specialist.
  - Success: the causal and system-model documents use ASD-STE100 Simplified
    Technical English, separate observations from inferences, include direct
    prediction and target data, describe only executed code, and make no ARC
    ability claim from loss or state change when strict pass-at-1 is zero.

- [ ] **BCA-06 — Retire the obsolete Example 21 path**
  - OpenSpec tasks: 10.1–10.3.
  - Dependencies: BCA-05.
  - Assigned specialist: migration implementation specialist.
  - Success: the BrainCell command is the only active Example 21 command; old
    latent-workspace entry points, modules, tests, diagnostics, index builder,
    image index configuration, and active references are removed; raw ARC files
    remain; the public `braintrace` API is unchanged.
  - Sequence rationale: the OpenSpec migration plan places retirement after the
    baseline documents and before structural and Dale stages. That explicit
    migration order takes precedence over checklist section numbering.

- [ ] **BCA-07 — Measured structural and Dale stages**
  - OpenSpec tasks: 7.1–7.6 and 8.1–8.4.
  - Dependencies: BCA-06.
  - Assigned specialist: structural-plasticity implementation specialist.
  - Success: each pruning, addition, compaction, excitatory, and inhibitory arm
    uses the declared 5% count and measured evidence; no dense neuron-pair array
    or random primary selection is used; accepted arms produce at least one
    false-to-true strict change with no regression and finish in at most 300
    seconds; deferred biology remains disabled.

- [ ] **BCA-08 — Plot and synchronize implementation-truth documents**
  - OpenSpec tasks: 9.1, with final verification of 9.2–9.3.
  - Dependencies: BCA-07.
  - Assigned specialist: technical documentation specialist.
  - Success: the explicit plot preserves predictions and reports executed
    counts; the causal and system-model documents use ASD-STE100 Simplified
    Technical English, separate observations from inferences, include direct
    prediction and target data, and describe only executed code.

- [ ] **BCA-09 — Integrated validation, independent review, and release handoff**
  - OpenSpec tasks: 11.1–11.3.
  - Dependencies: BCA-06, BCA-07, and BCA-08.
  - Assigned specialist: Example 21 implementation specialist; independent
    review by Revy; release approval by Rainbow.
  - Success: focused tests, coverage, proof, decoder, ordinary-run, strict
    OpenSpec, and `git diff --check` gates pass; public callables and messages
    pass docstring and Simplified Technical English review; no unrelated
    worktree changes remain; Revy approves before Rainbow receives the release.

## Backlog

- **BCA-B01 — Retire algorithm-roadmap finding F-22.** Run the required
  multi-population SNN bias comparison in the external benchmark repository.
  This is separate from the BrainCell ARC replacement.
- **BCA-B02 — Complete the external benchmark adversarial review.** Review
  vacuity handling, ignored configuration fields, stub-sensitive assertions,
  and B4 protocol bias. Record each finding and disposition before scheduling a
  code change.
- **BCA-B03 — Optional biological mechanisms.** Consider AMPA, GABAa, HCN,
  calcium-dependent adaptation, NMDA, electrical junctions, morphology,
  neuromodulation, or persistent memory only as separate one-feature measured
  changes after BCA-09.

## Completed

- **2026-08-24 — Legacy latent-reasoning Example 21 research and replacement
  specification integrated.** The `add-latent-reasoning-example` checklist is
  complete. Its implementation is superseded by the active BrainCell ARC
  replacement and remains historical until BCA-06 removes the active path.
- **2026-07-25 through 2026-07-26 — Algorithm-axis phases P0–P4 completed and
  P5 implemented out of tree.** The two unfinished P5 items are BCA-B01 and
  BCA-B02; completed phase detail remains in the historical algorithm-axis
  roadmap rather than the active sequence above.

## Architecture-review dispositions

- **Keep and sequence — incorrect CSR endpoint interpretation in Example 20.**
  Severity: high. Value: high. BCA-07 must use CSR rows as sources and columns
  as targets, with hand-calculated tests. This prevents wrong contribution and
  connection-selection evidence.
- **Keep as a hard gate — old Example 21 is large, slow, and indirect.**
  Severity: high. Value: high. BCA-01 through BCA-06 replace it with the direct,
  bounded path before later biology is promoted.
- **Defer — optional biological detail.** Severity: low for the baseline.
  Value: unknown. BCA-B03 holds this work until one accepted direct baseline
  exists and a separate measured change is approved.
- **Do not create a task — random E/I assignment or random regrowth.** These
  conflict with the approved measured-selection requirements and have no place
  in the active sequence.
