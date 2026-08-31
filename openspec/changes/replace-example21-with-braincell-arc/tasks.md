## Execution map

This file is the execution authority for the change. Paperclip issues must
reference these task identifiers. They must not copy this checklist into a
second queue.

### Active

- [ ] **E21-01 — Workspace repair.** Dependencies: none. Assigned specialist:
  Rainbow, Release Manager. Success: the isolated execution policy reports
  `git_worktree`, base `origin/main`, and a non-primary checkout. The current
  checkout is non-primary, but `origin/main` does not resolve. Do not dispatch
  implementation until this gate passes.

### Upcoming

- [ ] **E21-02 — Baseline implementation.** Dependencies: E21-01. Assigned
  specialist: one BrainCell and BrainTrace implementation agent. Scope: tasks
  1.1–6.4. Success: all temporary-proof, finite-difference, strict-score,
  coverage, decoder, and runtime gates in sections 1–6 pass. Keep one owner
  because this work changes `21-braincell-arc.py` and its sibling test together.
- [ ] **E21-03 — Structural stages.** Dependencies: E21-02. Assigned
  specialist: the E21-02 implementation agent. Scope: tasks 7.1–7.6. Success:
  exact five-percent pruning and addition arms use observed evidence, preserve
  strict results, physically compact accepted topology, and finish within the
  declared run limit.
- [ ] **E21-04 — Dale stages.** Dependencies: E21-03. Assigned specialist: the
  E21-02 implementation agent. Scope: tasks 8.1–8.4. Success: separate measured
  excitatory and inhibitory candidates keep effective signs valid; deferred
  biology remains disabled.
- [ ] **E21-05 — Evidence and retirement.** Dependencies: E21-02; E21-03 and
  E21-04 for claims about accepted structural or Dale stages. Assigned
  specialist: the E21-02 implementation agent. Scope: tasks 9.1–10.3. Success:
  implementation-truth documents describe only executed behavior, the optional
  plot is non-mutating, and only the new Example 21 executable remains active.
- [ ] **E21-06 — Independent review.** Dependencies: E21-05. Assigned
  specialist: Revy, independent Reviewer. Success: requirements traceability,
  frozen-interface compliance, focused coverage above 90 percent, runtime
  limits, and OpenSpec validation pass. Rejected work returns to the E21-02
  implementation agent. The engineering issue must set `maxReviewRounds` to
  50.
- [ ] **E21-07 — Release approval and integration.** Dependencies: E21-06.
  Assigned specialist: Rainbow, Release Manager. Scope: tasks 11.1–11.3 and
  release integration. Success: all final gates pass on the reviewed revision;
  Rainbow approves and finalizes the release. No other role may merge or mark
  the release complete.

### Requirements audit

The 42 implementation tasks cover all 19 requirements and 59 scenarios in
`specs/pp-prop-braincell-arc/spec.md`. No requirement gap needs a new task.
The main delivery risk is ownership overlap: sections 1–10 converge on one
executable and one sibling test module, so they are one serial implementation
lane. Dependency packaging may proceed in a separate worktree only after
E21-01, because its files do not overlap the core module. Documentation can
start only after executed evidence exists. Ideas found during implementation
belong in the backlog and do not expand this frozen change.

Architecture-review disposition: the algorithm-axes roadmap records open P5
independent-review debt. Severity is medium and value is high, but it concerns
the separate out-of-tree benchmark suite and does not block this Example 21
change. Keep it in the standard Paperclip backlog after the control plane is
available; do not add it to this execution map. The missing `origin/main`
dispatch base is a critical operational finding and is sequenced first as
E21-01 because every implementation issue depends on it.

## 1. Runtime and compatibility

- [ ] 1.1 Add `braincell==0.1.0` to the Example 21 image, development
  requirements, and applicable project extras, and keep the raw ARC copy;
  verify that a clean development install and image build import BrainCell,
  BrainState, BrainUnit, JAX, and BrainTrace.
- [ ] 1.2 Add the small Hodgkin-Huxley and PP-Prop compatibility fixtures to
  `21-braincell-arc_test.py`; verify finite `mA/cm²` execution, the expected
  current-unit error, every declared BrainCell 0.1.0 constructor value, reset
  gate values, compiler relations for input and recurrent weights, and finite
  direct readout gradients.
- [ ] 1.3 Add the one-step centered finite-difference fixture and spike-path
  fixture; verify the custom 1 by 4 CSR, separate reset copies, declared
  objective and derivative tolerance, the deterministic threshold crossing,
  finite gradients, and at least one nonzero spike-path gradient without a
  BPTT call.

## 2. ARC data and events

- [ ] 2.1 Implement the direct raw-file ARC loader and its integer-grid contract;
  verify both practice-role and evaluation-role fixtures, all invalid grid edge
  cases, a corrective error above ten demonstrations, direct task identifiers,
  and ordinary rejection of evaluation data.
- [ ] 2.2 Implement the fixed 705 by 441 event encoder and Boolean advance mask;
  verify exact round trips at minimum and 30 by 30 sizes, byte stability,
  event-specific one-hot and all-zero fields, all-zero padding, fixed request
  positions, and byte-identical inference input after target-only mutation.
- [ ] 2.3 Add the fixed eight-task training order, four-task validation order,
  and two-task proof order; verify that direct paths resolve only the declared
  files and that no index, hash, fingerprint, or synthetic-task path runs.

## 3. Exact loss, prediction, and result

- [ ] 3.1 Implement request-masked shape and row loss; verify two shape terms,
  all 900 maximum-grid cell terms, zero non-request loss, and changed loss for
  each changed strict datum when its class logits differ.
- [ ] 3.2 Implement the 360-value direct voltage readout and integer decoder;
  verify independent height and width argmax, all row and column slices,
  output dimensions 1 through 30, colors 0 through 9, integer dtype, and no
  query-input residual or target input.
- [ ] 3.3 Implement zero-tolerance query exactness and strict task pass-at-1;
  verify that one wrong dimension, cell, non-integer dtype, or query makes the
  applicable exact Boolean false, that integer width does not affect exactness,
  and that no partial or average score is calculated.
- [ ] 3.4 Implement the exact small `result.json` schema and atomic writer;
  verify JSON integer cells, reproducible Booleans, rejected extra fields,
  clear oversized-record errors, and encoded size at or below 256 KiB.
- [ ] 3.5 Implement the compressed array-only checkpoint writer and loader;
  verify format value 1, `allow_pickle=False`, every declared array and code,
  three optimizer step counts, exact round trip, reset runtime state, invalid
  data rejection, 32 MiB limit, atomic write, distinct child path, and
  byte-unchanged parent after child failure.

## 4. Sparse BrainCell model

- [ ] 4.1 Implement deterministic sparse input and recurrent topology with
  `brainstate.random` weight initialization; verify 2,048 neurons, 14,112 input
  connections, 16,384 recurrent connections, no duplicate or self recurrent
  connection, CSR row-as-source semantics, and no dense input or recurrent
  parameter.
- [ ] 4.2 Implement the bounded input and prior-spike recurrent current-density
  paths and the exact single-layer BrainCell Hodgkin-Huxley cell declared in the
  design; verify every geometry, capacitance, threshold, initializer,
  surrogate, solver, ion, channel, conductance, reversal, temperature, shift,
  reset-gate, voltage, and spike value, and verify that a spike first affects
  recurrent current on the next event.
- [ ] 4.3 Implement false-advance state freezing with
  `brainstate.transform.cond` and the compiled event driver with
  `brainstate.transform.for_loop` and `jit`; verify bitwise state identity for
  biological and eligibility state, zero padded loss and gradient, identical
  final prediction with inserted padding, and no bare repeated Python loop.
- [ ] 4.4 Implement the matched integration check; verify one `0.1 ms` step
  against two compiled `0.05 ms` substeps per event, record maximum voltage and
  spike-event differences, and apply every finite, `1 mV`, spike, prediction,
  and strict-result selection gate.
- [ ] 4.5 Implement complete episode reset and direct state-dependent decoding;
  verify reset voltage, gates, spikes, and eligibility between queries, retained
  parameters and Adam state during training, and at least one changed proof
  prediction after the decoder-boundary state intervention.

## 5. PP-Prop training

- [ ] 5.1 Compile input and recurrent state-changing parameters with BrainTrace
  PP-Prop `single-step`; verify complete hidden-state relations and fail the
  proof for a missing or unintended non-temporal temporal parameter.
- [ ] 5.2 Implement one accumulated PP-Prop optimizer update per query episode
  with trace decay 0.95, norm clip 1.0, and Adam learning rates 0.001, 0.0003,
  and 0.003; verify exact update counts, task order, finite optimizer state, and
  no target value in a model event.
- [ ] 5.3 Implement the eight-update proof schedule and 64-update ordinary
  schedule; verify that all proof updates use `d631b094`, `46f33fce` is
  forward-only, and a failed gate stops without extra updates, task changes,
  BPTT, synthetic data, or a fallback answer system.

## 6. Backend, time, and temporary proof gates

- [ ] 6.1 Implement separate CPU and GPU backend-probe processes, synchronized
  median timing, finite-state validation, literal lower-median selection, and
  frozen backend selection; use one warmed plus three timed full 705-event
  PP-Prop gradient calls for the first `d631b094` query, with no optimizer
  update or file write; verify prediction-byte stability, one concise timing
  line, and no timing field in `result.json`.
- [ ] 6.2 Add the warmed decoder benchmark from executed request states; verify
  five direct readout and grid-construction calls for each fixed-validation
  query at or below 100 ms, record every call, and record neural request and
  full 705-event context-to-grid times separately.
- [ ] 6.3 Run the real-data temporary proof on `d631b094` and `46f33fce`;
  verify training-task-only updates, unchanged model state across validation,
  all required mechanism observations, a changed recurrent weight, a changed
  direct prediction, direct pretraining and post-training loss components,
  actual prediction and target grids, voltage-only, sodium-gates-only,
  potassium-gate-only, spikes-only, all-state, and null interventions, direct
  strict data, and an end-to-end time at or below 180 seconds.
- [ ] 6.4 Run the complete co-located Example 21 test module; verify more than
  90% meaningful coverage and a total time at or below 60 seconds.

## 7. Example 20 structural adaptation

- [ ] 7.1 Implement task-row normalization, neuron contribution, recurrent
  connection contribution, pre-clip `etrace_grad` gradient mass, stable ranking,
  task owners, and structural twins; verify hand-calculated fixtures, direct
  voltage-readout effect, mean-spike relay and transmission, maximum task
  protection, unowned zero scores, shared task ties, stable item ties, and the
  corrected CSR row-as-source transmission and outgoing strength.
- [ ] 7.2 Implement one five-percent neuron-pruning mask and one five-percent
  recurrent-connection-pruning mask; verify exact ceiling counts, zero-strict
  pruning block, incident connection removal, no optimizer update before the
  causal gate, and rejection of every strict regression.
- [ ] 7.3 Implement physical compaction and optimizer remapping; verify direct
  count changes, biological-connection ceiling, preserved surviving Adam data,
  reset eligibility, one intentional recompilation, and prediction bytes and
  strict Booleans identical to the accepted mask.
- [ ] 7.4 Implement five-percent structural-twin neuron addition; verify donor
  selection from measured task evidence, copied input and incoming wiring,
  split outgoing and readout values, inherited owner and Dale label, zero new
  moments, connected new neurons, and clear failure when valid donors or budget
  are insufficient.
- [ ] 7.5 Implement five-percent measured recurrent-connection addition; verify
  absent non-self pairs, mean-spike source and gradient-mass target ranking,
  stable 256 by 256 tiles, at most 65,536 resident pairs, global top-set and
  stop-bound correctness, no dense neuron-pair array, typed and untyped initial
  values, zero new moments, no random regrowth, and exact addition count.
- [ ] 7.6 Implement one-candidate-per-arm structural execution and the fixed 64
  PP-Prop updates for additions; verify promotion only for at least one direct
  false-to-true strict change with no true-to-false change and a complete arm
  time at or below 300 seconds.

## 8. Dale stages and optional biology

- [ ] 8.1 Implement separate five-percent excitatory and inhibitory selection
  from the same untyped parent; verify measured sign coherence, activity,
  pre-clip `etrace_grad` gradient mass, task ownership, lesion evidence, stable
  ties, and no random ratio.
- [ ] 8.2 Implement the differentiable sparse `weight_fn` for typed neurons
  by calling `braintrace.sparse_matmul` directly across training, addition,
  pruning, and compaction; verify the `1e-6` effective floor,
  inverse-softplus conversion, zero baseline types, exact candidate counts, no
  effective sign violation after one update and one structural operation, and
  raw signed behavior for untyped neurons.
- [ ] 8.3 Add the Dale candidate runner and strict gate; verify separate parent
  checkpoints, fixed 64-update arms, false-to-true promotion with no regression,
  and no AMPA or GABAa mechanism in a type-assignment arm.
- [ ] 8.4 Add only the guard and documentation for deferred biological
  features; verify that default construction activates none. Do not implement
  an optional mechanism in this change. A later one-feature experiment must
  reject a slow or strict-flat arm.

## 9. Plot and implementation-truth documents

- [ ] 9.1 Implement the explicit two-dimensional Matplotlib topology plot;
  verify checkpoint neuron and connection counts, owner and Dale groups, no
  ordinary-run plot, and identical prediction bytes before and after plotting.
- [ ] 9.2 Write the causal explanation in
  `docs/specs/2026-08-24-example21-causal-explanation.md`; verify ASD-STE100
  Simplified Technical English, separate observations and inferences, and
  actual prediction and target data for each measured claim.
- [ ] 9.3 Write the executed system model in
  `docs/specs/2026-08-24-example21-system-model.md`; verify that it describes
  only code that runs and uses BrainCell, BrainState, BrainTrace, neuron,
  connection, layer, Dale-type, model-cell, prediction, and output-shape terms
  consistently.

## 10. Retire the obsolete path

- [ ] 10.1 Replace the active Example 21 README and image command with the
  BrainCell command; verify that the documented proof and run commands resolve
  to `21-braincell-arc.py` and the image retains the named raw ARC files.
- [ ] 10.2 Remove the old Example 21 entry point, `latent_workspace*` production
  and test modules, specifically including `21-arc-agi-latent-reasoning.py`,
  `21-latent-reasoning-in-context.py`, its sibling test, and
  `docs/diagnostics/example21_*.py`; remove the ARC index builder and test,
  generated index command, obsolete Docker arguments and labels, and index or
  source-manifest environment values; verify that no active command or import
  references a removed module.
- [ ] 10.3 Run import and reference scans after removal; verify no change to the
  public `braintrace` API and no synthetic, BPTT, copy, rule, candidate, forest,
  reranker, partial-score, average-score, or large-result path in the new
  executable.

## 11. Final validation

- [ ] 11.1 Run the focused Example 21 tests, coverage gate, three-minute proof,
  warmed decoder gate, and one five-minute ordinary run; verify each literal
  limit and report the direct strict count before any diagnostic observation.
- [ ] 11.2 Run `openspec validate replace-example21-with-braincell-arc --type
  change --strict --no-interactive`, `openspec validate --all --strict
  --no-interactive`, and `git diff --check`; verify that all commands pass on the
  worktree branch.
- [ ] 11.3 Review every new public callable and user-facing message; verify
  NumPy-style docstrings, ASD-STE100 Simplified Technical English, sentence-case
  concise errors with corrective actions, descriptive names, no unnecessary
  code comments, and no unrelated worktree changes.
