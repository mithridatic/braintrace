## 1. Feasibility spike (throwaway, `tmp/`, not released)

- [ ] 1.1 Build a minimal episode generator and a linear oracle readout; sweep symbol count and spike encoding until supported-query accuracy is at least 0.9 at two bindings and at most 0.6 at eight bindings. Record the chosen symbol count, encoding rate, and demonstration tick budget.
- [ ] 1.2 Run a latent population for `R` zero-input ticks across candidate membrane constants and `W_f` spectral radii; find a setting whose mean firing rate at `r = R` is at least a stated fraction of its rate at `r = 0`. Record the chosen constants.
- [ ] 1.3 Compile a two-state model skeleton through `braintrace.compile(..., braintrace.pp_prop, ...)` and confirm no control-flow or ETP diagnostic warnings are raised, and that the memory contraction is absorbed into the hidden-to-hidden transition as designed.
- [ ] 1.4 Write the spike's outcomes into `design.md` (D5 constants, D9 defaults) and, if the binding range moved, into the spec. Stop and report if any gate in 1.1 or 1.2 cannot be met.

## 2. Task module

- [ ] 2.1 Implement `latent_workspace_task.py`: per-episode bijection draw, demonstration pairs, held-out query, deterministic oracle, spike encoding of symbols and phase.
- [ ] 2.2 Implement the supported/short split producing matched episode pairs with byte-identical query-phase inputs and targets.
- [ ] 2.3 Implement binding-count control across the two-through-eight range, including the overflow condition when bindings exceed slot capacity.
- [ ] 2.4 Write `latent_workspace_task_test.py`: oracle agreement verified independently of the generator, rule variation across episodes, no rule leakage in inputs, byte-identical matched queries, short condition omits the queried binding, supported condition includes it exactly once, overflow raises or reports, malformed configuration raises naming the quantity, plus a hypothesis property test over symbol counts and binding counts.

## 3. Model module

- [ ] 3.1 Implement `latent_workspace_model.py` phase vector and arithmetic gating over one flat time axis; assert no Python loop drives the model and no inner `scan` consumes a `ParamState`.
- [ ] 3.2 Implement the ingestion population, the key/value projections as ETP `matmul` operations, and the slotted one-hot memory write into `brainstate.HiddenState` factors.
- [ ] 3.3 Implement the latent population, its zero-input recurrence, the memory read as a hidden-state contraction, and the linear readout.
- [ ] 3.4 Implement the shuffled-memory control as a column permutation preserving shape and magnitude.
- [ ] 3.5 Write `latent_workspace_model_test.py`: factored read equals the dense outer-product read (hypothesis property test — the correctness keystone), phase mask activates exactly one sub-map per tick, ingestion leaves every `ParamState` bitwise identical, differing demonstrations produce differing memory, memory storage scales with slots not with the square of the width, shuffle preserves shape and magnitude, `R = 0` is well-formed.

## 4. Analysis module

- [ ] 4.1 Implement `latent_workspace_analysis.py`: participation ratio and step-to-step trajectory norm per latent iteration.
- [ ] 4.2 Implement linear probes with an explicit disjoint fit/score split, for the answer from each `H_r`, for the answer from the memory read at the query encoding, and separately for the full rule.
- [ ] 4.3 Implement the comparison line that states plainly when memory-only decodability matches or exceeds final-workspace decodability.
- [ ] 4.4 Write `latent_workspace_analysis_test.py`: participation ratio on inputs of known rank, trajectory norm on a constructed fixed point and a constructed divergence, probe fit/score sets provably disjoint with no leakage, answer and rule probes reported separately, null-separation line fires on constructed data, mismatched leading dimensions raise naming the shapes.

## 5. Entry point

- [ ] 5.1 Implement `21-latent-reasoning-in-context.py`: CLI, seeded configuration, training per latent depth in `{0, 1, 2, 4, 8}` on a mixed binding-count distribution.
- [ ] 5.2 Implement the frozen-model intervention grid over binding count, supported versus short context, and intact versus shuffled memory, with no retraining.
- [ ] 5.3 Implement the plain-English report: per-depth accuracy, per-binding-count accuracy, the supported-versus-short contrast, the control arms, the four geometry measurements, probe split counts, and the claim-boundary paragraph.
- [ ] 5.4 Implement the Agg PNG: accuracy versus latent depth, accuracy versus binding count under both context conditions, and the per-iteration decodability curve.
- [ ] 5.5 Implement `--smoke` exercising every phase, arm, and reported measurement at reduced size.
- [ ] 5.6 Write `21-latent-reasoning-in-context_test.py`: smoke entry point returns the documented result mapping, same seed reproduces reported metrics within tolerance, every configured depth and binding count appears in the report, both control arms appear, and no test in the file asserts anything about the gradient estimate.

## 6. Documentation and release

- [ ] 6.1 Update `docs/specs/2026-08-16-pp-prop-latent-reasoning.md` (written during planning) with the spike's measured constants and any reporting detail that moved during implementation.
- [ ] 6.2 Add the Example 21 catalog row and the two axis-map rows to `examples/pp_prop/README.md`.
- [ ] 6.3 Run the focused example tests and the repository's normal example gate; record the results.
- [ ] 6.4 Confirm no scratch or spike artifact from Task 1 is tracked for release, and that the branch is clean and pushed.
