## 1. Specification reset

- [x] 1.1 Replace the symbol-lookup OpenSpec proposal, design, and capability delta with the standard-ARC, one-model variable-effort contract.
- [x] 1.2 Replace `docs/specs/2026-08-16-pp-prop-latent-reasoning.md` before implementation, including data provenance, exact scoring, 2,048/16,384 scale, controls, and claim boundaries.
- [x] 1.3 Run strict OpenSpec validation and resolve every structural or semantic error.

## 2. ARC data and provenance

- [x] 2.1 Replace `latent_workspace_task.py` with validated `ArcGrid`, `ArcPair`, `ArcTask`, query-episode, source-manifest, canonical-fingerprint, and split-leakage APIs.
- [x] 2.2 Support standard per-task JSON plus collection and JSONL adapters needed by configured public corpora, with source roles and rejection accounting.
- [x] 2.3 Implement lossless fixed-shape row-event encoding for variable grids, demonstrations, and multiple test queries without target or metadata leakage.
- [x] 2.4 Implement BrainState-random color, dihedral, and demonstration-order training augmentations that preserve task semantics and never touch evaluation data.
- [x] 2.5 Add co-located task tests for malformed grids/tasks, round-trip encoding, multi-query identity, canonical deduplication, split leakage, provenance, augmentation consistency, padding, and target-leakage resistance.

## 3. Exact scoring and trajectory analysis

- [x] 3.1 Replace `latent_workspace_analysis.py` with output-logit validation, deterministic pass@1/pass@2 candidate decoding, and exact query and strict task metrics.
- [x] 3.2 Add clearly labelled shape and valid-cell pixel diagnostics that cannot satisfy exact success.
- [x] 3.3 Add per-step provisional-grid changes, entropy/margin, spike/rate, voltage, displacement, convergence, saturation, and silence summaries.
- [x] 3.4 Add intact/control trajectory comparison including a byte-identical causally-null determination.
- [x] 3.5 Add co-located analysis tests covering one-cell failures, wrong shapes, second-candidate success, multi-query strictness, constructed fixed/saturated/silent trajectories, malformed logits, and null controls.

## 4. Recurrent spiking model

- [x] 4.1 Replace `latent_workspace_model.py` with an Example-18-style BrainPy LIF network using BrainTrace `Linear`/`SparseLinear`, `AlignPostProj`, `Expon`, and `CUBA` components.
- [x] 4.2 Construct a deterministic no-self-edge sparse topology with exactly 16,384 edges for the 2,048-neuron full configuration using `brainstate.random`.
- [x] 4.3 Implement compiled context and zero-input latent rollouts with `brainstate.transform.for_loop`, exposing query-terminal state and the complete 32-step trajectory.
- [x] 4.4 Implement the low-rank height, width, and 30×30×10 color readout plus pp-prop `ETraceConfig` and terminal loss integration.
- [x] 4.5 Implement exact state reset/snapshot/restore and deterministic 64-neuron slot ablation without parameter mutation.
- [x] 4.6 Add co-located model tests for physical scale, exact edge count, component types, deterministic topology, zero latent input, state/parameter separation, reset reproducibility, checkpoint semantics, and ablation bounds.

## 5. Training, evaluation, and reports

- [x] 5.1 Replace `21-latent-reasoning-in-context.py` with CLI/configuration, manifest loading, split checks, one-model pp-prop optimization, and shared optimizer state across 8/16/32 effort updates.
- [x] 5.2 Evaluate one frozen 32-step trajectory at 0/8/16/32 on byte-identical tasks, aggregate exact metrics over all queries and tasks, and retain per-query repeat-intact reproducibility evidence.
- [x] 5.3 Implement no-context, deranged-demonstration, truncation, and slot-ablation arms without retraining; require the slot-ablation checkpoint-0 state, candidates, and exact metrics to match before attributing later differences to the intervention.
- [x] 5.4 Emit a machine-readable result and plain-English report containing configuration, device, counts, provenance, exact scores, diagnostics, trajectories, controls, runtime, numerical reproducibility with literal byte identity reported separately, and claim boundary.
- [x] 5.5 Emit an Agg plot of exact quality versus effort, trajectory dynamics, spike/voltage behavior, and control deltas.
- [x] 5.6 Add a plumbing-only `--smoke` path and co-located entry-point tests covering every effort checkpoint and control without treating fixture scores as scientific evidence.

## 6. Documentation and qualification

- [x] 6.1 Update the Example 21 README catalog and axis-map rows so they describe standard ARC, one-model effort, exact scoring, and public-data requirements.
- [x] 6.2 Run focused tests with more than 90 percent meaningful coverage of changed production modules, then the repository's normal example gate.
- [x] 6.3 Run a full GPU structural qualification proving 2,048 neurons and exactly 16,384 edges; run training/evaluation qualification only when an approved non-evaluation public corpus is present and the repeat/ablation reproducibility gates pass for every evaluated query.
- [x] 6.4 Record exactly what was and was not empirically qualified, verify downloaded/generated artifacts are untracked, commit the completed worktree branch, and leave `main` unchanged.
