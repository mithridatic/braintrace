# H01 human-reference topographic search

Evidence snapshot: **2026-09-09**, base commit **34ca482**. This is a diagnostic map, not a new campaign or a human-validation claim.

[Open the complete Flying Logic tree](h01-human-reference-search.xlogic). [Structured question records](h01-human-reference-search.json). [Verification](h01-human-reference-search-verification.md).

## Reading the tree

Start with the response you want to explain. The six overview links are navigation: E, I, transfer and circuit problems can coexist. Within each diagnostic subtree, each diamond is one scoped binary question. Its Yes and No outcomes are complementary. A gray outcome is conditional unless marked **recorded**. A question may have no answered outcome.

| Node status | Meaning |
| --- | --- |
| YES / NO | Answer to this predicate under its stated conditions; not a global qualification verdict. |
| OPEN | More evidence is needed. |
| INCONCLUSIVE | An attempt did not yield a discriminating answer. |
| BLOCKED | A prerequisite or closed/stopped campaign prevents the proposed test. |

Every question preserves response coordinates, evidence and the next discriminating check. Both conditional outcomes remain visible. A failed tested dose excludes that dose as a sufficient correction, not every possible mechanism in its family. Interactions and untested mechanisms remain in the search space.

## Contract and boundaries

Exact count; 1 ms timing and recovery delays; 1 mV voltage; 0.05 ms spike phases. No trace alignment or pooled override. Usable-tier separate.

Usable-tier limits remain rate 15%, adaptation 25%, per-cycle width 20%, and AHP 2 mV, with their own resolvability checks. A usable pass never replaces the validation-grade contract. Missing events, cut-off recovery windows, and unavailable mappings remain explicit.

Released/spent: I 0.23 nA; E sweeps 53 and 55. Still sealed in this snapshot: E sweep 54 (330 pA), I Noise1 sweep 48. No holdout opened by this artifact.

No simulations, fitting, morphology edits, or revival of stopped campaigns. Open/blocked leaves require separately registered and bounded experiments.

H01 has no matching recorded voltage trace for these imported cells. Direct donor-response validation, numerical qualification, anatomy-supported construction, H01 prediction and network function are separate claims. A 1 ms finite runtime is not a human physiological protocol.

### Current priority

1. Select the failing response and check its input/measurement mapping (R02) and numerical resolution (R03).
2. For E gain, retain B3's measured excess and the failed tested doses; the combined intervention remains unobserved (E01-E05).
3. For I, keep the spent reserve failure and localize post-trough current balance before another parameter choice (I01-I03).
4. For transfer, do not conflate the amended short-window closure with the failed strict full-train check or unresolved 0.19 nA timestep (T01-T04).
5. For circuit function, obtain a discriminating baseline before delivery and placement conclusions (C04-C07); the 300 ms experiments were stopped.
6. For population scale, resolve the isolated importer failure before larger initialization/runs (P01-P02). New donors still need source-response reproduction (P04-P05).

## Method references

Book source folder: `C:/Users/J/Documents/Diagnosing Performance and Reliability/pages-md`. References use OCR file numbers, not printed pagination. The supplied _distill notes were used to interpret the method.

- **p014.md** — Topographic diagnosis asks what is happening; do not begin with a root-cause label.

- **p018.md-p021.md** — Binary questions cover all remaining possibilities and are quick, unambiguous and cheap; do not repeat the same split consecutively.

- **p035.md, p041.md** — Retain the search as an explicit tree; classify the observation before selecting the decomposition.

- **p144.md** — Functional Isolation: split the serial path and ask at which boundary the contrast remains.

- **p182.md-p183.md, p188.md, p192.md-p194.md** — Structural Dissection: retain the complementary subsystem, interactions, and each written split.

- **p022.md, p074.md** — Distinguish physical/mathematical constraints from man-made requirements. Use voltage/current and charge balance at electrical boundaries; no absolute-truth label for assumptions.

The book's binary-answer premise is not an assertion of absolute certainty. Engineering tolerances are requirements; physical conservation is a constraint; profile assignments and synaptic parameters are assumptions. Test each claim at its stated scope.

## Subtrees

- [Reference and measurement](#r-reference-and-measurement) — [Focused Flying Logic view](h01-human-reference-search/r.xlogic)

- [E donor: gain and waveform](#e-e-donor-gain-and-waveform) — [Focused Flying Logic view](h01-human-reference-search/e.xlogic)

- [I donor: refiring and accommodation](#i-i-donor-refiring-and-accommodation) — [Focused Flying Logic view](h01-human-reference-search/i.xlogic)

- [Simulator and H01 transfer](#t-simulator-and-h01-transfer) — [Focused Flying Logic view](h01-human-reference-search/t.xlogic)

- [Measured contacts and circuit function](#c-measured-contacts-and-circuit-function) — [Focused Flying Logic view](h01-human-reference-search/c.xlogic)

- [Population execution and reference coverage](#p-population-execution-and-reference-coverage) — [Focused Flying Logic view](h01-human-reference-search/p.xlogic)

## R. Reference and measurement

A comparison can be mis-specified or unresolvable before any mechanism is at issue.

### R01 — Matching H01 voltage recordings exist?

**Status: NO** · Reference boundary

**Scope / coordinates:** Imported H01 anatomical cells versus the human donor recordings used to constrain their electrical profiles.

**Remaining space:** Direct validation against these H01 cells, or prediction constrained by recordings from different human cells.

**Question:** A voltage recording is matched by provenance to each imported H01 cell being claimed as directly validated.

**Decision limit:** Exact cell identity and recording provenance; no substitution of donor identity.

**Evidence says:** H01 supplies anatomy and annotations, not voltage recordings matched to the imported cells.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Direct H01-response comparison is available. | Leaf; see next check |
| No | Use donor-response validation and label H01 transfer a prediction. | Leaf; see next check |

**Next check:** Retain donor and H01 identities separately in every response record; do not invent an H01 voltage target.

**Still unresolved after either answer:** Neither anatomical identity nor a successful donor fit establishes population-wide human accuracy.

**Sources:** [2026-09-05-h01-recorded-response-acceptance-proposal.md](../specs/2026-09-05-h01-recorded-response-acceptance-proposal.md), [h01-causal-model.md](../h01-causal-model.md).

### R02 — Input and measurement mappings complete?

**Status: OPEN** · Hierarchical / elemental

**Scope / coordinates:** One proposed comparison at a time: donor, sweep, command waveform, bias convention, initial state, temperature, voltage correction, detector and full observation window.

**Remaining space:** Comparisons with every required mapping explicitly pinned versus comparisons with at least one missing or inconsistent mapping.

**Question:** All mappings in the scope agree with the selected raw record and candidate manifest.

**Decision limit:** Exact provenance/convention agreement. Per-event contract: exact count; 1 ms crossing/peak/delay; 1 mV voltage; 0.05 ms phases. No alignment.

**Evidence says:** Later I forensics use command-only input; E gain uses recorded command plus recorded bias. Earlier I bias proposals and older holdout statements are superseded.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Proceed to the per-row resolution check. | Leaf; see next check |
| No | Resolve the specific mapping before interpreting a residual. | Leaf; see next check |

**Next check:** Audit the selected comparison against its current manifest and raw-record metadata; label unavailable items. Do not rerun or open data.

**Still unresolved after either answer:** Complete metadata is not a numerical or physiological pass.

**Sources:** [h01-i-energetic-result.md](../evidence/h01-i-energetic-result.md), [h01-e-gain-manifest.json](../evidence/h01-e-gain-manifest.json), [2026-09-05-h01-recorded-response-acceptance-proposal.md](../specs/2026-09-05-h01-recorded-response-acceptance-proposal.md).

### R03 — Is this response row resolvable?

**Status: OPEN** · Hierarchical / measurement

**Scope / coordinates:** Each event/input/phase separately, at the same frozen candidate, mesh and input conventions.

**Remaining space:** Rows with uncertainty below the applicable decision requirement versus rows whose uncertainty prevents that decision.

**Question:** Measurement and numerical evidence resolve the proposed comparison under its registered limit.

**Decision limit:** Use the actual registered per-row limit; usable-tier human spread must be below half its tolerance. Never borrow a limit from another row or window.

**Evidence says:** No global resolution verdict is justified: T03/T04 have different timestep outcomes, and interrupted circuit arms have no halving pair.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Interpret the residual and select the appropriate behavior subtree. | Leaf; see next check |
| No | Park this row as unresolvable; improve its measurement or numerics first. | Leaf; see next check |

**Next check:** Read retained repeat/refinement evidence for the selected row; state the missing measurement if absent.

**Still unresolved after either answer:** A resolvable row can still fail the contract; an unresolved row cannot pass by a small pooled score.

**Sources:** [2026-09-07-h01-usable-circuit-plan.md](../specs/2026-09-07-h01-usable-circuit-plan.md), [sp2-close-decision.json](../evidence/h01-i-transfer/sp2-close-decision.json), [decision.json](../evidence/h01-ie-inhibition/decision.json).

## E. E donor: gain and waveform

Frozen B3 overfires at low drive; higher-drive count agreement does not settle phase width or return.

### E01 — B3 has excess low-drive spikes?

**Status: YES** · Hierarchical / input and cyclical

**Scope / coordinates:** B3 g0-b3; Allen 541563728; 200/250/310 pA; recorded bias; nseg x9; CVode 1e-10; 2100 ms.

**Remaining space:** Low-drive exact-count agreement versus at least one excess-count row at 200/250 pA.

**Question:** At least one of the 200/250 pA model counts exceeds its matched human count.

**Decision limit:** Exact counts: human 1/5 at 200/250 pA; the 310 pA control is 10.

**Evidence says:** B3 gives 4/8/10 versus human 1/5/10 at 200/250/310 pA.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Localize the low-drive excess while retaining the 310 pA control. | E02 |
| No | Low-drive excess is absent; inspect other response failures independently. | Leaf; see next check |

**Next check:** Use E02-E05 to retain tested exclusions; no additional evaluation is authorized.

**Still unresolved after either answer:** Correct count at 310 pA is neither shared-input nor waveform validation.

**Sources:** [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json).

### E02 — Initiation is axon-first?

**Status: YES** · Structural dissection / initiation

**Scope / coordinates:** B3 at active SP3 inputs, including sweeps 50/53/56; soma and axon threshold crossings.

**Remaining space:** Axon-first versus simultaneous or soma-first initiation under the recorded detector.

**Question:** The recorded axon crossing precedes the soma crossing at the active inputs in scope.

**Decision limit:** Strict ordering of the recorded crossings; retain measured lead and numerical resolution from stage-g0, not a new threshold.

**Evidence says:** Stage-g0 and closure report axon-first initiation; it persists in the tested passive interventions.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain the initiation site and examine the tested passive levers. | E03 |
| No | Localize initiation before interpreting passive changes as gain corrections. | Leaf; see next check |

**Next check:** Keep B3 axonal initiation fixed when formulating the next bounded gain test.

**Still unresolved after either answer:** Axon-first initiation does not establish the right count, width, or H01 region map.

**Sources:** [stage-g0-decision.json](../evidence/h01-e-gain/stage-g0-decision.json), [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json).

### E03 — Ih half meets the registered gain bands?

**Status: NO** · Structural dissection / reversible intervention

**Scope / coordinates:** g1-ih-half versus B3, distributed Ih factor 75 to 37.5; SP3 stage G; same recorded inputs and controls.

**Remaining space:** That tested intervention meets every registered gain/preservation band or fails at least one.

**Question:** g1-ih-half satisfies all stage-G gain and preservation clauses.

**Decision limit:** Use stage-g-decision arms/g1-ih-half/bands and rejection. The 250 pA count band is 5-7.

**Evidence says:** 250 pA count remains 8; rest shifts -1.73 mV. The registered intervention is rejected.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain this intervention as a candidate for independent validation. | Leaf; see next check |
| No | Exclude this Ih dose as a sufficient registered correction. | E04 |

**Next check:** Carry the negative result to the next distinct tested split; do not repeat it.

**Still unresolved after either answer:** Other Ih doses and interactions are not exhaustively excluded.

**Sources:** [stage-g-decision.json](../evidence/h01-e-gain/stage-g-decision.json), [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json).

### E04 — Leak x1.5 meets the gain bands?

**Status: NO** · Structural dissection / reversible intervention

**Scope / coordinates:** g2-leak-150 versus B3; all registered regions; SP3 stage G.

**Remaining space:** That tested intervention meets every registered gain/preservation band or fails at least one.

**Question:** g2-leak-150 satisfies every stage-G gain and preservation clause.

**Decision limit:** Registered stage-G bands, including 310 pA count >=8 and preservation of sweep-43 return.

**Evidence says:** Counts become 0/0/6 at 200/250/310 pA; late return shifts about -3.8 mV. Registered clauses fail.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain the correction for independent validation. | Leaf; see next check |
| No | Exclude this leak dose as a sufficient correction with the preserved controls. | E05 |

**Next check:** Carry forward the failed tested dose; distinguish the unrun combined arm in E05.

**Still unresolved after either answer:** No continuous-dose theorem or exhaustive passive-family exclusion is established by this single dose.

**Sources:** [stage-g-decision.json](../evidence/h01-e-gain/stage-g-decision.json), [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json).

### E05 — Combined Ih/leak arm meets the bands?

**Status: BLOCKED** · Structural dissection / interaction

**Scope / coordinates:** Registered g3: Ih factor 37.5 plus leak x1.5 on B3, SP3.

**Remaining space:** The combined intervention meets all registered bands or misses at least one; no observation selects an outcome.

**Question:** An actual g3 evaluation satisfies every registered gain/preservation band.

**Decision limit:** SP3 stage-G registered bands; additive predictions are not measurements.

**Evidence says:** g3 was registered but not run. SP3 closed FAIL at 3 of 4; the additive prediction missed the bands.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | An interaction warrants further independent testing. | Leaf; see next check |
| No | This combined intervention fails under the tested conditions. | Leaf; see next check |

**Next check:** No automatic resumption. A new bounded reassessment must distinguish interaction, voltage/use-dependent current, and alternative-donor explanations.

**Still unresolved after either answer:** Slow Na inactivation, Kv7/M, other mechanisms, interactions and reference-family differences remain hypotheses, not an exhaustive list.

**Sources:** [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json).

### E06 — B3 passes every waveform/return row?

**Status: NO** · Hierarchical / within-cycle and post-pulse

**Scope / coordinates:** Frozen B3, released E inputs; retain every event index, waveform phase, and sweep-43 post-pulse sample.

**Remaining space:** All required rows pass or at least one fails; unavailable rows never count as passes.

**Question:** Every required waveform and post-pulse voltage row meets the approved contract with numerical support.

**Decision limit:** 1 mV voltage, 1 ms timing and recovery delays, 0.05 ms phases; shared parameters over inputs.

**Evidence says:** B3's sweep-43 late return is +1.4 to +1.7 mV; early-width defects remain. This alone prevents a full pass.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Proceed to independent transfer and held-out qualification. | Leaf; see next check |
| No | Keep the specific phase/return residuals separate from the gain problem. | Leaf; see next check |

**Next check:** First compare the retained within-cycle V/dVdt and model signed-current budgets at the offending input/event; register a phase-specific split before perturbing.

**Still unresolved after either answer:** Older perisomatic energetic maps are historical; do not describe B3 as having no axonal initiation.

**Sources:** [stage-close-decision.json](../evidence/h01-e-gain/stage-close-decision.json), [h01-e-continuation-result.md](../evidence/h01-e-continuation-result.md).

## I. I donor: refiring and accommodation

The model misses the early burst and later accommodation despite a supported loop/trough explanation.

### I01 — Does the energetic holdout pass the contract?

**Status: NO** · Hierarchical / cycle and input

**Scope / coordinates:** I finalist e-kv3-close2; released 0.23 nA holdout; command-only input.

**Remaining space:** Complete held-out contract pass versus one or more failed/unavailable required rows.

**Question:** The frozen finalist meets every approved contract row on the released 0.23 nA response.

**Decision limit:** Exact count, 1 mV voltage, 1 ms timing, 0.05 ms phases; repeatability bands are separate.

**Evidence says:** Count 26 versus 31; first peak residual 1.15 mV. The loop/trough explanation held at its claimed repeatability resolution, not the full contract.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Carry the frozen finalist into independent transfer qualification. | Leaf; see next check |
| No | Retain missing early burst and accommodation as unresolved responses. | I02 |

**Next check:** Use the post-trough current/voltage evidence and subsequent reserve result without reusing this input as a fresh holdout.

**Still unresolved after either answer:** An explanatory result on selected rows does not qualify the whole cell.

**Sources:** [h01-i-energetic-result.md](../evidence/h01-i-energetic-result.md).

### I02 — Axonal NaTg x1.5 restores refiring?

**Status: NO** · Structural dissection / sodium boundary

**Scope / coordinates:** g-natg-axon15, sha256 f4a2aa6c24d887343e7003a0a226d6f6d702dfd707e31ad05d44773ea9c5dcbe; 0.19/0.27 nA; command-only; x9; CVode 1e-10; 1500 ms.

**Remaining space:** Registered reserve intervention restores refiring while preserving the loop/trough, or it does not.

**Question:** Every registered count/refiring/preservation band holds at both inputs.

**Decision limit:** Cycle 2 <=22 ms and count 15-17 at 0.19; cycle 2 <=12 ms and count 40-43 at 0.27; retained width/peak/trough bands.

**Evidence says:** Cycle 2 is 35.068/16.496 ms, counts 14/34. Axon lead increases, but somatic refiring does not meet the bands. FAIL at cap.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain a supported sodium-boundary correction. | Leaf; see next check |
| No | Exclude this axonal NaTg density change as a sufficient correction. | I03 |

**Next check:** Next distinguish residual inward drive from persisting outward load using I03, with paired signed currents and voltage.

**Still unresolved after either answer:** The valid run spent the reserve. Earlier process crashes tested no physiological hypothesis.

**Sources:** [stage-1-decision.json](../evidence/h01-i-reserve/stage-1-decision.json), [h01-i-reserve-result.md](../evidence/h01-i-reserve-result.md).

### I03 — Recovered inward drive is sufficient?

**Status: OPEN** · Impedance / current balance

**Scope / coordinates:** Model post-trough interval, 6-10 ms after each complete trough, at 0.19/0.27 nA; soma and adjacent axial boundaries.

**Remaining space:** Sufficient inward charge in the retained load versus an inward/outward balance that still cannot produce timely threshold crossing.

**Question:** With outward load and initialization fixed, a preregistered inward-current replay reaches the specified threshold-time band.

**Decision limit:** No new numerical limit is invented. Register the replay amplitude/time course and crossing/charge limits from resolved baseline rows before an experiment.

**Evidence says:** The reserve does not identify a missing slow state uniquely. Somatic availability, tested Ca_LVA changes and the tested axonal density change narrow only their stated claims.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Dissect which inward boundary or state supplies the effect. | Leaf; see next check |
| No | Retain excessive outward/axial load, interactions and other missing dynamics. | Leaf; see next check |

**Next check:** First audit current signs and charge closure over matched V/time windows; formulate a reversible inward-versus-load isolation with fixed complement and a new cap.

**Still unresolved after either answer:** Human voltage does not directly measure individual ionic currents. A model replay is a causal test in the model, not a human current measurement.

**Sources:** [h01-i-reserve-result.md](../evidence/h01-i-reserve-result.md), [h01-i-trough-audit.json](../evidence/h01-i-trough-audit.json), [h01-z-map-i.md](../evidence/h01-z-map-i.md).

### I04 — Reserve waveform passes all inputs?

**Status: NO** · Hierarchical / within-cycle and cyclical

**Scope / coordinates:** Valid I reserve, 0.19/0.27 nA, every matched complete event and recovery interval.

**Remaining space:** All waveform rows pass versus at least one failure or unresolved required row.

**Question:** The reserve meets every contract waveform row and the shared-input count requirement.

**Decision limit:** Contract unchanged; report usable width and adaptation separately; unmatched events remain unavailable.

**Evidence says:** Counts 14/12 and 34/43 model/human; cycle-1 width about 0.220 versus human 0.267-0.273 ms, and later width/adaptation rows fail.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Proceed to independent numerical and transfer gates. | Leaf; see next check |
| No | Keep phase width and accommodation open even where trough voltage is acceptable. | Leaf; see next check |

**Next check:** Retain per-cycle phase-plane traces and compare early versus late current budgets; do not tune solely to the mean rate.

**Still unresolved after either answer:** Per-cycle discrepancies can coexist with a successful trough explanation.

**Sources:** [h01-i-reserve-result.md](../evidence/h01-i-reserve-result.md), [stage-1-decision.json](../evidence/h01-i-reserve/stage-1-decision.json).

## T. Simulator and H01 transfer

Keep the amended short-window closure, strict full-train check, timestep error, and anatomy effects separate.

### T01 — Short-window identity closed by amendment?

**Status: YES** · Functional isolation / simulator

**Scope / coordinates:** I r-braincell-matched-027 against NEURON b1-fixed-027; 0.27 nA, copied x9 mesh, same dt 0.005 ms; window 270-329.5 ms.

**Remaining space:** The amended recorded identity closure was met, or it was not; this is explicitly a recorded decision predicate.

**Question:** The basis reproduces the quoted values and identity_gate.closed is true under the recorded user amendment.

**Decision limit:** Amended basis: rise about 1.2e-9 ms, interpolated peak 5.4e-6 mV, width 1.9e-11 ms. Original 1e-6 mV automatic gate was not passed.

**Evidence says:** identity_gate.closed=true by user decision; basis/passed=false under the stricter automatic gate. Both facts are retained.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Keep the narrow closure; inspect the separate full-train predicate. | T02 |
| No | Investigate matched simulator behavior within this window. | Leaf; see next check |

**Next check:** Read T02 for the full-train numerical prediction rather than extending this closure.

**Still unresolved after either answer:** No inference to 0.19 nA, finer dt, H01 anatomy, or human waveform qualification.

**Sources:** [sp2-close-decision.json](../evidence/h01-i-transfer/sp2-close-decision.json), [h01-i-transfer-result.md](../evidence/h01-i-transfer-result.md).

### T02 — Strict full-train identity check passes?

**Status: NO** · Functional isolation / simulator

**Scope / coordinates:** I a1-matched-027 versus b1-fixed-027; same dt 0.005 ms and copied mesh; 270-1270 ms at 0.27 nA.

**Remaining space:** All strict identity rows satisfy the registered prediction, or one or more do not.

**Question:** The full-train comparison has equal event counts and all strict per-event identity errors within the registered limits.

**Decision limit:** Rise 1e-8 ms, peak 1e-6 mV, width 1e-8 ms; retain sampled and interpolated peak definitions separately.

**Evidence says:** 36/36 events; passed=false; max rise 1.415e-7 ms, raw voltage difference 8.33e-5 mV. Both sampled and interpolated peak errors exceed 1e-6 mV.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Strict confirmation holds for this configuration. | Leaf; see next check |
| No | Retain the strict prediction failure without undoing or broadening T01's amended decision. | Leaf; see next check |

**Next check:** Before any rerun, reconcile sampled/interpolated peak fields with the scorer and document the failing strict rows. No tolerance relaxation.

**Still unresolved after either answer:** Tiny simulator differences are not the millisecond fixed-step-vs-CVode drift, nor evidence of a human pass.

**Sources:** [sp2-close-decision.json](../evidence/h01-i-transfer/sp2-close-decision.json), [h01-i-transfer-result.md](../evidence/h01-i-transfer-result.md).

### T03 — 0.27 nA dt meets the crossing limit?

**Status: YES** · Hierarchical / numerical time

**Scope / coordinates:** NEURON finalist fixed dt 0.000625 ms versus CVode 1e-10, copied x9 mesh, 0.27 nA, 270-1270 ms.

**Remaining space:** Equal event count and all rise errors <=1 ms, or not.

**Question:** The fixed-step response has equal nonzero count and every paired rising-crossing error <=1 ms.

**Decision limit:** 1 ms rise error and exact count for this timestep check only.

**Evidence says:** 37/37 events; maximum rise error about 0.981 ms. This registered crossing check passes.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain this measured timestep result for the stated input. | T04 |
| No | Refine numerics before interpreting this crossing residual. | Leaf; see next check |

**Next check:** Evaluate the separate 0.19 nA result in T04; do not interpolate qualification across inputs.

**Still unresolved after either answer:** This is not a complete mesh, peak, width or human validation verdict.

**Sources:** [sp2-close-decision.json](../evidence/h01-i-transfer/sp2-close-decision.json), [h01-i-transfer-result.md](../evidence/h01-i-transfer-result.md).

### T04 — 0.19 nA dt meets the crossing limit?

**Status: NO** · Hierarchical / numerical time

**Scope / coordinates:** NEURON finalist fixed dt 0.000625 ms versus CVode 1e-10; copied x9 mesh; 0.19 nA; 270-1270 ms.

**Remaining space:** Equal count and all rise errors <=1 ms, or not.

**Question:** The fixed-step response has equal nonzero count and all rising-crossing errors <=1 ms.

**Decision limit:** 1 ms rise error and exact count; same criterion as T03, different retained input.

**Evidence says:** 14/14 events but maximum rise error about 1.864 ms. Cap exhausted; a further halving is projected, not measured.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain a measured timestep for this input. | Leaf; see next check |
| No | Keep numerical qualification open at 0.19 nA. | Leaf; see next check |

**Next check:** A new bounded scope would measure one further halving and the absent matched BrainCell 0.19 train; no run in this artifact task.

**Still unresolved after either answer:** Count agreement alone cannot resolve crossing-time error.

**Sources:** [sp2-close-decision.json](../evidence/h01-i-transfer/sp2-close-decision.json), [h01-i-transfer-result.md](../evidence/h01-i-transfer-result.md).

### T05 — Anatomy swap alone changes the response?

**Status: OPEN** · Structural dissection / anatomy and region map

**Scope / coordinates:** Donor versus H01 morphology under frozen electrical controls and matched input, numerical, and measurement conventions; region map changes recorded separately.

**Remaining space:** Qualified anatomy-only transfer contrast above the registered row limit versus no such contrast.

**Question:** A matched anatomy-only swap changes the selected response row beyond its established uncertainty.

**Decision limit:** Use per-row qualified numerical contrast limits, not donor physiological tolerance as a substitute for measurement resolution.

**Evidence says:** Current H01 profiles include inferred electrical region assignments; unresolved Y1 concerns the upstream axonal outflow/region map.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Dissect morphology/region boundaries carrying the contrast. | Leaf; see next check |
| No | Retain source-model/reference mismatch and other unsplit differences. | Leaf; see next check |

**Next check:** First enumerate anatomy and region-map differences; define one reversible matched swap with no invented cable and with both voltage and signed axial current.

**Still unresolved after either answer:** Changing anatomy and region assignment together cannot identify which caused a contrast; donor data cannot directly validate H01 physiology.

**Sources:** [h01-causal-model.md](../h01-causal-model.md), [2026-09-07-h01-transfer-isolation.md](../specs/2026-09-07-h01-transfer-isolation.md), [h01-network.md](../h01-network.md).

## C. Measured contacts and circuit function

Anatomical support, cable placement, event delivery, and receiver inhibition are distinct questions.

### C01 — Selected contacts have verified endpoints?

**Status: YES** · Structural dissection / anatomical identity

**Scope / coordinates:** The retained verified set only: annotations 124698307 (EE), 65017731 (EI), 8105899 (IE).

**Remaining space:** Both endpoints have accepted identity/spatial support for each selected contact, or at least one lacks it.

**Question:** Each contact in the stated retained set passes the recorded endpoint identity and spatial checks.

**Decision limit:** Use saved exact identity and 5-voxel endpoint audit criteria; proximity alone is insufficient.

**Evidence says:** The completed 3000-sample summary retains three verified anatomical contacts; new candidates did not enlarge the set.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Check cable support per contact before simulation. | Leaf; see next check |
| No | Retain unsupported contact candidates outside the verified simulation graph. | Leaf; see next check |

**Next check:** Keep IDs, sending/receiving cell and component support; do not treat the partial map as complete connectivity.

**Still unresolved after either answer:** The answer applies to this selected set, not every exported candidate or every biological connection.

**Sources:** [h01-network.md](../h01-network.md), [h01-endpoint-recheck-5voxel-3000.json](../evidence/h01-endpoint-recheck-5voxel-3000.json).

### C02 — EI contact has soma-connected cable?

**Status: NO** · Structural dissection / connected component

**Scope / coordinates:** Annotation 65017731: sender 3519995546, presynaptic component 55; receiver 3680152874 component 0.

**Remaining space:** Both endpoints connect to the retained simulated somata through released cable, or at least one does not.

**Question:** Both endpoint placements are on soma-containing connected components with supported cable.

**Decision limit:** Exact component connectivity; no guessed joins or substituted soma emissions.

**Evidence says:** The source is soma-free fragment 55. This anatomical contact is excluded from the simulation subset.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Construct this supported projection. | Leaf; see next check |
| No | Keep the contact anatomical-only and simulation-blocked. | Leaf; see next check |

**Next check:** Resolve missing cable/placement evidence before enabling this edge; no morphology repair is part of this task.

**Still unresolved after either answer:** Two other verified contacts are constructible; that does not repair this edge.

**Sources:** [h01-network.md](../h01-network.md), [h01-verified-network-validation.md](../evidence/h01-verified-network-validation.md).

### C03 — Four-cell subset executes finite steps?

**Status: YES** · Functional isolation / runtime

**Scope / coordinates:** Four cells, two supported projections in two separate pairs, 75,605 compartments; saved real-network validation and throughput configurations.

**Remaining space:** Finite real initialization and stepping exist, or they do not.

**Question:** The retained four-cell evidence includes finite initialization and at least one actual compiled timestep.

**Decision limit:** Finite values and completed actual timestep; report the exact run duration rather than imply a long physiological protocol.

**Evidence says:** Real one-step dt/duration 0.005 ms exists; throughput checks extend to 2000 steps. Construction alone is not the evidence for this Yes.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Runtime entry is demonstrated for the specified subset. | Leaf; see next check |
| No | Localize construction, initialization or execution failure before functional claims. | Leaf; see next check |

**Next check:** Use C04-C07 for functional inhibition; do not infer it from runtime or fixture delivery.

**Still unresolved after either answer:** No full 104-cell run, measured-pair human response, or training adapter qualification follows.

**Sources:** [h01-network.md](../h01-network.md), [h01-verified-network-validation.md](../evidence/h01-verified-network-validation.md), [h01-network-throughput.md](../evidence/h01-network-throughput.md).

### C04 — 300 ms baseline is discriminating?

**Status: INCONCLUSIVE** · Hierarchical / circuit measurement

**Scope / coordinates:** SP5 disconnected baseline, 300 ms, E drive 0.6 nA, dt 0.005 ms; 12 I pulse onsets 20-295 ms.

**Remaining space:** Completed baseline meets the firing/discrimination conditions, or a completed baseline fails them.

**Question:** The completed baseline has >=3 E spikes, first E spike 20-60 ms, and one I spike per pulse.

**Decision limit:** Registered baseline conditions plus the measured halving limit before any delay/inhibition verdict.

**Evidence says:** The baseline was stopped after construction and before a trace. Other 300 ms arms were not launched.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Proceed to measured-site event and conductance checks. | C05 |
| No | Repair the registered drive/measurement setup before testing inhibition. | Leaf; see next check |

**Next check:** Any resumption requires a separately bounded plan. Retain the stopped run as untested, not a failed baseline or spent physiological evaluation.

**Still unresolved after either answer:** All downstream answers remain pending even though their hypothetical branches are shown.

**Sources:** [decision.json](../evidence/h01-ie-inhibition/decision.json), [h01-ie-inhibition-result.md](../evidence/h01-ie-inhibition-result.md).

### C05 — Contact event produces delayed conductance?

**Status: BLOCKED** · Functional isolation / serial delivery

**Scope / coordinates:** SP5 measured IE placement 8105899; presynaptic site crossing, event timestamp, receptor conductance, receiver site voltage.

**Remaining space:** Event-to-conductance path completes with registered timing, or at least one link fails.

**Question:** Each qualifying presynaptic event reaches the supported source site and yields receptor conductance after the configured delay.

**Decision limit:** Delay and conductance conventions from the selected manifest; compare timestamps at the simulation resolution with a disconnected control.

**Evidence says:** No completed 300 ms measured arm. Fixture delivery and a short benchmark do not answer this predicate.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Inspect local versus somatic response in C06. | C06 |
| No | Split propagation, event detection, routing, and receptor response at the first missing boundary. | Leaf; see next check |

**Next check:** On future authorized SP5 execution, record the event and conductance alongside V and signed synaptic current; first satisfy C04.

**Still unresolved after either answer:** Assumed synaptic magnitude/delay are not measured human contact parameters.

**Sources:** [decision.json](../evidence/h01-ie-inhibition/decision.json), [h01-ie-inhibition-manifest.json](../evidence/h01-ie-inhibition-manifest.json), [h01-verified-network-validation.md](../evidence/h01-verified-network-validation.md).

### C06 — Measured placement changes E firing?

**Status: BLOCKED** · Functional isolation / receiver response

**Scope / coordinates:** Completed SP5 measured arm versus disconnected arm, same E/I drives and frozen profiles; compare every E cycle.

**Remaining space:** At least one E count/timing effect exceeds qualified numerical uncertainty, or none does.

**Question:** The measured contact changes E count or delays at least one matched E spike beyond the established halving limit.

**Decision limit:** Registered per-cycle shift limit from measured/measured-halved, with equal counts in the halving pair and per-cycle ranges <0.1 ms.

**Evidence says:** No 300 ms measured or halved arm completed. Gate verdict is not discriminable; placement hypothesis undetermined.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Functional inhibition is supported for these assumed parameters and inputs. | Leaf; see next check |
| No | Check whether the failure to affect soma firing is placement-dependent in C07. | C07 |

**Next check:** Retain site IPSP, soma IPSP, source event, conductance, and E cycles; do not call an unresolvable shift absence of inhibition.

**Still unresolved after either answer:** Even a positive result would not validate a human circuit response; no recording of this pair exists.

**Sources:** [decision.json](../evidence/h01-ie-inhibition/decision.json), [2026-09-07-h01-functional-inhibition.md](../specs/2026-09-07-h01-functional-inhibition.md).

### C07 — Soma placement restores a resolved effect?

**Status: BLOCKED** · Structural dissection / placement

**Scope / coordinates:** SP5 soma-control versus measured-site and disconnected controls; preserve receptor kinetics, magnitude, delay and drives.

**Remaining space:** Soma relocation produces the registered effect or it does not, conditional on qualified baseline/delivery/measurement.

**Question:** The soma-control produces >=0.3 mV soma IPSP and at least one E delay beyond the qualified shift limit.

**Decision limit:** Registered >=0.3 mV soma IPSP and measured delay threshold; predicted measured-site IPSP <0.05 mV is not observed.

**Evidence says:** The soma-control arm was not launched. Placement remains undetermined.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Placement/load contributes under these fixed model conditions. | Leaf; see next check |
| No | Keep drive, receptor strength, kinetics, and interactions in the remaining space. | Leaf; see next check |

**Next check:** After C04-C06 and new authorization, perform the registered matched placement contrast; soma placement is a diagnostic control, not a replacement measured edge.

**Still unresolved after either answer:** Do not promote the diagnostic soma control into the anatomical network.

**Sources:** [decision.json](../evidence/h01-ie-inhibition/decision.json), [h01-ie-inhibition-manifest.json](../evidence/h01-ie-inhibition-manifest.json).

## P. Population execution and reference coverage

Small-network execution exists; 40-cell construction and donor coverage remain separate limitations.

### P01 — 40-cell stage completes construction?

**Status: NO** · Functional isolation / build before run

**Scope / coordinates:** SP8 40-cell prefix, merged tree 0488c84; --include-isolated --cells 40 --control ei; build-only.

**Remaining space:** Actual construction succeeds for every selected cell, or at least one import/build fails.

**Question:** The build completes all selected morphologies and discretization without error.

**Decision limit:** Successful real construction of all 40 selected cells, not an extrapolated cost estimate.

**Evidence says:** Build fails at 82.6 s, before discretization, in H01Archive.load for cell 5805562981: from_points requires at least two points.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Proceed to separately measured initialization and runtime. | Leaf; see next check |
| No | Localize the import failure before investigating network stepping. | P02 |

**Next check:** Use P02 to retain the recorded import-level localization.

**Still unresolved after either answer:** Subsequent init/run stages were not launched; no 40-cell runtime measurement exists.

**Sources:** [h01-population-build-40.json](../evidence/h01-population-build-40.json), [h01-population-build-40.md](../evidence/h01-population-build-40.md).

### P02 — Import alone reproduces the failure?

**Status: YES** · Functional isolation / importer boundary

**Scope / coordinates:** Offline H01Archive.load of 104 largest components; same BrainCell SWC reader, no simulation.

**Remaining space:** Failure reproduced without network initialization or stepping, or only later in the pipeline.

**Question:** The standalone import reproduces the branch-construction failure.

**Decision limit:** Same from_points exception on the released component; exact node/parent structure retained.

**Evidence says:** Seven of 104 largest components fail. Near-coincident 0.032 um nodes interact with reader np.allclose at coordinates near 3000 um; both leaf and branch-point cases occur.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Keep the fault localized to import/reader geometry handling. | Leaf; see next check |
| No | Investigate construction context and subsequent stages. | Leaf; see next check |

**Next check:** A future repair needs a reproducer for both leaf and branch-point cases, preserved geometry/provenance, and full 104-component audit before rerunning 40 cells.

**Still unresolved after either answer:** This artifact applies no repair and does not authorize collapsing nodes or inventing cable.

**Sources:** [h01-population-build-40.md](../evidence/h01-population-build-40.md), [h01-population-build-40.json](../evidence/h01-population-build-40.json).

### P03 — Twelve-cell finite execution exists?

**Status: YES** · Functional isolation / staged execution

**Scope / coordinates:** SP8 12-cell subset: four incident cells plus eight selected isolated cells; use successful retained attempts, not killed attempts.

**Remaining space:** At least one actual finite execution exists for this subset, or only construction/projections exist.

**Question:** The saved 12-cell evidence contains a completed real finite runtime after initialization.

**Decision limit:** Completed real run and finite traces, with measured duration/load/attempt retained.

**Evidence says:** Attempt 3 completed 1 ms at dt 0.005 ms in both ei and disconnected configurations with finite traces; conductance probes were zero in both. Earlier killed attempts remain untested.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Retain the measured subset as the runtime boundary. | Leaf; see next check |
| No | Keep execution open despite construction success. | Leaf; see next check |

**Next check:** Report the selected attempt's measured configuration; do not extrapolate qualification to 40 or 104 cells.

**Still unresolved after either answer:** Includes isolated cells and borrowed profiles; runtime is not physiological coverage.

**Sources:** [h01-population-build-12.json](../evidence/h01-population-build-12.json), [h01-population-build-12.md](../evidence/h01-population-build-12.md).

### P04 — Added SST donor reproduces its counts?

**Status: NO** · Hierarchical / reference before transfer

**Scope / coordinates:** HL5MN1 source fit, 100/150 pA, sweeps 44/35, original and half dt.

**Remaining space:** Exact counts agree at both inputs, or at least one differs.

**Question:** The source model reproduces both matched donor counts.

**Decision limit:** Exact counts; retain separate repeat bands. Latency limits use |dt-pair difference| x2.95.

**Evidence says:** Human 14/34; model 16/30 at both dt values. Source reproduction rejected.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Continue waveform and transfer qualification. | Leaf; see next check |
| No | Isolate unchanged source-condition mismatch before using this donor as qualified physiology. | Leaf; see next check |

**Next check:** Audit initial state, bias, temperature, waveform and mechanism conventions under a frozen source model; preserve two-input constraints.

**Still unresolved after either answer:** Halving-dt count agreement rules out that count change over the tested pair, not every numerical issue.

**Sources:** [h01-sst-reproduction.md](../evidence/h01-sst-reproduction.md), [stage-hl5mn1-decision.json](../evidence/h01-donors/stage-hl5mn1-decision.json).

### P05 — Added L4 donor reproduces its counts?

**Status: NO** · Hierarchical / reference before transfer

**Scope / coordinates:** Allen 527952884 source fit, 100/90 pA, sweeps 69/39, original and half dt.

**Remaining space:** Exact counts agree at both inputs, or at least one differs.

**Question:** The source model reproduces both matched donor counts.

**Decision limit:** Exact counts; 100 pA repeat band 17-20 is separately reported and cannot replace exact 20.

**Evidence says:** Human 20/12; model 19/8 at both dt values. Being within the first repeat band does not satisfy both exact counts.

| Answer | What it establishes | Next question |
| --- | --- | --- |
| Yes | Continue waveform and transfer qualification. | Leaf; see next check |
| No | Isolate the source-condition mismatch before transfer claims. | Leaf; see next check |

**Next check:** Compare pinned initial state, temperature, waveform/bias and mechanism versions; keep every input in the test.

**Still unresolved after either answer:** Anatomical labels and imported templates do not establish complete population reference coverage.

**Sources:** [h01-l4-reproduction.md](../evidence/h01-l4-reproduction.md), [stage-allen-l4-decision.json](../evidence/h01-donors/stage-allen-l4-decision.json).

## Evidence precedence and known traps

- T01's amended closure is true while its original automatic gate is false. T02's strict full-train prediction fails. Keep both fields; do not choose whichever produces a pass.
- SP3's combined arm was not run. Predicted failure is not an observed No.
- I's command-only convention supersedes earlier proposals to add acquisition bias. E's recorded-bias convention is different.
- E sweep 53 and I 0.23 nA were originally sealed but have been used; E sweep 55 was subsequently released. They are not fresh holdouts. Preserve E 54 and I Noise1 48.
- The 300 ms SP5 baseline was interrupted; it is not evidence of absent inhibition.
- The 40-cell failure occurred in morphology import. Finite four-/twelve-cell execution does not cure it.
- Seven failing components include both leaf and branch-point cases. A leaf-only repair would not cover the recorded failure set.
- Anatomical contact 65017731 has verified endpoints but a soma-free source fragment; those statements are compatible.
- Older energetic maps describe earlier candidates. Do not project a passive-axon description onto B3.
- This snapshot hashes retained source documents/decisions. Large ignored raw traces were not newly replayed or independently rescored.

## Updating this map

Edit the structured question records and the matching native/Markdown annotations together, keeping stable IDs. Change an answered state only when evidence with the same retained conditions supports it; add a new scoped question if conditions change. Refresh evidence hashes and repeat the structural/native verification. Keep old hypotheses as conditional until a discriminating observation exists.
