# Brainiac business-value priorities

Date: 2026-08-25

Decision horizon: the current Example 21 replacement change, through its release gate.

Audience: Chief of Staff and Project Manager.

## Evidence basis

**Specified:** Brainiac seeks a lifelike brain model that is efficient and learns quickly. The active OpenSpec change replaces the old Example 21 path with a 2,048-neuron BrainCell Hodgkin-Huxley model trained by BrainTrace PP-Prop on real ARC practice tasks. Strict task pass-at-1, direct predictions, bounded runtime, sparse storage, and no answer shortcuts are the declared evaluation rules.

**Evidence:** OpenSpec tasks 1.1 through 6.3 are complete. The real-data Gate 4 proof completed eight ordered PP-Prop updates in 101.57 seconds. It recorded recurrent-weight movement of 0.03779, changed a direct prediction, reduced the recorded shape and row losses, preserved validation state, and passed the required state and null interventions. The proof establishes mechanism execution only. It does not establish useful ARC ability, strict task success, fast learning, or an efficiency advantage. Task 6.4, the complete co-located test, coverage, and 60-second gate, remains open. A separate 24-neuron delayed-cue baseline did not learn: PP-Prop ended at 0.4961 validation accuracy, the BPTT oracle ended at 0.4883, and the frozen-recurrence arm ended at 0.4961. PP-Prop was 20.3% slower than BPTT and used 0.8% more sampled peak process memory in that small CPU run. Because the BPTT oracle also failed, this evidence shows an invalid evaluation profile, not a PP-Prop failure. The older V48 ARC model changed outputs when some inputs were removed, but it failed strict task success and did not show useful dependence on demonstration outputs.

**Assumption:** The current phase optimizes evidence value before product reach. The evaluation order is: valid direct learning evidence, strict ARC behavior, bounded efficiency, biological detail, then cleanup and release polish. Relative complexity ratings below are analyst estimates from the specified scope, not delivery forecasts.

**Unknown:** No customer segment, usage count, willingness to pay, funding decision, deployment channel, operating-cost target, or comparable implementation effort is supplied. No current evidence shows useful ARC ability, rapid adaptation, efficiency advantage, human-like cognition, or consciousness.

RICE is **not defensible for any item**. Reach has no defined beneficiary or decision count, and effort has no shared measured point scale. A comparable RICE calculation becomes possible when the Project Manager declares one decision population for this horizon and the implementation team supplies relative effort points for every remaining item.

## Completed work, 2026-08-25

Completed work is evidence and a maintained prerequisite. It is not part of the active priority queue.

## 1. BrainCell compatibility and learning-rule fixtures

**Outcome and JTBD:** When researchers change the biological model or online learning path, they need dependency, state, unit, gradient, and spike-path checks so an executable result is technically interpretable.

**Kano and beneficiaries:** Basic. Primary: Brainiac researchers. Secondary: reviewers and future contributors. Inconvenience: fixture maintenance and pinned dependency constraints.

**Pain, workaround, and cost of not building:** Without these checks, import success or a finite run can mask wrong current units, reset state, temporal relations, or gradients. Manual inspection is the weak workaround. Not building blocks every credible learning claim and makes later experiment results ambiguous.

**Canvas:** The material value proposition is credible experimental infrastructure. Key resources are BrainCell, BrainTrace, BrainState, and the compatibility fixtures. Key activities are dependency control and independent gradient checks. Partnerships are upstream package maintainers. Cost is dependency and fixture maintenance. Customer, channel, relationship, and revenue effects are unknown.

**Porter:** Dominant impact is Technology and Infrastructure support for Operations. It unlocks trustworthy experiments and adds version-pin and regression-test load. Other primary activities are not material in this research phase.

**VRIO:** Valuable, but not shown rare or hard to imitate. Organization is present through review gates. Position: parity.

**SWOT:** Strength: explicit, falsifiable checks. Weakness: small fixtures do not prove end-to-end learning. Opportunity: detect interface drift early. Threat: upstream dependency change can invalidate results.

**PESTEL:** Technological dependency stability is material. Legal and procurement terms for dependencies are unknown. Other factors are not material to this phase.

**Priority:** RICE not defensible. Estimated complexity 4/10, M. Impact/Effort: Quick Win. Impact Criticality: Critical. Dependencies: none. Main risk: passing local derivatives may create false confidence in system-level learning. Confidence: high that this is a gate, based on completed review evidence.

**Balanced Scorecard:** Financial proxy: avoided invalid experiment reruns, indicator = compatibility failures caught before full proof runs. Customer proxy: reviewer trust, indicator = independent gate outcome. Internal Process: reproducibility, indicator = fixture pass rate. Learning & Growth: clearer failure localization, indicator = failures assigned to dependency, state, unit, or gradient boundaries.

**Recommendation:** **Build, current phase.** This is complete and remains a required regression gate.

## 2. Real ARC data, event, loss, prediction, result, and checkpoint contracts

**Outcome and JTBD:** When the model trains and evaluates on ARC, researchers need a lossless target-isolated input and exact output contract so observed performance cannot come from leakage, partial credit, or hidden shortcuts.

**Kano and beneficiaries:** Basic. Primary: experiment decision-makers. Secondary: reviewers and maintainers. Inconvenience: strict validation rejects malformed or oversized artifacts.

**Pain, workaround, and cost of not building:** The old path used engineered routing and partial-score surfaces that did not establish direct strict ARC behavior. Ad hoc encoders and result inspection are the current workaround. Without this contract, Gate 4 and every later comparison lack a valid outcome measure.

**Canvas:** Value proposition: direct, auditable ARC evidence. Key resources: public practice data and deterministic encoding. Key activities: validation, target isolation, exact decoding, scoring, and atomic checkpointing. Channel value is a small reproducible result artifact. Revenue and customer relationship effects are unknown.

**Porter:** Dominant impact is Inbound Logistics and Operations. It controls what data enters the experiment and how results leave it. It adds schema maintenance and fixed-capacity constraints.

**VRIO:** Valuable and organized. The contracts are reproducible and therefore imitable. Position: parity, with high prerequisite value.

**SWOT:** Strength: direct and lossless measurement. Weakness: a small fixed screen limits generalization claims. Opportunity: comparable experiments. Threat: overfitting decisions to the fixed practice screen.

**PESTEL:** Legal data-use terms are an explicit unknown. Technological data integrity is material. Other factors are not material.

**Priority:** RICE not defensible. Estimated complexity 5/10, M. Impact/Effort: Quick Win. Impact Criticality: Critical. Dependency: compatibility gate. Risk: exact contracts can still support an ineffective learner. Confidence: high, based on completed Gate 2 evidence.

**Balanced Scorecard:** Financial proxy: avoided invalid runs, indicator = rejected leakage or malformed-data cases. Customer proxy: evidence credibility, indicator = exact prediction and target artifacts. Internal Process: deterministic inputs and bounded outputs, indicators = byte equality and artifact size. Learning & Growth: experiment comparability, indicator = repeatable task results.

**Recommendation:** **Build, current phase.** Complete and mandatory for all later evidence.

## 3. Sparse BrainCell baseline and PP-Prop training

**Outcome and JTBD:** When Brainiac tests its core thesis, researchers need a sparse biological recurrent model that receives finite online gradients and changes executed behavior so they can test learning without BPTT or answer shortcuts.

**Kano and beneficiaries:** Performance. Primary: Brainiac research leadership. Secondary: researchers, reviewers, and prospective technical adopters. Harm risk: compute use and misleading mission claims if mechanism execution is confused with useful ability.

**Pain, workaround, and cost of not building:** The old Example 21 was large, slow, and indirect. Conventional BPTT or non-biological models would test a different thesis. Not building removes the core mechanism under evaluation and blocks all mission-relevant learning evidence.

**Canvas:** The central value proposition is online learning in a sparse biologically grounded recurrent model. Key resources are BrainCell dynamics, BrainTrace PP-Prop, sparse topology, and specialized know-how. Key activities are compiled training and state-safe evaluation. Potential future segments and funding value are assumptions until external evidence exists.

**Porter:** Dominant impact is Operations and Technology. It creates the experimental capability and imposes compiler, numerical-stability, sparse-state, and optimizer support load.

**VRIO:** Valuable if it learns efficiently. Rarity and imitability are unknown because no market or capability comparison is supplied. Organization is partial: Gate 4 shows mechanism execution, but useful learning is unproved. Position: temporary advantage only if later strict screens succeed; otherwise parity or disadvantage.

**SWOT:** Strength: direct alignment with the mission and sparse online-learning architecture. Weakness: current external baseline shows no learning or efficiency advantage. Opportunity: credible evidence for fast adaptation without BPTT. Threat: biological complexity may consume resources without improving strict behavior.

**PESTEL:** Technological maturity, GPU portability, energy use, and reproducibility are material. Economic value is unknown until compute and learning performance are measured at relevant scale. Social and legal risks arise if lifelike or conscious claims exceed evidence.

**Priority:** RICE not defensible. Estimated complexity 9/10, XL. Impact/Effort: Major Project. Impact Criticality: Critical. Dependencies: compatibility and data contracts. Risks: numerical instability, trace/compiler gaps, failure to learn, and no efficiency gain. Confidence: high that the capability is mission-critical; low that it will produce useful ARC ability.

**Balanced Scorecard:** Financial proxy: compute efficiency, indicators = wall time and peak memory against matched valid baselines. Customer proxy: useful model behavior, indicator = zero-tolerance task pass-at-1. Internal Process: correct sparse online learning, indicators = finite nonzero gradients, weight movement, and state invariance. Learning & Growth: mechanism understanding, indicator = causal interventions that change direct predictions or loss.

**Recommendation:** **Build complete. Maintain, but do not expand it.** Use the completed bounded proof as evidence of execution, then require strict task evidence before claims of useful learning.

## 4. Gate 4 real-data mechanism, backend, and timing proof

**Outcome and JTBD:** When leadership decides whether to invest beyond the baseline, it needs one bounded real-data run that shows the complete mechanism executes, learns, affects predictions, preserves validation state, and selects a valid backend from direct measurements.

**Kano and beneficiaries:** Basic for the experiment verdict. Primary: Chief of Staff and research leadership. Secondary: Project Manager, reviewers, and implementers. Inconvenience: separate CPU/GPU processes and strict evidence collection add operational work.

**Pain, workaround, and cost of not building:** Unit tests alone could not show end-to-end mechanism execution. The prior workaround was component evidence. Without the proof, later causal and structural claims would remain unsupported. The proof is now complete. Its strict result does not justify useful-ability claims.

**Canvas:** Value proposition: a go, revise, or stop decision from direct evidence. Key activities are synchronized backend comparison, eight PP-Prop updates, forward-only validation, intervention tests, and bounded reporting. Key resources are the fixed real-data pair, CPU/GPU environments, and the implemented baseline. Revenue and market reach are unknown.

**Porter:** Dominant impact is Operations. It converts internal capability into decision evidence. Outbound value is the bounded proof artifact. Burden: hardware parity, compilation, timing discipline, and evidence retention.

**VRIO:** The proof itself is imitable, but credible negative as well as positive evidence is valuable. Organization is strong if independent review remains enforced. Position: parity as a process; it may reveal a differentiated capability.

**SWOT:** Strength: smallest direct test of the declared mechanism. Weakness: eight updates on two tasks cannot establish useful ARC ability. Opportunity: stop low-value expansion early or justify the next strict screen. Threat: treating prediction change as task success.

**PESTEL:** Technological hardware variance and environmental reproducibility are material. Environmental impact should be tracked through direct runtime and hardware use, not inferred. Other factors are not material.

**Priority:** RICE not defensible. Estimated complexity 7/10, L. Impact/Effort: Major Project. Impact Criticality: Critical. Dependency: sparse baseline and contracts. Risks realized and controlled by the proof include runtime, prediction-change, validation-state, and intervention checks. The remaining strategic risk is that technically valid execution is not useful ARC ability. Confidence: high in the execution verdict and low in any broader capability claim.

**Balanced Scorecard:** Financial proxy: bounded experiment cost, indicators = total wall time and backend timings. Customer proxy: credible direct behavior, indicators = actual grids and strict booleans. Internal Process: proof completeness, indicator = all Gate 4 checks and independent review. Learning & Growth: causal understanding, indicator = intervention effects and explicit null result.

**Recommendation:** **Build complete. Maintain the evidence.** Do not repeat this as active work unless a regression invalidates it. Passing proves execution only, not useful ARC ability.

## Active and upcoming priorities

The sections below map business value to the remaining OpenSpec task groups. Ownership is taken from the current Paperclip issue assignments as of 2026-08-26. The Project Manager owns any later reassignment.

| Value group | OpenSpec tasks | Dependency | Success criterion | Assigned issue and specialist |
|---|---|---|---|---|
| Test and coverage gate | 6.4 | Completed 1.1 through 6.3 | Co-located module has more than 90% meaningful coverage, completes within 60 seconds, and all limit failures fail closed | BRA-8, Revy, Senior Software Engineer. The board marks this issue done. |
| Structural evidence and adaptation | 7.1 through 7.6 | Completed Gate 4 proof; 7.2 also requires at least one strict validation pass for pruning | Measured attribution passes hand fixtures; each accepted 5% arm improves at least one strict task with no regression; compaction preserves prediction bytes; each arm finishes within 300 seconds | BRA-46 coordination; BRA-49 implementation, Cody, Software Engineer. |
| Dale candidates and biology guard | 8.1 through 8.4 | Accepted untyped parent from group 7 | Separate measured E/I arms preserve signs; promotion requires a strict false-to-true change with no regression; optional biology stays inactive | BRA-10, Cody, Software Engineer. |
| Plot and implementation-truth documents | 9.1 through 9.3 | Accepted stable checkpoint and direct evidence | Plot matches checkpoint and does not change predictions; documents describe executed code and separate observation from inference | BRA-11, Cody, Software Engineer. |
| Obsolete-path retirement | 10.1 through 10.3 | Validated replacement and stable documentation | One active Example 21 remains, no obsolete references remain, and the public API is unchanged | BRA-12, Cody, Software Engineer. |
| Final independent validation | 11.1 through 11.3 | Groups 6 through 10 complete or explicitly cut | Literal runtime and coverage gates, strict OpenSpec validation, clean diff check, language and docstring review, and independent release decision | BRA-13, Revy, Senior Software Engineer. |
| Release finalization | Release gate after 11.1 through 11.3 | BRA-13 approved | Repository release checks pass; release finalization and roadmap handoff are recorded | BRA-14, Rainbow, Release Manager. |

## 5. Bounded structural adaptation

**Outcome and JTBD:** When a functioning baseline has measurable deficiencies, researchers need evidence-led pruning and growth so the model can adapt capacity without dense storage or random structural search.

**Kano and beneficiaries:** Delighter. Primary: researchers testing adaptive architecture. Secondary: future technical adopters. Inconvenience: more experiment arms, optimizer remapping, and structural-state complexity.

**Pain, workaround, and cost of not building:** Fixed topology may constrain learning. The workaround is to keep the baseline topology. Not building does not block the current-phase mechanism verdict. It delays evidence about structural adaptation, a mission-aligned but unproved differentiator.

**Canvas:** Potential value proposition: measured self-modification. Key activities are contribution measurement, pruning, compaction, and bounded additions. Key resources are trustworthy causal metrics and sparse remapping. Costs are added experimental and maintenance complexity. Commercial blocks are unknown.

**Porter:** Dominant impact is Technology development. It may improve Operations but adds major support load in sparse storage, optimizer state, and causal attribution.

**VRIO:** Potentially valuable and rare, but value, rarity, and imitability are unproved. Organization is not ready before the baseline proof. Position: possible temporary advantage, currently unverified.

**SWOT:** Strength: strong mission fit. Weakness: higher causal and implementation complexity. Opportunity: better capacity allocation. Threat: adaptive changes can overfit the fixed screen or obscure whether PP-Prop learned.

**PESTEL:** Technological reproducibility and compute burden are material. Other factors are not material or are unknown.

**Priority:** RICE not defensible. Estimated complexity 9/10, XL. Impact/Effort: Major Project. Impact Criticality: High. Dependencies: OpenSpec 6.4 must close; task 7.2 pruning also requires at least one fixed validation task to pass strictly. Gate 4 execution evidence is complete. Risk: high opportunity cost before strict baseline competence. Confidence: medium on option value, low on near-term outcome value.

**Balanced Scorecard:** Financial proxy: compute and model-size efficiency, indicators = sparse counts, runtime, and peak memory. Customer proxy: strict behavior, indicator = no-regression task pass-at-1. Internal Process: safe remapping, indicator = invariants after each change. Learning & Growth: structural causality, indicator = one-arm measured gain versus unchanged parent.

**Recommendation:** **Execute the bounded group 7 child approved on 2026-08-26.** BRA-49 owns implementation under BRA-46. Keep pruning fail-closed while strict validation remains zero. Run structural arms one 5% change at a time, in separate processes, and reject an arm after no direct strict gain or any strict regression. Gate 5 review remains independent, and the Release Manager remains the only release finalizer.

## 6. Measured Dale candidates and deferred biological mechanisms

**Outcome and JTBD:** When the untyped baseline works, researchers need controlled E/I sign experiments so they can test whether added biological structure improves strict behavior without random assignment.

**Kano and beneficiaries:** Delighter. Primary: biological-model researchers. Secondary: future research partners. Inconvenience: constrained optimization and more comparison arms.

**Pain, workaround, and cost of not building:** The baseline lacks Dale typing and richer synapses. Keeping signed untyped weights is a valid workaround. Not building has low current-phase cost because the mission verdict does not require these mechanisms and premature detail can hide the learning question.

**Canvas:** Potential value proposition is biological fidelity with measured task value. Key resources are accepted structural parents and stable sparse sign enforcement. Partnership value with neuroscience researchers is an assumption. Revenue and channel effects are unknown.

**Porter:** Dominant impact is Technology. It adds model and review burden before any proven operating benefit.

**VRIO:** Biological mechanisms are available to others. Unique value would depend on evidence that they improve online adaptation. Position: parity until measured improvement exists.

**SWOT:** Strength: greater biological plausibility. Weakness: weak direct link to ARC value. Opportunity: discover useful inductive bias. Threat: novelty bias and slower experiments.

**PESTEL:** Social claim discipline is material: biological detail must not be described as human-like cognition. Technological and environmental costs require measurement. Other factors are unknown.

**Priority:** RICE not defensible. Estimated complexity 7/10, L. Impact/Effort: Time Sink under current evidence. Impact Criticality: Medium. Dependencies: an accepted untyped parent from OpenSpec group 7. Confidence: high that it should not precede strict baseline evidence; low that it will improve results.

**Balanced Scorecard:** Financial proxy: added compute cost, indicator = matched runtime delta. Customer proxy: strict improvement, indicator = task pass-at-1 delta. Internal Process: sign preservation, indicator = zero violations after update. Learning & Growth: biological hypothesis evidence, indicator = matched candidate result.

**Recommendation:** **Defer.** Test Dale candidates only after an accepted untyped parent exists. Cut AMPA, GABAa, NMDA, extra channels, compartments, morphology, neuromodulation, and persistent memory from the current phase. Reconsider one mechanism at a time only after a direct need is observed.

## 7. Implementation-truth plot and documents

**Outcome and JTBD:** When reviewers or future contributors inspect the experiment, they need concise artifacts that distinguish observations from inferences so they can understand what executed and what the evidence supports.

**Kano and beneficiaries:** Basic for research governance. Primary: reviewers and maintainers. Secondary: prospective collaborators. Inconvenience: documents must be revised when execution changes.

**Pain, workaround, and cost of not building:** Raw JSON and code inspection are slow and invite overclaiming. Without implementation-truth documents, decisions are harder to audit and knowledge is lost. These documents cannot substitute for Gate 4 evidence.

**Canvas:** Value proposition: transparent and reusable knowledge. Channels are repository documents and plots. Key activities are evidence curation and causal explanation. Cost is maintenance. Revenue impact is unknown.

**Porter:** Dominant impact is Outbound Logistics and Service for research evidence. It also supports Infrastructure. The burden is synchronization with executed code.

**VRIO:** Valuable but easy to imitate. Organized documentation discipline supports reliability. Position: parity.

**SWOT:** Strength: improves auditability. Weakness: stale documents can mislead. Opportunity: faster review and onboarding. Threat: polished narratives can outrun evidence.

**PESTEL:** Social and legal claim accuracy is material. Other factors are not material.

**Priority:** RICE not defensible. Estimated complexity 3/10, S. Impact/Effort: Quick Win after stable evidence exists. Impact Criticality: High for release, Medium now that Gate 4 evidence exists. Dependency: accepted stable checkpoint and structural disposition. Risk: documenting intended rather than observed behavior. Confidence: high.

**Balanced Scorecard:** Financial proxy: reduced review and onboarding rework, indicator = review corrections. Customer proxy: clarity, indicator = reviewer acceptance. Internal Process: traceability, indicator = every claim linked to direct evidence. Learning & Growth: retained knowledge, indicator = resolved observation versus inference labels.

**Recommendation:** **Build after structural decisions, before retirement and release.** Gate 4 evidence is available. Update only claims supported by the accepted checkpoint.

## 8. Obsolete-path retirement, independent validation, and release

**Outcome and JTBD:** When the replacement has passed its evidence gates, maintainers need one supported Example 21 path and independent release validation so users do not encounter conflicting commands or unreviewed claims.

**Kano and beneficiaries:** Basic. Primary: maintainers and technical users. Secondary: reviewers. Inconvenience: removal breaks the old unsupported Example 21 command surface.

**Pain, workaround, and cost of not building:** Parallel old and new paths increase confusion and maintenance. The workaround is to label the old path obsolete. Early retirement would remove fallback and diagnostic value before the replacement is validated. Skipping independent review weakens all prior evidence.

**Canvas:** Value proposition: a clear, supportable release surface. Channels are repository commands, documentation, and release artifacts. Key activities are removal, validation, review, and release control. Cost value is reduced maintenance surface. Revenue effects are unknown.

**Porter:** Dominant impact is Outbound Logistics and Service, supported by Infrastructure. It reduces ongoing support load but adds migration and release-check work.

**VRIO:** Release hygiene is valuable but common and imitable. Position: parity.

**SWOT:** Strength: simpler supported surface and independent assurance. Weakness: loss of old diagnostics. Opportunity: clearer adoption and maintenance. Threat: premature deletion or self-approval.

**PESTEL:** Legal dependency and data notices must remain correct. Technological compatibility and reproducibility are material. Other factors are unknown.

**Priority:** RICE not defensible. Estimated complexity 5/10, M. Impact/Effort: Quick Win only after dependencies; otherwise a Time Sink. Impact Criticality: Critical for release, Low for the learning verdict. Dependencies: task 6.4, structural and Dale dispositions, truth documents, and independent validation. Gate 4 execution evidence is complete. Confidence: high.

**Balanced Scorecard:** Financial proxy: reduced maintenance burden, indicator = active Example 21 paths. Customer proxy: predictable use, indicator = supported command and documentation checks. Internal Process: release assurance, indicator = independent validation and clean change checks. Learning & Growth: preserved rationale, indicator = evidence retained before deletion.

**Recommendation:** **Defer, then Build in the release phase.** Retire the old path only after the replacement is validated. Independent review must precede release. The Release Manager alone finalizes the release.

## Portfolio synthesis

### Executive summary

Brainiac has completed the bounded Gate 4 mechanism proof. It has not shown useful ARC learning, strict task success, or an efficiency advantage. The next priority is OpenSpec task 6.4 because it closes the test and coverage gate. After that, obtain strict baseline evidence before high-cost structural or biological expansion.

**Strategic thesis:** Prove direct online learning and strict behavior in the smallest bounded biological model before buying biological detail, structural complexity, or release polish.

### Comparable priority ranking

This ranking uses mission impact, evidence value, and dependency position. It does not use incomparable RICE scores.

| Rank | Feature group | Decision | Dependency override |
|---:|---|---|---|
| 1 | Task 6.4 test and coverage gate | Build now | Closes the remaining Gate 4 quality boundary |
| 2 | Task 7.1 measurement foundation | Build after 6.4 | Enables defensible structural decisions without authorizing pruning |
| 3 | Tasks 9.2 and 9.3 truth documents | Build against stable evidence | Required for auditability; must follow the accepted execution state |
| 4 | Tasks 7.2 through 7.6 structural arms | Defer | Pruning is blocked until one strict validation pass; additions still require measured deficit and strict gain |
| 5 | Tasks 8.1 through 8.3 Dale candidates | Defer | Requires an accepted untyped parent and strict gain |
| 6 | Task 8.4 optional-biology guard | Build with group 8 boundary work | Cheap scope protection; no optional mechanism is authorized |
| 7 | Tasks 9.1 and 10.1 through 10.3 plot and retirement | Build later | Requires a stable accepted checkpoint and replacement surface |
| 8 | Tasks 11.1 through 11.3 final validation | Build last | Release dependency, not an earlier learning feature |
| 9 | Extra biological mechanisms | Cut from current phase | No observed need or strict value |

### Priority matrix

| Quadrant | Items |
|---|---|
| Quick Wins | Task 6.4; task 8.4 guard; truth documents after stable evidence; retirement after validation |
| Major Projects | Structural adaptation; Dale candidate implementation |
| Fill-Ins | None supported by current evidence |
| Time Sinks | Dale and richer biology before strict baseline evidence; retirement before replacement validation |

### Balanced Scorecard roll-up

| Perspective | Portfolio objective | Indicator |
|---|---|---|
| Financial / resource | Bound the cost of each decision | Direct wall time, peak memory, artifact size, and failed-run count |
| Customer / evidence consumer | Produce credible direct behavior evidence | Actual prediction grids, targets, exact query values, strict task pass-at-1, and independent gate outcomes |
| Internal Process | Preserve valid causal and release boundaries | Target isolation, sparse invariants, validation-state invariance, dependency gates, and independent release approval |
| Learning & Growth | Learn which mechanism changes behavior and strict outcomes | Matched interventions, weight movement, prediction change, and one-variable structural or biological arms |

### Portfolio buckets

**Must-haves:** task 6.4, preserved compatibility and ARC-contract regressions, implementation-truth documents, independent validation, and controlled release. The sparse baseline and Gate 4 proof are completed prerequisites.

**Differentiators:** useful strict ARC learning with PP-Prop; measured sparse structural adaptation. These are hypotheses, not demonstrated advantages.

**Option bets:** Dale typing and one later slow biological mechanism after a direct need is measured.

**Cuts for this phase:** AMPA, GABAa, NMDA, extra channels, compartments, morphology, neuromodulation, persistent memory, synthetic qualification, partial-credit scoring, routing systems, reranking, and larger ARC runs before the bounded proof.

### Surprises and opportunity cost

- Final validation is critical for release, but it creates little new learning value until active feature groups and their dispositions finish. Its criticality is conditional on release.
- The compact contracts and fixtures are cheaper than model innovation but have high leverage because they prevent invalid evidence.
- Structural adaptation is attractive and mission-aligned, but it is a poor current choice. It can consume major effort before the baseline shows useful learning.
- Biological fidelity is not a business-value proxy. More mechanisms can reduce iteration speed and causal clarity.

### Sensitivity

The top recommendation changes only if one of these facts changes:

1. If task 6.4 fails its coverage or runtime gate, priority remains quality diagnosis. It does not move to structural work.
2. If leadership declares biological fidelity more important than learning quality and efficiency, Dale and mechanism work rises. This would change the stated mission evaluation order and must be explicit.
3. If a customer, funder, or partner requires a specific biological feature, that evidence can raise its impact. No such requirement is supplied.
4. If a fixed baseline screen records a strict pass and a measured capacity deficit, structural work rises. With zero strict passes, pruning stays blocked by specification.

### Decision support

**Build now:** complete OpenSpec task 6.4. Preserve the completed compatibility, contract, baseline, and Gate 4 proof evidence.

**Defer:** structural execution arms, Dale candidates, plot, retirement, and release in dependency order. Task 7.1 can follow 6.4 as a measurement foundation. Truth documents must follow the stable accepted execution state. The Project Manager owns exact roadmap sequencing.

**Cut now:** richer biological mechanisms and larger experiments that do not answer the current mechanism question.

Alternatives considered were to proceed directly to structural adaptation, add biological realism first, or clean and release the current surface. Each has lower decision value because it spends effort before the direct learning mechanism is proved.

The smallest unresolved evidence step is OpenSpec task 6.4: run the complete co-located Example 21 test module, measure meaningful coverage above 90%, verify completion below 60 seconds, and confirm that each declared limit fails closed. After it passes, the smallest strategic evidence step is a fixed strict baseline screen. A strict pass plus a measured capacity deficit can unlock the applicable structural arm. A zero strict result keeps pruning blocked and makes biological expansion a poor use of effort.
