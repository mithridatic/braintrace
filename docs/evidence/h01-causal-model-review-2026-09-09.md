# H01 causal model review record

Date: 2026-09-09.
Starting code revision: `41eb735`, branch `feat/h01-braincell`.
Diagram additions reviewed through `da552a7`.
Scope: documentation review. No simulation or model change.

## Book concepts

The local book is David J. Hartshorne's *Diagnosing Performance and Reliability*.
The review used its introduction, chapter concepts, and summary sections.
It did not check every worked example or every image against the transcription.

| Source | Concept used | Application |
| --- | --- | --- |
| [Introduction](<C:/Users/J/Documents/Diagnosing Performance and Reliability/01-introduction.md>) | Conditions and mechanism form a causal explanation. | Separate part names from explanations of response. |
| [Chapter 1](<C:/Users/J/Documents/Diagnosing Performance and Reliability/02-chapter-1.md>) | Progressive search and strong response contrasts. | Rank specified interventions without asserting a universal dominant cause. |
| [Chapter 2](<C:/Users/J/Documents/Diagnosing Performance and Reliability/03-chapter-2.md>) | Direct response, location, time, and paired energy variables. | Preserve voltage, current, and event phases. |
| [Chapter 3](<C:/Users/J/Documents/Diagnosing Performance and Reliability/04-chapter-3.md>) | Nested variation and measurement discrimination. | Separate within-spike variation, train drift, cell differences, and repeat variation. |
| [Chapter 4](<C:/Users/J/Documents/Diagnosing Performance and Reliability/05-chapter-4.md>) | Input, function, and their interaction. | Match inputs and state before assigning an implementation cause. |
| [Chapter 5](<C:/Users/J/Documents/Diagnosing Performance and Reliability/06-chapter-5.md>) | Controlled swaps and interacting parts. | An unrun combination cannot be excluded by an additive prediction alone. |
| [Chapter 6](<C:/Users/J/Documents/Diagnosing Performance and Reliability/07-chapter-6.md>) | Energy storage, flow, and source-load boundaries. | Retain membrane charge, signed currents, ion reservoirs, and state feedback. |
| [Chapter 7](<C:/Users/J/Documents/Diagnosing Performance and Reliability/08-chapter-7.md>) | Function boundaries expose gaps in knowledge. | Report construction, runtime, transfer, and physiological checks separately. |

These are diagnostic methods, not proofs of universal neuronal behavior.
The review does not adopt universal claims about randomness or a single dominant variable.
Active channel states and interactions remain explicit.

## Corrections and their basis

| Previous problem | Corrected statement | Direct basis |
| --- | --- | --- |
| Repeated historical accounts contradicted current results. | One current account per behavior; a separate history copy. | [Original snapshot](../h01-causal-model-history-2026-09-09.md). |
| Absence of M current was proposed as an explanation. | B3 already applies somatic Im; its role in the remaining defect is unresolved. | [Source audit](h01-e-gain-source-audit.json), [density table](../../braintrace/datasets/_h01_ei_parameters.py). |
| Failed doses excluded an entire passive family. | Only the tested interventions failed. The combined arm was not run. | [Gain decision](h01-e-gain/stage-close-decision.json), [source amendment](h01-e-gain-source-audit.md). |
| A high sodium-availability value excluded all sodium explanations. | The observation weakens the specified local explanation; broader mechanisms remain open. | [Trough audit](h01-i-trough-audit.json), [tested density change](h01-i-reserve/stage-1-decision.json). |
| Accepted transfer identity obscured a failed strict prediction. | The amended decision and the failed literal prediction are separate facts. | [SP2 decision](h01-i-transfer/sp2-close-decision.json), fields `identity_gate` and `prediction_outcomes`. |
| The corrected all-104 model remained described as construction-blocked. | Saved decisions report construction and initialization passes; compiled runtime has no completed verdict. | [Construction](h01-104-corrected-construction-decision.json), [initialization](h01-104-corrected-initialization-decision.json). |
| Numerical and anatomical changes accumulated in one narrative. | Calcium integration and component choice address separate boundaries. | [State probe](h01-ready-cell7196644737-state-decision.json), [component evidence](h01-soma-component-anchor-comparison.json). |
| Numerical finiteness implied physical validity. | The implicit calcium repair retained the extreme voltage on historical anatomy. | [Repair decision](h01-ready-cell7196644737-implicit-decision.json). |
| A repeated chart offset was treated as one mechanism. | Repeated differences constrain an explanation but do not identify a unique cause. | [E chart](h01-multivari-e.png), [I chart](h01-multivari-i.png), [feature summary](h01-multivari-chart.json). |

The underlying mistake was extending a local result beyond its conditions.
Future explanations need a source check, an explicit boundary, and a result that could contradict them.
A campaign stop is a resource decision. It is not a proof that untested models fail.

## Code review boundary

The review read the anatomy, region assignment, donor profile, channel, calcium,
construction, compiled solver, connectivity, output-site, and scoring modules.
The current causal document links each function to its source.
This was a semantic review, not a complete software audit.

Other work changed source files while this review was in progress.
Those edits are outside this documentation change and have no new qualification from this review.
Historical results retain their recorded source versions.

The history copy includes the concurrent diagram additions through `da552a7`.
The current model retains their per-cycle observations with narrower causal claims.
The [diagram spec](../specs/2026-09-09-h01-diagnostic-diagrams.md) identifies all associated outputs.
The six recorded model reruns belong to that separate work.

The diagrams are useful observations or conceptual summaries, not independent tests of their captions.
The old search-tree labels cannot exclude untested parameter families.
The source-load diagram cannot establish measured impedance where no impedance measurement exists.

## Language and evidence limits

The writing follows the short-sentence and single-topic structure in
[ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).
The document defines its technical terms and uses consistent names.
A prose check does not certify every dictionary word or technical term.

Saved decisions support the reported measurements.
This review did not rerun the simulations or independently reconstruct all source hashes.
Some trace references depend on local cache files or untracked arrays.

## Initial verification

All 74 local links in the current model, this record, and the review spec resolve.
The prose check found no sentences longer than 25 words in the current model.
No checked prose paragraph exceeds six sentences.
The history body matches the document at `da552a7` exactly after line-ending normalization.
Its added banner identifies the text as historical.
The current explanation is approximately 87 percent shorter by whitespace word count.

The change affects four documents only: the causal model, history, review record, and spec.
No model code or model tests were changed by this review.
No simulation was run. Existing source edits and output files remain outside this change.

## Current-path follow-up

The [follow-up record](h01-causal-current-observations-2026-09-09.md) adds measurements from retained I traces and checks B3 recording fields.
The I samples show large axial outflow after the trough, despite recovered sodium availability.
The earlier SK arm also changed somatic sodium density and used a different Kv3 closing factor from the finalist.
Its result cannot isolate the finalist's SK effect.

B3 has no saved channel-current observations in the inspected traces.
The earlier E current budgets used different parameters.
The causal document now states these limits and links specific tests for the missing evidence.
The original I result has an amendment to prevent reuse of its excessive necessity claims.

The error was relying on a test label without checking all applied parameters.
Future comparisons must verify the full parameter difference before attributing a response to one change.
This follow-up changes documents and derived evidence only.
It runs no simulation and changes no model parameters.

The follow-up check resolves 88 local links across the five reviewed documents.
The revised causal model, new evidence note, and follow-up spec contain no checked prose sentences longer than 25 words.
The tracked changes pass the whitespace check.
These checks do not certify full ASD-STE100 dictionary compliance.
