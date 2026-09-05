# Recorded human response acceptance proposal

Status: proposed. No new physiological pass or fitting run is authorized by
this file. The goal remains a human-constrained active model and measured
H01 circuit, with source anatomy and borrowed physiology kept distinct.

## Meaning of a pass

A pass means that one frozen model reproduces the specified donor recordings
within the engineering limits below. It does not establish population-wide
human accuracy. H01 has no matching measured voltage trace. Transfer to H01
therefore remains a model prediction constrained by other human cells.

These round limits are proposed engineering requirements. They are not
estimated biological standard deviations, optimizer weights, or numerical
solver tolerances. They were not selected to make the current candidate pass.
User agreement is required before applying them as the acceptance contract.

## Direct response requirements

Use the pinned raw records and conventions in
[human event requirements](../evidence/h01-human-event-requirements.json),
[recovery datums](../evidence/h01-human-recovery-datums.json), and the
[E acceptance table](../evidence/h01-l2-acceptance-table.md).
Do not shift or align traces. Check every event separately. No pooled score
can override a failed event or a failed input.

| Measurement | Proposed allowed absolute error |
| --- | ---: |
| Complete event count in the full stimulus window | Exact; no missing or extra events |
| Each rising and falling -20 mV crossing time | 1 ms |
| Each peak time | 1 ms |
| Each peak voltage | 1 mV |
| Each duration above -20 mV | 0.05 ms |
| Each rise-to-peak and peak-to-fall duration | 0.05 ms |
| Each complete recovery minimum voltage | 1 mV |
| Each peak-to-recovery-minimum delay | 1 ms |
| Each specified subthreshold sample voltage | 1 mV |

For a recovery window cut off by the end of a recording, compare the voltage
at the shared endpoint. Do not label an endpoint value a completed recovery
minimum. Report any unobserved recovery time as unavailable, not as a pass.
The observable endpoint must satisfy the 1 mV voltage limit.

One shared E parameter set must meet all released E calibration inputs. One
shared I set must meet all released I calibration inputs. Preserve the I
0.23 nA prospective fitting holdout and unopened E sweep 53. Freeze parameters
and the acceptance contract before evaluating either holdout. A failed holdout
remains a failure; fitting it removes its holdout status.

## Independent numerical and provenance gates

Keep numerical tests separate. They must show that integration, mesh, source
translation, and measurement error are small enough to resolve each claimed
physiological comparison. A small model-to-recording residual cannot excuse
a failed numerical gate. The existing direct numerical limits are unchanged.

Record every input waveform, bias current, initial state, temperature, voltage
correction, morphology, and mechanism version. Resolve a missing mapping before
calling its comparison a pass. Keep the diagnostic H01 I closing-time override
separate from the donor candidate until the complete donor gate supports it.

## Current result and next decision

The frozen I model fails the already established 0.27 nA count requirement:
40 events versus 43. Its first recovery minimum differs by 7.006 mV. The frozen
E model has the right count, but its fifth onset differs by about 64.48 ms.
Thus, neither frozen candidate passes this proposal. These examples do not
replace a full row-by-row assessment.

After agreement, apply the contract to all existing candidate records before
proposing any new bounded fitting campaign. Report all failed observations and
unavailable input mappings. Do not resume an unbounded sequence of parameter
tweaks. Keep the existing source and frozen candidates as controls.
