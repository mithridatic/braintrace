# Reserved layer-2 input

Use sweep 43 as a subthreshold calibration datum and sweep 50 as the
active calibration datum. Reserve sweep 53 from subsequent local fitting.
Select it from stimulus metadata as the lowest available long-square
command at least 100 pA above the reported 200 pA rheobase. This gives
310 pA. Do not inspect its direct voltage response to choose parameters.

Freeze the candidate parameters and calibration decisions before
extracting the reserved response. Verify source units, bias, quality,
and pulse timing before interpreting the comparison. Retain each event,
waveform, and subthreshold trajectory. Failure on this input prevents
claiming cross-input qualification even if sweep 50 passes.

The source fit lists sweep 51. The reservation applies to our subsequent
fitting, not to unknown decisions by the source-model developers. This
is the same donor and session, not external biological validation.
The [reservation record](../evidence/h01-l2-reserved-input.json) records
the source hash and selection metadata. No reserved voltage response
is extracted in this step.
