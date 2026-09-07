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

## Second reservation (2026-09-07)

Sweep 53 was opened once on 2026-09-06 for the energetic-search prediction and
is spent. Sweep 55 (350 pA) is now reserved under the same rules: it is the
highest long-square command not within 20 pA of an opened sweep and not listed
in the source fit (52 and 54 neighbour 53; 51 is the source-fit sweep). The
Allen analysis spike count of every sweep is on disk and was read by the sweep
inventory, so this reservation closes the waveform, timing, peaks and per-cycle
rows, not the count. `h01_l2_sweep_export.py` refuses to export sweep 55 until
`h01-prediction-e2.json` exists, and the driver rejects `--sweep 55` until the
export exists.
