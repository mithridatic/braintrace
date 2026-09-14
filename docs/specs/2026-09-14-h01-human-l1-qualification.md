# Human L1 donor qualification: recording preparation

This implements the approved human-unity plan on campaign/h01-human-unity-20260914.
The six-term, 104-cell objective and all existing score requirements remain intact.
The selected acquisition is specimen 811953283, session 811953264, DANDI 000630
version 0.230915.2257. Its NWB SHA256 is
004f27f306180bd610ba2439e600f688c11be24adf5cea2a1663a22ba0336af6.

## Preparation and correction contract

Use the pinned author metadata and recording. Decode the MIES notebook with
explicit headstage (0) or global (8) scope: global online QC values are absent
from headstage 0. Preserve all online QC flags as source observations; they
must not be relabeled offline recording quality, since stimulus-search acceptance
can depend on evoking a spike. Compare with independently recomputed IPFX QC
using its unchanged criteria and pinned source revision. Retain failed sweeps.

Respect the last finite notebook value for each sweep. Holding current is zero
only when its enable flag is explicitly zero. Enabled holding without a value
is unavailable, never silently zero. Preserve bridge balance as instrument
metadata, without applying a second correction to an already compensated trace.
Keep the NWB command and separate amplifier holding current distinct.

Read acquisition and stimulus using their stored conversion factors, SI offsets,
units, sample rates and start times. Reject mismatched clocks, wrong clamp modes,
missing units and nonfinite samples. Preserve all samples and original timestamps.
MIES trailing storage zeros must be marked unavailable, consistent with the IPFX
recording-epoch convention. Preserve the source array separately; exported recorded
voltage uses NaN outside a retained validity mask. An entirely zero response is
unavailable. Do not infer full observed coverage from the stored sample count.
Preserve raw voltage. Apply no liquid-junction correction until its source and
recording convention are verified. The notebook temperature values near zero
Celsius are not a credible bath-temperature reference; resolve the protocol
before choosing mechanism temperature. Missing evidence does not authorize
using B3's correction or temperature by analogy.

## Frozen data use before optimization

Calibration candidates: sweeps 4-12, 14, 16. Held-out candidates: sweeps 13, 15,
and 17-34. The split uses sweep identity and stimulus family, not fit scores.
Sweeps 0-3, 35-36 are voltage-clamp setup/ending checks, never sodium-current
kinetics evidence. Offline QC may remove a failed sweep from use but may not move
it between roles or substitute another after seeing a model result. Record every
exclusion and all original candidates. All short-pulse and ramp families stay
outside parameter fitting, providing independent-input tests.

Exposure disclosure: raw-array finiteness, source spike-array lengths, and author
aggregate QC were inspected before this split. No fitting or optimization has
occurred, no held-out voltage plots have been inspected, and no model prediction
exists. These are preregistered independent-input tests, not untouched external
specimens. Later external-donor validation must use separately reserved specimens.

## Implementation and verification

Add docs/evidence/h01_l1_recording.py with sibling *_test.py. Cover notebook
scope, missing/disabled holding values, chronology, unit conversion, mismatched
clocks, nonfinite data, source hash mismatch and held-out export rejection.
Only calibration responses may be exported by this preparation tool. An eventual
holdout evaluation must freeze a candidate and prediction before accessing its
response. Target greater than 90 percent line coverage. Use local CPU only;
this stage executes no model and buys no compute.

Follow preparation with protocol resolution, human-mechanism selection, model
fitting and independent testing. A preparation pass is not a donor_accuracy or
type_coverage pass. Record direct voltage/current observations and visually inspect
the calibration data before interpreting physiology under the observation contract.

## Protocol and mechanism source follow-up

Resolve the recording reference from this study's own methods and retain a new
protocol receipt beside the unchanged preparation artifacts. Record protocol bath
temperature as a range, not an exact specimen measurement. Record the stated
junction potential separately from an applied voltage transformation. Preserve the
raw-reference exports until the signed transformation is explicitly qualified.

Inspect the pinned author notebook and its archived release for the nucleated-patch
potassium-current inputs. Distinguish saved notebook outputs, complete source
tables, and time-resolved raw currents. Verify archive size/checksum before reading
its member list; do not execute upstream notebooks. A missing file in these two
releases establishes an acquisition gap there, not proof that no public human
measurements exist. Retain exact source identities and the search boundary.

This stage is read-only source analysis plus new evidence receipts. It neither
fits a model nor accesses held-out response arrays. The existing eight/nine
calibration/holdout QC split and the six population values remain unchanged.

### Recover raw channel experiments from published NWB sessions

Join the human cell names visible in the author channel-table preview to pinned
metadata specimen and Ephys_Roi_Result identifiers, then join the latter to DANDI
session identifiers. DANDI subject identifiers identify tissue subjects and must
not be mistaken for individual cell identifiers. Start with non-excluded human
cells; retain any excluded match in the inventory without using it to fit.

Fetch at most three eligible files, capped at 170 MB combined and 80 MB each,
checking upstream size and SHA256. Inspect acquisition/stimulus identities and
actual voltage commands to distinguish nucleated-patch step families from setup
checks. Do not infer channel evidence from VoltageClampSeries alone. Preserve
all original data, clocks, conversion factors, source hashes and command families.
Any comparison to the author's table is reproduction, not independent validation.
Newly observed channel records are discovery/calibration material, not holdouts.
This local CPU source audit runs no membrane model or optimization.

Implement `docs/evidence/h01_l1_voltage_clamp.py` with sibling tests to export
registered nucleated-patch sweeps from the hash-pinned session 923103553. Reconstruct
the command from the stored DAC waveform plus explicitly enabled amplifier holding
voltage, keeping both separately. This is a command, not measured patch voltage;
do not label the measured total current as isolated potassium current. Preserve
SI conversions/offsets and notebook compensation metadata. Reject wrong source,
mode, family, holding units, clocks, nonfinite arrays and ambiguous terminal zeros.
Zero current can be real: never silently strip it using the current-clamp voltage
padding heuristic. Retain all accepted samples and report unresolved coverage/QC.

Export a deterministic low/middle/high selection in the total, prepulse and
sustained families, their first leak controls, and first/middle/last recovery
conditions. Inspect common-scale current/command plots before deriving kinetic
parameters. The first extraction is discovery evidence and changes no metric.

After recovering the LAMP5 source, search eligible PAX6 metadata for an appropriate
kinetic source under the same 170 MB combined acquisition cap. Restrict to human,
exclude=No, transcriptomic QC true, Nucleated extraction and matching CDH12 type;
rank by descending NWB byte size as a discovery heuristic for additional sweeps,
not by current or fit outcomes. Record this selection rule before opening arrays.
Keep all whole-cell current-clamp responses in this new specimen unopened. A
source match by type does not establish kinetic equivalence between specimens.

The largest PAX6 file (session 902511350) contains additional whole-cell stimulus
families but no nucleated-patch channel families. Retain this negative inventory.
For the third and final acquisition in this audit, inspect the next eligible file
(session associated with specimen 839897427). Raise only the combined byte cap
to 210 MB; the three-file and 80 MB per-file limits remain. This candidate also
shares the H19.06.351 donor with a cell in the author channel-table preview. No
whole-cell responses or fit outcomes inform this follow-up selection.

Both inspected PAX6 files lack nucleated-patch families. Before any further full
download, allow HTTP Range reads of HDF5 headers for the next three eligible
PAX6 files. Cap each probe at 2 MB of actual fetched bytes; require HTTP 206 and
the exact requested Content-Range. Inspect the last acquisition's family as a
discovery screen, not an exhaustive negative inventory. Cache fixed-size blocks
locally in memory. These partial reads cannot verify a whole-file digest and
must remain explicitly preliminary until a positive source is fully acquired.

The 64 KiB block probe exceeded the per-file cap because sparse metadata occupied
many blocks. Repeating with 8 KiB blocks stayed below the same cap and identified
`NucVCzrecovs1_DA_0` in PAX6 CDH12 session 840043481. Acquire this positively
screened file as the fourth full file, increasing the full-file total cap to
265 MB. Preserve the preliminary range receipts and both negative full inventories;
do not treat the failed range probes as absent channel evidence. This additional
acquisition directly addresses the selected donor's transcriptomic class.

## Prospective kinetic calibration and external validation

Use PAX6 CDH12 specimen 840043506, session 840043481, donor H19.06.351 as the
channel-calibration source. Its full hash-verified NWB contains nucleated-patch
sweeps 70-141. Before decoding its currents, reserve PAX6 CDH12 specimen
835648767, session 835648738, donor H19.03.306 for external kinetic validation.
Only author metadata and its last acquisition header have been inspected; the
full file and response arrays remain unopened. This exposure is disclosed in
the source contract. Freeze candidate kinetics, protocol predictions, QC and
scoring rules before accessing those validation currents. Do not tune against
them. Preserve the original specimen 811953283 whole-cell data split unchanged.
LAMP5 specimen 923103580 remains discovery/reproduction material, not a holdout.

## PAX6 current preparation and leak-control assessment

Extend the exporter with an explicit session argument limited to the two pinned
discovery/calibration sources. Reject the reserved external validation session
before file access. Register the PAX6 family mapping from the acquired inventory
(70-141); retain all existing LAMP5 identities and behavior. Test both sources,
wrong source/session combinations and the external-response access boundary.

Export PAX6 total-current sweeps 70, 74, 78, prepulse 89, 93, 97, sustained
104, 108, 112, all leak controls 79-88 and 113-122, and recovery 123, 132, 141.
Retain all samples and instrument metadata. Assess repeated control stability
using their full waveforms, with a fixed 900-990 ms baseline and 1100-2100 ms
command window checked against the actual waveform. Compare the first and last
control, preserving repeat identity. Any linear subtraction is a conditional
estimate, not proof of channel isolation; assess its sensitivity to the selected
control and retain capacitive transients. A control at one amplitude cannot
establish linearity over the full depolarizing range. No external response is
opened and no kinetic parameters are fitted until these observations are reviewed.

Implement conditional plain-step subtraction as baseline-centered response minus
the baseline-centered control scaled by their command increments. Preserve raw,
scaled-control and conditional-current arrays separately. Require identical finite
clocks, complete pulse coverage, a constant pre-step command through pulse onset,
a constant nonzero pulse, return to baseline, and matching holding potentials
within 0.1 mV. Reject prepulse/recovery histories: one scaling factor cannot model
their different capacitive and steady-state offsets. Use total and sustained
families only. Test against an analytic passive-current plus active-current oracle,
and reject incomplete windows, different holdings, zero controls, extra transitions
and nonfinite traces. This calculation is an explicit linear-response assumption,
never itself a gate passing human potassium-current isolation.

After visual inspection, the conditional +70 mV total-current estimates using
controls 79 and 88 show a consistent large decaying component beyond the onset
artifact. Fit each on 1110-2090 ms, retaining all samples, with the descriptive
envelope C + A1 exp(-t/tau1) + A2 exp(-t/tau2), relative to 1110 ms. Use nonnegative
amplitudes bounded at 10000 pA, offset -1000 to 1000 pA, and both time constants
1-10000 ms. Report optimizer status, boundary proximity, original data, predictions,
residuals and sensitivity to the control. These are conditional decay time scales,
not identified A/D molecular channels, voltage-dependent rate laws or a physiological
qualification. The 10 ms onset exclusion follows the observed control transients;
it cannot establish behavior of faster gates. Do not access external currents.
