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
