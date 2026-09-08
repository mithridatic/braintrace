# Candidate direct pyramidal datum from Allen

Allen's specimen query links recording name `H16.06.013.11.18.02` to
specimen 562381210, ephys result 562381187, and NWB file 743065376.
The published ModelDB optimizer names a morphology with this recording
prefix. This supplies a candidate reference path, not proof that the
optimizer's manual targets came from this cell. The Allen structure label
is middle temporal gyrus. Allen's cell-detail endpoint identifies layer 5,
spiny dendrites, an intact apical dendrite, and a full reconstruction.
Its reconstruction ID 696537235 matches the published morphology filename.
These annotations support a candidate pyramidal identity; they are not a
direct assay of transmitter release. Layer 5 differs from the H01 layer-2
example and must remain explicit in any physiological transfer.

The [downloaded NWB](https://api.brain-map.org/api/v2/well_known_file_download/743065376)
contains paired current and voltage series with conversion factors of 1
to amperes and volts. Local SHA-256 is
`729ba18dc09524659f7aa187767667e34c135e2f50c5af29657dee4178a7e49f`.
This is a locally computed digest, not a separately published checksum.

Sweep 52 has a contiguous positive command from 1020 to 2020 ms, at
approximately 0.11 nA. Its separate bias-current field is 0.009433365 nA.
Sampling is 50 kHz. The extraction retains all 11 complete excursions
above -20 mV. First rising crossing is 1058.189623 ms, sampled peak is
35.625004 mV, and duration above -20 mV is 1.064763 ms.

These voltages retain the file's stored datum. No additional junction
correction was applied. Check that correction and the bias convention
before comparing with channel reversal potentials or other recordings.
The bias field explicitly uses amperes. Blank per-series comments,
description, and source fields do not resolve the junction correction or
whether acquisition bias is included in the delivered stimulus waveform.
The [datum record](h01-pyramidal-allen-562381210-datum.json) retains all
events, clocks, and qualification limits. Full paired arrays and source
metadata are cached under `.cache/human-pyramidal-modeldb`.

This is a source-linked individual input/output candidate. It is not an
H01 donor recording, a Wilbers recording, or a validated excitatory model.
