# Wilbers voltage-command source boundary

The release contains a voltage waveform, `Figures5_6/APclamp/H_wide.csv`.
It has 800752 samples in one column. The sodium AP-clamp script uses
125 samples per millisecond and subtracts 10 mV from a selected window.
The helper sends that waveform to a voltage clamp. It records a simulated
sodium current. The helper uses a sodium reversal potential of 141 mV
and a temperature of 34 degrees Celsius for this channel assay.

Thus, the source demonstrates an imposed-voltage channel test. It does
not establish a matched current input for a freely firing cortical cell.
Do not transfer the assay reversal potential or voltage shift to the
whole-cell model without checking the recording conditions.

The waveform contains 240 complete excursions above -20 mV before the
script's voltage shift. This is a description of the released command,
not a validated spontaneous or current-evoked human firing response.
The acquisition identity, generating current, and preprocessing remain
unverified. No excitatory-cell target is promoted from this file.

The [manifest](h01-wilbers-command-waveform.json) records file identifiers,
SHA-256 hashes, verified release checksums, the source clock, and the first
five direct excursions. Files are cached under `.cache/wilbers2023`.
The source scripts were inspected, not executed. Their Python stepping
loops must not be copied into the BrainCell simulation driver.

Sources: [dataset release](https://doi.org/10.34894/L5J0SD),
[voltage command](https://dataverse.nl/api/access/datafile/318541),
[sodium assay script](https://dataverse.nl/api/access/datafile/318544), and
[assay helper](https://dataverse.nl/api/access/datafile/318538).

## Recording-method check

The [paper's methods](https://research-portal.uu.nl/ws/files/206755646/sciadv.ade3300.pdf)
explain the sodium assay's -10 mV command shift: it compensates for the
junction-potential difference between CsCl and K-gluconate solutions.
It is not a fitted correction to a whole-cell voltage mismatch.

The whole-cell protocol used 3 ms pulses at 150% of each cell's short-pulse
rheobase, measured in 10 pA steps. Thus, a typical current from the paper
cannot replace this waveform's missing individual stimulus amplitude.
The paper identifies the assay commands as previously recorded waveforms.
The specific recording identifier and preprocessing chain for `H_wide.csv`
remain unverified. This narrows the provenance gap but does not close it.

## Candidate recording label

The Figure 1 workbook defines its filename column as an individual
recording number. Among 36 human 40 Hz rows, the first five values of
threshold plus amplitude match the command's first five sampled peaks
for `H20.29.179.11.91.01.nwb`, labelled Frontal. The maximum difference
is below 5e-14 mV. The next complete candidate differs by up to 0.596443 mV.

This is a strong numerical association, not an explicit source-manifest
link. The raw NWB file, stimulus amplitude, and preprocessing chain are
still required before accepting a matched current-input/voltage-output
reference. Do not relabel the H01 anatomy from this other recording.

The [first-peak search](h01-wilbers-waveform-label-search.json) and
[five-peak comparison](h01-wilbers-waveform-label-five-peaks.json) retain
all candidate rows. Two rows lack required features and are marked
incomplete. The initial analysis assumed complete columns and raised a
missing-column error. It now retains missing values explicitly; future
matching must validate completeness before calculating a comparison.
