# Layer-2 human reference candidate

Allen specimen 541563728 is annotated human, layer 2, spiny, with an
intact apical dendrite and a full reconstruction. Selection used these
annotations before inspection of its voltage response. The recording
is not from the H01 donor and is not a Wilbers recording.

NWB file 618092534 was downloaded to `.cache/human-pyramidal-l2`.
Its local SHA-256 is
`cc180b297d762cb82833eb044566e21f77708c562295f86827ff3ed9ebcf3e0f`.
The file identifies Allen pipeline 1.0. The
[AllenSDK reader](https://raw.githubusercontent.com/AllenInstitute/AllenSDK/master/allensdk/core/nwb_data_set.py)
documents that pre-1.1 files store SI values despite incorrect conversion
attributes. Those attributes were not applied a second time.

The [acquisition protocol](https://s3.amazonaws.com/webflow-prod-assets/689cfbd308fa7373b604d290/68ee796e6df1cc984c3ab432_Documentation_Cell_Types_Database-Electrophysiology_Overview.pdf)
reports uncorrected acquisition voltages and a maintained bias current.
The derived array subtracts 14 mV and retains the original reported
voltage separately. Command and bias remain separate; their sum is also
stored as the interpreted total injected current.

Sweep 43 applies a 110 pA command from 1020 to 2020 ms, with bias
-3.711859 pA. It has no complete excursions above corrected -20 mV.
Corrected voltage during the pulse ranges from -84.531258 to -74.093750 mV.
The metadata reports a 200 pA rheobase. This sweep is retained as a
subthreshold datum, not presented as a spiking response.

The [datum record](h01-pyramidal-allen-l2-datum.json) preserves conventions
and source annotations. Sweep-quality checks and a separately selected
suprathreshold response remain necessary before model qualification.
