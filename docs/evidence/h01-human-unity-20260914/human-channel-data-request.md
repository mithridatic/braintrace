# Draft request for original human channel records

Status: prepared locally; not sent. This request is scoped to unresolved inputs
for the Wilbers pyramidal-channel branch. It is not a claim that all H01 work
requires these files, or that obtaining them will qualify a model.

Intended recipients: the corresponding authors listed in the retained article,
Natalia Goriounova and Huibert Mansvelder. No external communication is authorized
by this document.

Subject: Original human voltage-clamp protocols and analysis inputs for eade3300

We are testing human-only channel models against the measurements in your
pyramidal-neuron study, DOI 10.1126/sciadv.ade3300. We have inspected version 3.0
of the [Dataverse release](https://doi.org/10.34894/L5J0SD), including the Fig. 2-4
workbooks, H_wide.csv, channel_kinetics.py and channel_tools.py. We would appreciate
an existing archive location or clarification for the following inputs:

1. **Potassium recovery protocol and original traces.** Fig. 4F depicts a
   conditioning train labelled x200 at 40 mV, followed by recovery at -80 mV.
   Fig. S5L depicts one conditioning depolarization, while the released
   K_recovery implementation uses a continuous 5000-ms depolarization. Which
   command was delivered for the Fig. 4 recovery workbook observations? The
   original human NWB stimulus/current arrays, sweep identifiers, conditioning
   pulse widths and intervals, recovery delays, and amplitude measurement windows
   would let us reproduce the experimental operator without guessing a waveform.

2. **Sodium AP-clamp correspondence.** For the Hwide fields in data_fig2_na.xlsx,
   please identify the original human NWB sweeps and the mapping from the released
   H_wide.csv samples to the 200 reported amplitudes. We retained all 16 human
   recording rows, including H21.29.194, H21.29.195 and H21.29.197 filename groups.
   Please clarify baseline/leak subtraction, the meaning of leakSubt=0, any
   filtering, and the peak-selection windows used for those amplitudes. The
   command has 240 detected AP peaks in its complete file; our first-200 crop is
   an analysis choice, not a verified experimental sweep mapping.

3. **Original kinetic analysis and QC.** The Na plotting code refers to an
   exclude column that we did not find in the released Fig. 3 workbook. Please
   identify the corresponding QC table and the MATLAB scripts that extracted
   activation, inactivation and recovery observations from the NWB currents.
   The model-fitting code also references Experimental means CSV files absent
   from the 65-file release inventory we inspected. An archive pointer would be
   sufficient. For potassium, clarification of the inactivation time-constant
   dictionary entry labelled ms-1 would also help; the fitted exponential and
   figure imply time constants in ms.

4. **Record linkage and conditions.** Where available, please supply a
   deidentified recording-to-donor/specimen mapping, actual recording temperature,
   voltage-reference convention, series-resistance compensation and QC metadata.
   We currently treat filename groups as groups, not verified independent donors.
   We keep 25 C sodium kinetics separate from the 34 C AP-clamp comparison and do
   not interpret fitted thermal multipliers as measured human Q10 values.

We need only the relevant human records and their experimental metadata, not
identifying participant information. We preserve the published summaries and
failed model predictions, and would cite any additional archive or clarification.

## Local evidence supporting the request

- [Protocol reconciliation](human-channel-protocol/README.md): inspected Fig. 4,
  Fig. S5, Table S2, original command samples, scripts and selected AP amplitudes.
- [Source selection](human-pyramidal-channels/README.md): pinned release inventory,
  workbook identities, missing QC field and mixed-species kinetic fit provenance.
- [Executed sodium comparison](sodium-thermal-transfer-r2/README.md): a shared
  two-factor change reduces combined validation error but fails four individual
  recordings; it was rejected. Original currents would constrain a different
  hypothesis, not retroactively validate that candidate.
- [Retained article XML](human-channel-protocol/paper.xml): correspondence,
  acquisition methods, offline analysis and public-release statement.

The L1 PAX6 raw voltage-clamp recordings belong to a different study and have
already been acquired. This request must not be described as recovery of those
missing data. No H01 score changes follow from preparing or sending a request.
