# Original recording provenance

The original Allen recording is specimen 528687520, NWB file 618112937.
The local file has SHA256
`b6412208aaeee590bfe73cbbc3f3e3d4da0622252d0a7b09961d3dc3595f27a4`.
The specimen name agrees with the source morphology name.

The four calibration exports match NWB sweeps 20, 23, 35, and 39.
Each export starts 37500 samples, or 750 ms, into the original sweep.
Each has the -14 mV junction correction applied.
All 363499 samples per export agree within 0.000006 mV.
The reserved 0.23 nA trace was not read by this audit.

| Step, pA | Measured bias current, pA |
| ---: | ---: |
| -110 | 25.62674 |
| -50 | 28.78366 |
| 190 | 31.44537 |
| 270 | 31.44537 |

The bias is stored separately from the stimulus waveform.
The published optimizer and our reference omit it.
Do not assume that adding it alone will repair the model.
Fitted leak parameters may already compensate for its omission.
Test the baseline and slow channel states with the bias before further fitting.

Allen's [acquisition white paper](https://s3.amazonaws.com/webflow-prod-assets/689cfbd308fa7373b604d290/68ee796e6df1cc984c3ab432_Documentation_Cell_Types_Database-Electrophysiology_Overview.pdf)
describes a 10 kHz Bessel acquisition filter and bias current used to maintain
the initial resting voltage. It reports uncorrected acquisition voltages.
The file sampling rate is 50 kHz. The white paper is protocol evidence;
it is not a per-sweep measurement of the filter transfer function.

The first inspection incorrectly applied the NWB conversion attributes.
Pipeline 1.0 already stores SI values. The attributes are obsolete.
The [Allen SDK reader](https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/core/nwb_data_set.py)
explicitly handles this case. The saved audit checks the pipeline version
before reading values and verifies the complete exported voltage arrays.
This prevents a second unit conversion from entering later comparisons.

Run `python -m docs.evidence.h01_pv_acquisition_audit` with h5py 3.14.0.
The [JSON report](h01-pv-acquisition-audit.json) retains source hashes and errors.
The NWB file remains in the cache. No corrected trace replaces the source data.
