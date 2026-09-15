# Human pyramidal channel source checkpoint

The existing Wilbers potassium model is not a human-only kinetic source:
its inactivation and recovery parameters were fitted to pooled human and mouse
means. The acquired tables provide a replacement input: 19 human sodium
recordings and 13 human potassium recordings, selected with original identities,
row numbers, missing values and variable dictionaries retained. These are
recordings, not counts of independent donors. No mechanism parameters or six-term
scores changed at this checkpoint.

## Source and acquisition

The source is the L2/L3 pyramidal study, [Wilbers et al., eade3300](https://research-portal.uu.nl/ws/files/206755646/sciadv.ade3300.pdf),
and its [Dataverse release](https://doi.org/10.34894/L5J0SD), version 3.0, CC0.
This is distinct from the fast-spiking interneuron study eadf0708.
[acquisition.json](acquisition.json) pins the 12 acquired files, published SHA1,
byte counts, SHA256 and release metadata hash. The original workbooks and scripts
are retained byte for byte. Upstream scripts were inspected as text, not executed.

[Sodium records](sodium-human-records.json) select 19 of 34 source rows using
the authors' H/M filename convention. [Potassium records](potassium-human-records.json)
select 13 of 43 source rows and additionally require the explicit Species column
to agree. Nonhuman rows remain available in the original workbooks. Selection
performs no physiological quality filtering. The Figure 2 AP-clamp workbooks are
preserved, but their response values have not been analyzed at this checkpoint.

The workbooks contain derived experimental measurements. The original NWB
current traces have not been acquired. Filenames are retained as acquisition
leads. Sodium kinetics were measured at 25 C and potassium kinetics at 34 C.
The source Q10 of 2.3 has not been independently qualified for human thermal
transfer. Published model conductance densities also are not an independent
human-only calibration. Molecular subtype and spatial distributions remain open.

## Input limitations before fitting

- The sodium plotting script uses an `exclude` column that is absent from the
  released sodium workbook. Its full author QC cannot be reproduced by treating
  that missing column as false. All 19 human rows have RsComp=70; fit-quality and
  voltage restrictions still vary by observable.
- The potassium variable dictionary labels inactivation time constants with
  units `ms-1`. The fitting expression uses `exp(-t/tau)` with time in ms and the
  paper displays time constants. Resolve and record this discrepancy explicitly
  before using those fields in a fit. Recovery fields used in the plot are
  explicitly labelled ms in the dictionary.
- The released potassium plotting script refers to `data_fig5.xlsx` and Figure 5
  names, while the released kinetic workbook is `data_fig4.xlsx`.
- The fitting support code expects `Experimental means/*.csv`; none occur among
  the 65 files in the pinned release. The original fitting workflow is therefore
  not reproduced by this acquisition.
- Some derived inactivation constants lie near apparent fitting bounds. Preserve
  them; do not silently discard them or replace them with the published model's
  constants. Per-record protocol availability and missing values differ.

## Direct observations and verification

[The recovery plot](human-recovery-observations.png) was generated from the
selected source rows and visually inspected. Its [130 plotted observations](direct-observations.json)
retain recording identities. Sodium recovery generally slows with depolarization
and varies substantially between recordings. Potassium has 12 fast and 11 slow
recovery observations, with visible missing-record gaps and a broad spread.
The plot includes all available values without author QC or voltage filtering;
it is descriptive, not a model fit or qualification. Missing values are not zeros.

The species-selection helper has 10 passing sibling tests and 100% line coverage
([test receipt](tests.xml), [coverage](coverage.json)). Tests cover missing,
duplicate, unknown and contradictory identities, exclusion of mouse rows, and
preservation of missing values and QC flags. This verifies selection behavior,
not human physiology. Source hashes and a second workbook-to-JSON comparison
are recorded in [analysis-receipt.json](analysis-receipt.json).

## Next experiment

Register a human-only potassium kinetic estimate with explicit observable,
protocol, source QC, parameter bounds and validation split before fitting.
Resolve current-time-constant versus gate-time-constant semantics and the unit
discrepancy first. Do not reuse cross-species means. Acquisition of original
traces remains necessary for protocol-level waveform validation. Keep the
rejected PAX6 initial-state path closed and existing reserved responses unopened.
Donor coverage, final-population dynamics and anatomy-transfer physiology remain
unqualified; this source checkpoint does not close them.
