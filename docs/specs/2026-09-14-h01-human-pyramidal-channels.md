# Human-only pyramidal channel inputs

Scope: approved human-only mechanism acquisition and replacement under the
all-six-term qualification plan. This stage prepares the available direct human
source constraints before replacing any parameter. Preserve all 104 cells and
the existing donor calibration/holdout rules and tolerances.

Pin Wilbers et al., Science Advances eade3300, and Dataverse 10.34894/L5J0SD
version 3.0. This is the L2/L3 pyramidal study, distinct from the same author's
fast-spiking interneuron study eadf0708. Check file sizes and published checksums,
retain SHA256 and original bytes, and inspect upstream scripts as text only.

The paper states that potassium inactivation and recovery were fitted using
cross-species means. Sodium kinetic parameters were fitted independently by
species. Therefore, the existing H01K_Wilbers2023 implementation cannot be
qualified as human-only merely because the source filename is k_human.mod.
Preserve its equations and historical results; correct provenance descriptions.
Do not redeploy or promote it under the current requirement.

Extract human rows from Figure 3 sodium and Figure 4 potassium workbooks,
retaining all columns, original row numbers, variable dictionaries, missing
values and QC flags. Sodium source species follows the authors' H/M filename
prefix convention. Potassium additionally has an explicit Species column;
the two sources of identity must agree. Reject unsupported prefixes, missing
identities, duplicate identities and contradictory species labels. Nonhuman
rows remain in the preserved original workbook and are excluded from the
human-only derived input. Do not fit parameters to mixed-species means.

These workbooks contain derived experimental measurements, not the original
NWB current traces. Do not label them raw voltage-clamp recordings or assert
that the original NWBs have been acquired. The filenames are acquisition leads.
Retain the known source-data boundary before claiming protocol-level validation.

Sodium kinetics were recorded at 25 C; potassium kinetics and AP-clamp experiments
at 34 C. The source Q10=2.3 does not become independently measured human thermal
behavior because it occurs in a human-labeled file. Subsequent sodium validation
must evaluate at 25 C first, with an explicit separate thermal-transfer check.
Na/K molecular subtype identities and spatial distributions remain unqualified.

This acquisition and species selection changes no mechanism parameter, fitting
split or six-term score. It supplies a concrete human-only input for the next
potassium re-estimation. Before that fitting run, register the measured observable,
source QC, protocol, parameterization, limits and independent validation. A gate
time constant must not be silently compared with a differently fitted current
time constant. Preserve per-record observations and inspect source distributions
before introducing a fit objective.
