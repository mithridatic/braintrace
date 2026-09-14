# Human donor data access established

Current analysis: [PAX6 currents and control assessment](pax6-840043481/README.md)
prepares the type-matched source, tests linear-control sensitivity and records
why the first descriptive decay fits are unsuitable for kinetic deployment.

Latest follow-up: [human L1 channel-current sources recovered](channel-923103553/README.md)
provides verified LAMP5 and type-matched PAX6 recordings, direct current observations
and a distinct-donor kinetic validation reservation.

Follow-up: [recording preparation and QC](preparation.md) now establishes eight
calibration and nine held-out sweeps passing standard IPFX QC, with explicit
padding masks and exclusions. This acquisition receipt remains the original
data-access record; it does not qualify a donor model.

The missing fitted L1 donor has an actionable data source. On 2026-09-14, the
published DANDI 000630 version 0.230915.2257 was queried directly: 210 human cells,
open access, CC-BY-4.0 electrophysiology. The source collection and analysis
metadata were known leads in the September 7 donor survey. This follow-up verifies
actual access, specimen correspondence and one complete recording/morphology pair.

The authors' metadata at AllenInstitute/patchseq_human_L1 commit
7777857c1b89b19735e45fd9c9bd7493cc1d805c contains 45 rows meeting all of these
source conditions: human; target layer 1; electrophysiology and morphology present;
exclude=No; transcriptomic QC true. Of those, 36 match an electrophysiology session
in the pinned DANDI version, and 17 are MTG. These counts describe acquisition
candidates. They do not constitute new validated donors or H01 type matches.

The first asset inspected, specimen 1010240295, was marked exclude=YES by the
authors. It is retained only in cache as an access probe and is excluded from
fitting selection. Selection thereafter used the stated source conditions,
then MTG, smallest recording size, and specimen ID as a tie-break. No response
score or spike count selected the candidate.

## Selected acquisition

- Specimen 811953283, recording session 811953264, human MTG L1 aspiny neuron,
  transcriptomic type PAX6 CDH12; author exclude=No.
- DANDI asset 23666b66-2d4d-4b4e-a5f0-bbc39e898252: 19,022,226 bytes. The downloaded
  SHA256 equals the repository's declared SHA256.
- Matching Brain Image Library morphology: human/811953283/811953283_raw.swc,
  419,447 bytes. A local SHA256 is retained; no upstream SWC digest was available
  in the directory listing, so the SWC hash is not independently authenticated.
- HDF5 inspection with h5py 3.16.0 found 37 acquisitions: 31 CurrentClampSeries
  and six VoltageClampSeries. Every response and paired stimulus array is finite,
  with matching sample count, start time and sample rate. The voltage-clamp series
  include seal/access checks; their presence is not human channel-kinetics data.
- Current-clamp protocols include subthreshold, long-pulse search/rheobase,
  suprathreshold, short-pulse and ramp stimuli. Individual sweep metadata and
  conversion factors are in [source-access.json](source-access.json).

The raw NWB and SWC are retained in `.cache/human-source-followup` on this
worktree. The receipt records public download URLs, exact dataset versions,
specimen/session identities, hashes, selection conditions and every sweep's
timing and units. No model run or fit was performed, and no score changed.

## What can proceed independently

The next work is per-sweep QC and protocol correction, an independent fitting and
holdout contract, morphology interpretation, and mechanisms supported by human
measurements appropriate to this class. A fitted donor must pass its own human
recording tests before transfer onto audited H01 anatomy. The existing Wilbers
implementation is a source to assess, not automatically a qualified L1 model.

Missing paired recordings for the specific H01 cells limits claims about their
actual physiology. Human-donor fitting and transfer validation can still proceed
under the approved six-term contract. An unrecorded experimental measurement
cannot be generated retrospectively by simulation. No collection of new biological
tissue or contact with external researchers was undertaken.

## Sources

- [Published human L1 electrophysiology](https://dandiarchive.org/dandiset/000630/0.230915.2257)
- [Pinned author metadata](https://github.com/AllenInstitute/patchseq_human_L1/blob/7777857c1b89b19735e45fd9c9bd7493cc1d805c/data/human_l1_dataset_2023_02_06.csv)
- [Morphology collection and protocols](https://doi.brainimagelibrary.org/doi/10.35077/g.606)
- [Selected specimen morphology files](https://download.brainimagelibrary.org/group/20230127/U01Lein/human/811953283/)

DANDI attribution: Chartrand, Lee, Dalley, Lein and Kalmbach (2023), Human L1
patch-seq electrophysiology, doi:10.48324/dandi.000630/0.230915.2257.

## Recovery constraint checkpoint

[All 19 PAX6 paired-pulse conditions](pax6-recovery-840043481/README.md) now
constrain a conditional recovery envelope. Structured residuals and baseline/history
questions remain; external validation currents stay unopened. The current input
receipt is [revision 5](fitting-inputs-r5.json). No six-term score was promoted.

## Conditioning and history checkpoint

[All nine matched test-voltage pairs](pax6-conditioning-840043481/README.md)
correct the earlier sweep-89 command interpretation and expose different responses
to the same actual command across protocol families. Conditional decay fits are
not qualified for deployment. [Revision 6](fitting-inputs-r6.json) is the current
input receipt; the original whole-cell split and all six scores remain unchanged.
