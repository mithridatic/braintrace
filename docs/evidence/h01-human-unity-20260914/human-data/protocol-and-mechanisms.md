# L1 protocol resolved; channel kinetics remain unqualified

The recording-method reference is now available. The new
[fitting-input receipt](fitting-inputs-r2.json) supersedes the earlier preparation
receipt without changing the calibration/holdout split or population scores.

The study's own methods specify a 32-34 C bath, 10 kHz Bessel filtering and
50 kHz sampling. Voltages were reported without correcting the measured -14 mV
liquid-junction potential. A subset of nucleated patches underwent voltage-clamp
measurements separating fast, slow and sustained potassium-current components.
These are study-level methods, not a temperature measurement or a potassium-current
recording from specimen 811953283.
[Primary paper, page 12](https://research.vu.nl/ws/files/288952685/science.adf0805_Morphoelectric_and_transcriptomic_divergence_of_the_layer_1_interneuron_repertoire_in_human_versus_mouse_neocortex.pdf).

The [protocol receipt](protocol-reference.json) records the access limitation:
the web search index supplied the primary methods text, while the direct PDF
download returned HTTP 403. There is no claimed local PDF hash. Existing voltage
arrays remain on their original reference. Applying a signed correction, retaining
the original arrays, and testing against double correction are subsequent work.
The protocol range must not become an invented exact specimen temperature.

## What the channel source actually provides

The pinned authors' notebook reads `L1_homologous_channel_cells.csv` from an
author-local path. Its saved outputs include human PAX6 observations: one slow,
two fast and four total conductance-density values counted across cells. These
counts describe saved author outputs; this audit did not recompute the fits or
execute the notebook. The table preview also contains mouse data. It is not a
complete, human-only kinetic source.
[Pinned notebook](https://github.com/AllenInstitute/patchseq_human_L1/blob/7777857c1b89b19735e45fd9c9bd7493cc1d805c/notebooks/l1_cross_species.ipynb).

The [retained extract](channel-notebook-extract.json) preserves relevant code and
text outputs, their cell indices and the source SHA256. The referenced CSV is
absent from the pinned Git tree. The archived v0.3 release was also downloaded:
41,874,969 bytes, matching its published MD5. All 78 archive members were examined;
its notebook references the same absent CSV. The [archive audit](channel-archive-audit.json)
retains the full member list, source hashes and exact archived reference.
[Author archive](https://zenodo.org/records/15313024).

This is a specific acquisition gap in the checked releases, not proof that the
measurements cannot be obtained elsewhere. Complete species/specimen-linked raw
current traces, patch geometry, voltage reference and protocol timing are still
needed to qualify kinetics. Saved conductance summaries cannot supply activation,
inactivation and recovery dynamics. Existing human pyramidal Na/K mechanisms also
do not constitute validation in this L1 PAX6 donor.

No model, optimizer or diagnostic dynamics ran. No held-out response arrays were
opened. Receipt hashes and unchanged splits were checked; executable code did not
change. Further source acquisition and the corrected anatomy transfer remain
actionable independently of the missing channel file. The six-term objective is
still open.
