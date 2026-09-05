# Source model for the layer-2 reference

Allen model 626170538 is a perisomatic biophysical fit attached to
specimen 541563728. The source lists sweep 51 as its fitting sweep.
Our selected sweep 50 is not in that list. This fact does not prove
that sweep 50 was untouched during all source model development.

The [published fit](https://api.brain-map.org/api/v2/well_known_file_download/626185228)
sets initial and leak voltage to -83.979935 mV, temperature to 34 C,
and junction potential to -14 mV. Its baseline is close to the corrected
sweep-50 baseline of -84.232297 mV. Closeness of baseline does not
establish agreement of the active response.

Ten channel mechanisms and calcium dynamics act in the soma. The
remaining regions have passive parameters in this fit. Channel densities,
calcium parameters, and passive parameters are fitted values. They are
not separate direct measurements from this cell. These mechanisms are
not the Wilbers channel measurements and must retain separate provenance.

The source morphology and all eleven required mechanism files are cached
under `.cache/human-pyramidal-l2/source-model`. The
[source manifest](h01-pyramidal-allen-l2-model-source.json) records URLs
and local SHA-256 values. No independent source digest was supplied.
The mechanisms have not yet been compiled or simulated here.

The inspected [AllenSDK helper](https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/model/biophysical/utils.py)
replaces the reconstructed axon with two 30 um sections of diameter
1 um. It sets section segment counts from section length. This replacement
is a model assumption, not measured anatomy. The helper reads the command
waveform; it does not separately add the recorded bias in `read_stimulus`.
Source reproduction must preserve this distinction from a command-plus-bias
intervention. Do not silently add bias to the source fit or treat its omission
as proof that the experiment had no bias.

The downloaded helper is an inspected current reference, not a verified
historical execution environment. Its local SHA-256 is
`7d35ca03cbfd0fae79d4e9e8c3ee7b832377a20120ece5a00ad9b020122f213f`.
Next work must reproduce the source setup, record solver and morphology
choices, and compare direct events before any transfer onto H01 anatomy.
