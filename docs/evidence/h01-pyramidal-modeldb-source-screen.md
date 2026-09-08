# Pyramidal optimizer source screen

The pinned ModelDB 267587 tree contains a young pyramidal optimizer under
`Single Cell Modelling/Optimizations/L5Pyr_young_NSG`. Its `init_active.py`
was downloaded and checked against the pinned Git blob hash before inspection.

The script loads a separate `active.pkl` file but calls
`get_active_features(data_active_dict, 1)`. This selects manually supplied
feature targets. For example, the low-step spike-count target is
3.369565217391304, which cannot be a count from one direct spike train.
The script supplies a separate dispersion value and targets waveform and
interval features. These targets do not establish a matched individual
current-input/voltage-output response.

No trace file was loaded or executed during this screen. Its provenance
remains unresolved. A model's morphology filename also does not establish
that its fitting traces were recorded from that reconstructed cell.
Do not replace the missing direct excitatory reference with these feature
targets or transfer an L5 fit to H01 L2 anatomy without stating the mismatch.

Source: [pinned optimizer](https://github.com/ModelDBRepository/267587/blob/82cdd91bc93942ba19315371330a2412e064baf5/Single%20Cell%20Modelling/Optimizations/L5Pyr_young_NSG/init_active.py),
especially the manual branch and its call. The source script is cached at
`.cache/human-pyramidal-modeldb/init_active.py`.
