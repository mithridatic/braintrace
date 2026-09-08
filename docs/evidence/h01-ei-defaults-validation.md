# E/I default integration evidence

The default E profile is h01-l2-kv3-ninety-ca133. The default I profile is
h01-pv-regional-mesh-axon2187. Both profiles retain source hashes and expose
source mode. The selected parameters are shipped in the Python package.
The runtime does not read the evidence directory or private cache for parameters.

The channel audit reads exposed rates from pinned NEURON mechanisms. It covers
720 cases across source and candidate modes, voltage, and gate direction.
The PV source fit overrides must be applied before reading rates. NEURON must
refresh mechanism voltage before an exposed rate procedure is called. These
export requirements were checked after initial audit failures. The exact PV
poles have a declared 0.0001 mV source perturbation exception; other cases use
2e-7 absolute and relative rate tolerances. A separate equilibrium test checks
the unchanged E Kv3 time constant when the gate derivative is zero.

The donor geometry check uses independent saved NEURON response arrays. Over
0.001 to 2 ms, maximum absolute soma-voltage errors were:

| Profile | Error (mV) | Limit (mV) |
| --- | ---: | ---: |
| E candidate | 0.0001376435 | 0.1 |
| E source | 0.0000709211 | 0.1 |
| I candidate | 0.0001659392 | 0.1 |
| I source | 0.0005396665 | 0.1 |

This validates early responses on donor geometry. It does not validate spike
transfer over a full stimulus sweep. Source and candidate mismatches with
human recordings remain independent physiological failures.

The H01 example ran two measured components with explicit inferred electrical
partitions. At 0.005 ms steps, both 10 ms responses were finite. The 0.1 nA pulse
ran from 2 to 5 ms. Neither cell crossed 0 mV. Direct soma voltages were:

| Role | H01 cell | At 2 ms | At 5 ms | At 10 ms |
| --- | --- | ---: | ---: | ---: |
| E | 810151953 | -83.99501218 | -79.14219211 | -81.22315642 |
| I | 678539249 | -81.76229155 | -77.03939189 | -82.92762743 |

Voltages are in mV. The NPZ file retains all 2,000 samples per cell. The JSON
file retains the source identity, profile, and electrical region intervals.
The example establishes a subthreshold execution demonstration. It does not
establish a connected circuit, inhibitory suppression, or H01 spike stability.

The source tag for the I cell is interneuron. PV identity is not established.
The electrical map assigns unclassified remaining cable to basal-dendrite
parameters and does not model myelin insulation. These are recorded assumptions.

Tests cover frozen metadata, physical densities, regional changes, source mode,
units, current ownership, gate singularities, phase boundaries, finite compiled
responses, and rejection of missing, overlapping, or incomplete region maps.
Future circuit tests must hold starting state and input fixed, change one
synaptic intervention, and retain local voltage and individual spike events.
Human measurement targets must not be relaxed to make the selected candidates pass.

Final focused suite: 56 passed with UserWarning treated as an error. Coverage
of the seven new or changed production modules is 253 of 254 statements
(99.6%; coverage.py displays 99%). Every measured module exceeds 90%.
These results cover this integration change, not the entire repository.

```powershell
.cache/validation/Scripts/python.exe -m coverage run -m pytest braintrace/datasets/h01_ei_profiles_test.py braintrace/datasets/h01_ei_cell_test.py braintrace/datasets/h01_l2_channels_test.py braintrace/datasets/h01_l2_cell_test.py braintrace/datasets/h01_pv_cell_test.py braintrace/datasets/h01_pv_channels_test.py braintrace/datasets/h01_pv_calcium_test.py -q -W error::UserWarning
```
