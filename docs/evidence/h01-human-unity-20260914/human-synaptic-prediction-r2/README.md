# Human synaptic prediction passes the sweep-level gate

A shared synaptic-current candidate, filtered by the previously frozen human
cell response, reduces validation RMSE from 0.462780 to 0.301178 mV (34.92%).
All 15 validation sweeps improve, and no fitted parameter reaches its bound.
It passes the registered gate for further qualification. However, 16 of 90
individual validation events worsen, including 14 by more than 5%. It is not a
trial-by-trial physiological reproduction, and no six-term population score changes.

## Model, source and split

The [specification](../../../specs/2026-09-14-h01-human-synaptic-prediction.md)
fixes the model, bounds, split and decision. The original human ABF source and
electrode mapping remain those of the [paired acquisition](../human-paired-synapse/README.md).
The electrical candidate's R=72.5590 MOhm and tau=13.0774 ms remain fixed; the
electrode series term is excluded from the synaptic-current path.

The [frozen synaptic candidate](frozen-candidate.json) was written before the
runner extracted validation EPSPs. Sweeps 0-14 supply calibration and 15-29
validation. This is a within-recording comparison, not independent-donor or
blind validation. All 30 sweeps and 180 commanded events remain included.

| Inferred effective-current parameter | Fitted value |
| --- | ---: |
| Peak current | 130.0317 pA |
| Rise time | 0.150432 ms |
| Decay time (rise plus fitted gap) | 1.175869 ms |
| Delay after recorded presynaptic crossing | 1.928160 ms |

These parameters describe a mathematical input under the frozen cell and
observation models. They are not recorded ionic currents, isolated receptor
kinetics or a voltage-dependent conductance. In particular, the postsynaptic
3-kHz acquisition filter and the effective cell response limit interpretation
of the fast fitted rise. No molecular or reversal-potential claim is made.

The candidate sums contributions from all six actual presynaptic spike times,
including earlier-event tails. Both human and candidate voltages have their own
pre-event mean subtracted over [-10,-2) ms, using all 400 source samples. Scoring
uses every sample from 0 through 50 ms inclusive: 2,501 per event, 225,090 per
split. Original -10 through 50 ms pre/post voltages, clocks and indices are
preserved in [observations.npz](observations.npz), along with inferred currents
and predictions. Spontaneous deflections remain in the residuals; they are not
silently attributed to evoked input or excluded.

## Integrity correction before optimization

The [initial run](../human-synaptic-prediction/README.md) stopped before fitting.
Sweep 2's fifth spike has a tiny falling-phase recrossing at index 68280, whose
local maximum is 0.305176 mV. Its adjacent samples and full waveform remain in
the original observation bundle. The corrected extraction associates exactly
one recorded upward crossing with each positive command pulse. It reports the
extra crossing explicitly; all its original voltage samples remain inside the
fifth event window. There are no other unassigned crossings in the 30 sweeps.

The pyabf 2.3.8 Pulse reconstruction floors epoch duration divided by period.
For the source 40,150-sample epoch, 10,000-sample period and 150-sample width,
that yields four pulses even though a fifth complete pulse fits at the end.
A regression test reproduced four starts where five were required. The local
adapter now evaluates the pulse phase at every source index, supports only
Step/Pulse epochs and rejects unsupported forms. The installed dependency is
unchanged. Corrected positive starts are 28125, 38125, 48125, 58125, 68125 and
93275. The last entry is the separate recovery pulse. Both old and corrected
commands are retained.

The [corrected command view](corrected-command.png) was opened and inspected:
the restored fifth command aligns with the observed fifth spike, while the
earlier parser output incorrectly stays at zero. This is a source-reconstruction
correction, not an independent measurement of delivered current. The passive
candidate's negative Step epochs are unaffected.

The mistake was accepting a parser-derived command and an undifferentiated
zero-crossing count without checking them together against the source epochs
and waveform. Future ABF work should verify pulse cardinality, source timing,
event association and unsupported epoch types before fitting. The original
failure remains preserved; the split, model bounds and rejection criteria were
not changed after an optimization or validation result.

## Direct visual review and limits

[Calibration events](calibration-events.png), [validation events](validation-events.png),
[all 180 residuals](all-event-residuals.png), and the [inferred current/voltage pair](inferred-current-and-voltage.png)
were opened and inspected. The candidate follows the common EPSP rise and decay,
but one shared strength misses large event-to-event differences. Calibration
sweep 1/event 0 has a pre-existing deflection and subsequent decline through the
baseline, which the evoked candidate cannot reproduce. Sweep 1/event 1 and
sweep 14/event 5 are substantially larger than the candidate. In validation,
sweep 15/events 0 and 5 have strong baseline/background discrepancies, while
sweep 16/event 0 is larger and event 5 contains later additional deflections.

These mismatches remain visible on common scales and in the complete residual
arrays. [108 phase observations](direct-samples.json) retain original pre/post
voltages and indices alongside separately labelled inferred currents. They do
not establish a unique release mechanism or identify a specific missing channel.
Every validation event ordinal improves in aggregate; that does not override
the 16 individual-event regressions. The [result](result.json) retains every event
and sweep error. The weakest sweep-level improvement is 9.26% in sweep 15.

## Verification and next boundary

The corrected process finished in 12.924 s within the unchanged 120-second cap,
using local CPU closed-form convolution. The optimizer converged after 16
reported evaluations and 72 calls including numerical derivatives. No remote
run or new source acquisition occurred. The affected suite passes 60 tests;
the synaptic helper has 100% line coverage and the ABF adapter 93%.

Tests cover convolution against independent quadrature, coincident time constants,
current normalization, causality, superposition, preceding-event tails, baseline
subtraction, malformed inputs, final pulse-period boundaries and ambiguous
command-associated spikes. Independent source/equation checks verify the retained
observations, inferred outputs and decision. The next qualification boundary is
prediction beyond this same recording and comparison with the deployed synaptic
path. This current-based candidate cannot be silently substituted for a
conductance-based synapse or declared valid across other voltages, donors, types
or H01 geometries. The all-104 objective remains unmet.
