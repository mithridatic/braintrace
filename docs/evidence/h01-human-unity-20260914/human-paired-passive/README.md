# Human test-pulse model predicts the later sweeps

The frozen electrical-response candidate passes its registered gate for further
synaptic-response work. Validation RMSE falls from 0.666883 to 0.309740 mV, a
53.55% reduction, and all 15 validation sweeps improve over the baseline-only
reference. This is a within-recording prediction of one human cell's -10 pA
test pulse, not a full donor model or a population score improvement.

## Source, command and split

The original human pair, its source SHA256 and electrode identities are retained
in the [acquisition checkpoint](../human-paired-synapse/README.md). Physical ADC 2
(IN 2) is recorded at array index 1. The ABF has enabled DAC outputs 0-3; pyabf's
ordinary channel-associated command for array index 1 is DAC 1, not DAC 2.
Here the independently reconstructed DAC 1, 2 and 3 waveforms are identical in
all 30 sweeps: zero, -10 pA over source indices [153125,163125), then zero.
The waveform identity resolves the choice between those DACs for this test.
It does not independently verify amplifier wiring or measure delivered current.

The [specification](../../../specs/2026-09-14-h01-human-paired-passive.md) fixes
sweeps 0-14 for calibration and 15-29 for validation. Sweeps 0, 1 and 29 were
viewed earlier; this is not blind or independent-donor validation. The candidate
was [frozen](frozen-candidate.json) before the runner extracted validation
responses. No parameters changed after scoring those responses.

Each sweep's baseline is the mean of its 2,000 original samples at indices
[150625,152625), from -50 ms to just before -10 ms relative to pulse onset.
Every original sample at indices 153125 through 173125 inclusive enters the
pulse/return comparison: 20,001 samples per sweep, 300,015 per split. Original
absolute within-sweep clocks, voltages and baseline samples remain in
[observations.npz](observations.npz). Model time is relative integer sample index
times 0.02 ms, so command endpoints are exact source-grid endpoints; no voltage
sample is interpolated. No spontaneous event or unsuccessful sweep is removed.

## Model and result

One shared circuit uses a parallel RC response plus an instantaneous series term.
The [fit](fit.json) has 8 optimizer evaluations and 32 calls including numerical
derivatives, converges normally, and reaches no parameter bound.

| Effective parameter | Fitted value |
| --- | ---: |
| R | 72.5590 MOhm |
| tau | 13.0774 ms |
| Instantaneous series term Rs | 20.9702 MOhm |

These are effective response parameters conditional on the reconstructed command,
baseline operator and circuit. Rs is not a measured access resistance. The header
reports a 3-kHz postsynaptic filter and zero-valued access-resistance/capacitance
fields; the zeros are not interpreted as physical measurements. Filtering,
bridge compensation and distributed membrane response can affect the fitted
instantaneous term. Neither a specific membrane capacitance nor a synaptic
conductance is qualified by this fit.

The reference predicts each sweep's frozen pre-pulse baseline throughout. It is
a no-pulse-response reference, not an alternative fitted circuit. All per-sweep
errors are retained in [result.json](result.json). Validation sweep 23 improves
least: RMSE 0.5155 to 0.4850 mV. Sweep 15 improves from 0.6602 to 0.0836 mV.
Passing the comparative gate demonstrates useful prediction; it does not claim
that every sample or every sweep has a small absolute error.

## Direct visual review

The [absolute responses](absolute-responses.png), [separately centered responses](centered-responses.png)
and [all 30 residual trajectories](all-residuals.png) were opened and inspected.
The candidate follows the main negative relaxation and return in the selected
calibration and validation sweeps. Return is faster in several observed sweeps
than in the shared candidate. The instantaneous model term also does not reproduce
the acquisition transition sample by sample. Calibration sweep 14 has a more
negative pulse plateau than the prediction.

Positive deflections remain during and after the test pulses. Validation sweep 29
has a large deflection near +60 ms; sweep 20 has a large late-return deflection,
visible in the residual view. Calibration sweep 10 has a sustained negative
residual through most of the window, consistent with an imperfect baseline or
other unmodeled input/state; this plot alone does not identify its cause.
No mean trace replaces these discrepancies. The common residual color scale
retains all extrema. [Sixty-six direct phase samples](direct-samples.json) bind
the displayed sweeps to their original source indices and absolute voltages.

## Verification and consequence

The complete process finished in 2.430 s under its 120-second wall cap, local
CPU only ([terminal receipt](terminal.json)). The affected paired-response suite
passes 34 tests; the new circuit helper has 17 tests and 100% line coverage.
They cover independent analytic pulse/return values, current/resistance units,
series discontinuity, zero current, sign reversal, long-time return, invalid
clocks/parameters and synthetic recovery of a known circuit.

Independent checks bind all 30 source windows and pre-pulse baselines to the
original ABF data matrix, reconstruct the prediction from the frozen equations,
and recompute the held-out decision. This candidate may now support a registered
synaptic-response comparison with the cell's electrical response constrained.
Such a comparison must account for spontaneous activity and distinguish injected
current through the electrode from synaptic current entering the membrane; Rs
must not automatically be applied to a synaptic current. Human-only donor
qualification, H01 geometry transfer and full-population controls remain required.
No six-term score changes.
