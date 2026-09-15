# Predict the human pair's test-pulse response

Approved parent: human-only qualification of all 104 cells, including electrical
load and synaptic response. The acquired human pair provides a -10 pA test pulse
and simultaneous postsynaptic voltage. This experiment tests an effective
electrical-response model before attempting a synaptic-current inference.

## Source and fixed split

Use only source ABF 359652, SHA256
a3f7af41e09b8885e0b8432ed79a6563c3e7b1c1d7735e25048409c5045640aa,
from the previous paired-response acquisition. The summary explicitly maps
postsynaptic IN2; the ADC header maps it to recorded channel index 1, physical
ADC 2. Reconstruct all enabled DACs 1, 2 and 3 separately from their epoch tables
and require identical commands for every sweep. Those outputs all command
0 -> -10 -> 0 pA at sample indices 153125 and 163125 (3062.5 and 3262.5 ms).
This waveform identity removes the choice between those DACs for this experiment;
it is not independent evidence of the amplifier wiring or delivered current.
Keep the distinction between reconstructed command and recorded voltage explicit.

Calibration: sweeps 0-14. Validation: sweeps 15-29. This is a prospective
within-recording split, not independent-donor validation or a blind holdout;
sweeps 0, 1 and 29 were previously viewed. Do not use validation residuals or
responses to change bounds, initialization or the observation window. Freeze the
candidate before evaluating validation responses. All 30 sweeps remain included.

## Model and observation

For each sweep, estimate one baseline voltage as the mean of all source samples
from 50 to 10 ms before pulse onset. Freeze that baseline before prediction.
Retain the unmodified original voltage as well as the separately centered data.
Use every original sample from pulse onset through +400 ms inclusive, covering
the 200-ms pulse and 200-ms return. Do not filter, resample, remove spontaneous
events, estimate offsets from the response, or replace negative results.

The mathematical candidate is a parallel RC response plus an instantaneous series
term. Its voltage change during the pulse is
I * (R * (1-exp(-t/tau)) + Rs) / 1000,
where I is pA, resistances are MOhm, tau and time are ms, and voltage is mV.
After the pulse it is I * R * (1-exp(-duration/tau)) *
exp(-(t-duration)/tau) / 1000. Before onset it is zero. These equations are
species-neutral circuit mathematics; parameters are fitted only to this human
recording. Rs is an effective instantaneous term, not a measured access resistance.
The source reports a 3-kHz postsynaptic telegraph filter and zero-valued capacitance
and access-resistance header fields. Do not interpret the zeros as measurements.
Acquisition filtering, bridge compensation, dendritic load and spontaneous input
limit the interpretation of fitted parameters as physical membrane quantities.

Fit one shared R, tau and Rs by equal-sweep ordinary least squares over all
calibration samples. Bounds: R [1,1000] MOhm, tau [0.05,200] ms, Rs [0,100] MOhm;
initial values 100 MOhm, 10 ms and 1 MOhm. Fit log R, log tau and linear Rs, one
start, at most 60 optimizer evaluations. Bounds are numerical hypotheses, not
human evidence. Closed-form vector evaluation has no simulated timestep loop.
Local CPU, 120-second wall cap; no new download or remote computation.

## Frozen decision

Reference prediction is each sweep's same frozen baseline with no pulse response.
Advance the candidate only for subsequent synaptic-response work if the optimizer
converges, R and tau are not at their bounds, validation combined RMSE improves
by at least 25% over that reference, and no individual validation sweep worsens
by more than 5%. Retain every per-sweep error and all original response samples.
Inspect absolute and centered model/voltage overlays for sweeps 0, 1, 14, 15, 16
and 29, plus residuals for all sweeps, with common scales. An accepted prediction
does not prove a unique circuit, membrane capacitance, a synaptic conductance,
donor accuracy, anatomy transfer or any full-population ledger term.

Test analytic step and return behavior, units, zero-current and long-time limits,
series discontinuity, malformed parameters and inputs. Independently verify
stored predictions and original sample membership. Do not widen failed bounds
or tune the candidate after validation. If rejected, keep the result and name
the observed contradiction before proposing another model.
