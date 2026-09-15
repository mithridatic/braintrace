# Joint human voltage-dependent current candidate

Implements the approved human-only donor qualification plan. The previous
initial-state-only recovery repair remains rejected. This is a new mathematical
candidate spanning voltage protocols, not a molecular channel assignment.

## Candidate and observation equation

Use two availability populations and one sustained population sharing activation
m, with current (V-E)*(g1*m^2*h1 + g2*m^2*h2 + gs*m^2). Activation asymptote is
logistic((V-vm)/km), with constant tm. Both availability asymptotes are
logistic((vh-V)/kh); population j has tau dj+(rj-dj)*h_inf(V). Every state is
carried across the entire recorded command history; initial gates equal the
first holding-command asymptotes. E and the three conductances are fitted in
the original uncorrected command reference. Neither E nor g is reported as a
measured ionic reversal or membrane conductance.

Add leak gL*(V-Vhold) and two observation transients Cj/tcj*(V-zj), with
dzj/dt=(V-zj)/tcj. These are conditional recording-response components, not
extra membrane channels. Subtract initial active holding current once, and pass
the entire sum through one first-order observation filter with time constant tf.
Initial transient/filter deviations are zero. This permits continuous measured
current at command transitions. It does not simulate clamp feedback or establish
the actual patch voltage. The later whole-cell model must distinguish membrane
and observation components rather than deploy amplifier artifacts as channels.

Implement exact constant-command solutions. Gate-state propagation and filtered
sums of exponentials use vectorized affine composition across command segments;
no Python timestep/model loop. Include the equal filter/forcing-rate limit and
underflow-safe long holds. All values and signed current predictions must remain
finite. Preserve conditional component outputs and original command history.

Parameter order and initial values:
g1=1.5, g2=2, gs=1 nS; vm=-20 mV, km=15 mV, tm=3 ms;
vh=-50 mV, kh=10 mV; d1=25, d2=700, r1=80, r2=500 ms;
E=-85 mV; gL=0.3 nS; C1=2 pF, tc1=0.2 ms; C2=10 pF, tc2=20 ms;
tf=0.08 ms. Bounds: g1/g2/gs 0-50; vm -60 to 20; km/kh 2-35;
tm 0.04-100; vh -100 to 0; d1 1-300; d2 50-20000;
r1 1-3000; r2 10-20000; E -120 to -50; gL 0-3;
C1/C2 0-100; tc1 0.04-10; tc2 1-500; tf 0.04-3.
These are registered candidate bounds, not independently measured human ranges.

## Frozen training and prediction roles

Use only source human PAX6 specimen 840043506, session 840043481, SHA256
30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944.
Fit total-current sweeps 70,72,74,76,78; controls 79,83,88; tails 99,101,103.
Every response has already been exposed. Retain all source samples; center on
the pre-control 35-44 ms mean. Fit every original sample for the first 10 ms
after each command transition and every 25th original sample elsewhere,
starting at source index zero. Exclude only times before 35 ms from fitting.
Weight selected residuals by the square root of the number of following source
intervals they represent. This is a registered time-quadrature fitting objective,
not equal weighting of the retained subset and not full-sample least squares.
Score the final model on EVERY original sample from 35 ms onward. Preserve
the selection and weights so the approximation is explicit and reproducible.
Direct views of these calibration protocols were inspected before choosing this
sampling; no unseen response determines selection, weights or bounds.

Before opening their responses, freeze the fitted candidate and predict sustained
protocol sweeps 105,106,107,109,110,111 using only their commands/metadata. Exclude
previously viewed sustained sweeps 104,108,112 from this prediction set. No
reserved external donor or whole-cell response is opened. This is same-specimen
protocol prediction, not external validation. If fitting fails, preserve it and
do not open these six new responses.

Candidate must converge within 80 least-squares evaluations and one 600-second
local CPU process cap. Use complex-step Jacobians and one registered initial
point; do not widen bounds or restart unchanged after timeout. Retain fitted
parameters, active bounds, terminal status, predictions, residuals and full-sample
per-sweep errors. On the six new protocols, compare the frozen complete candidate
with its own frozen leak/transient/filter-only prediction. Require at least 25%
pooled full-sample RMSE improvement, no individual sweep over 5% worse, finite
complete output, and no kinetic parameter at a bound. Failure retains all cases
and rejects deployment. A pass permits further kinetic qualification only; it
does not change the six-term ledger or establish a human donor model.

## Verification

Co-located tests must cover a separate analytic leak/filter oracle, filtered
capacitive step, quadrature of active current at one held voltage, equal rates,
long holds, segment subdivision invariance, batched versus individual protocols,
initial equilibrium, complex-step derivative agreement, malformed clocks,
nonfinite and inadmissible parameters, missing coverage and invalid query IDs.
Target greater than 90% helper coverage. Preserve all direct raw-current and
prediction views; inspect first/middle/last protocol traces and every residual
before drawing a model-development conclusion. Read the existing direct-observation
contract and causal model. No neuronal or population rollout is authorized by
this bounded calibration; no animal kinetic parameters enter this candidate.
