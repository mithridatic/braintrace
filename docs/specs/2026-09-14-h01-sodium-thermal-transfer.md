# Sodium kinetic scaling tested against human AP-clamp responses

Approved parent: human-only qualification of all 104 cells and six original terms.
This experiment changes two kinetic scaling parameters and tests human responses;
it is not another source-acquisition exercise. Preserve the existing source model.

## Frozen data and model

Use only the sealed Hwide command and 16 human response sequences from e96973a0.
Train on filename group H21.29.194 (7 records). Validate separately on H21.29.195
(5) and H21.29.197 (4). All source trajectories were previously inspected; this
is a prespecified group split, not a blind holdout or confirmed independent-donor
split. Do not inspect validation fit errors until the candidate is frozen.

Freeze the published sodium equilibrium curves and 25 C kinetic equations in
braintrace/datasets/h01_wilbers.py. Use the sodium command with its one -10 mV
shift already applied; do not add the whole-cell wrapper's shift again. The
AP-clamp reversal is 141 mV as in the source AP-clamp helper. At 34 C the source
baseline scales both activation and inactivation rates by 2.3**0.9. The candidate
changes only two positive rate multipliers at 34 C, one per gate, with fixed
bounds [0.25,8]. These bounds are numerical hypotheses, not biological evidence.
The 25 C kinetics remain unchanged. Fit log multipliers with bounded least squares,
one start at the baseline, maximum 60 function evaluations. No additional starts,
bound expansions, gains, voltage shifts or current-specific fits after viewing
validation outcomes. A constant conductance factor cancels in normalization.

## Numerical and observation path

Use the complete original baseline and AP1-200 command window, 0-5182 ms, at the
original 0.008-ms sample clock. Command is held constant between source samples,
as in the source Vector.play usage without continuous interpolation. Initialize
gates at the equilibrium of the first command sample, and retain that assumption.
Apply exact exponential gate updates per held-voltage interval, using
brainstate.transform.scan under jit with an explicit gate carry. No Python loop
drives repeated model steps. The inward current proxy is m**3*h*(141-V), with
both one-sided values retained at command transitions for peak extraction.

Take one peak inward current in each fixed source-AP window from 1 ms before
to 2 ms after the source voltage peak. Normalize each sequence by AP1, as in the
source sodium comparison; do not normalize by the largest current. Require all
200 source AP windows, positive finite peak currents, finite gates in [0,1], and
complete time coverage. Preserve direct voltage/current-proxy/gate samples and
source indices around AP1, AP5 and AP200, and all 200 amplitude predictions.
This ideal-command proxy is not measured ionic current or verified experimental
clamp tracking, and its absolute magnitude has no fitted conductance density.

Fit equal recording-weight squared residuals over the 200 normalized amplitudes
in the calibration group. Report every candidate evaluation and terminal status.
Use local CPU only, with a 600-second wall cap for the complete experiment.
If the initial compiled run is too slow to fit within the cap, stop with the
terminal evidence rather than silently shortening the 200-AP sequence.

## Candidate decision

Freeze the fitted parameters before scoring the other groups. Accept the
candidate for further mechanism qualification only if the optimizer converges,
both validation groups improve, their combined equally recording-weighted RMSE
improves by at least 5%, and no validation record worsens by more than 5%.
The numerical check repeats baseline and frozen candidate with two substeps per
source interval (0.004 ms), preserving the same held command and observation
windows. Require maximum difference in the 200 amplitude ratios <=0.001.
Do not widen that threshold after inspecting results.

Retain a rejected candidate, all validation errors and direct plots. If the gate
fails, close this two-scalar temperature-scaling branch rather than retrying it
with more fitting freedom. Even success qualifies only a comparative prediction
under the registered observation model; it does not establish a uniquely measured
Q10, resolve missing original currents or automatically promote any six-term score.
Test constant-command gate evolution against its analytic solution, source-grid
endpoint handling, normalized peak extraction, and invalid/incomplete inputs.

## Equivalent batching after the first runtime stop

The first implementation was terminated after 126.7 seconds without a baseline
result; its source and terminal receipts are retained under sodium-thermal-transfer.
No calibration or validation score was obtained. Re-express each exact interval
update as an affine map g_next=a*g+b. Compose prefixes within fixed blocks of
1024 intervals using associative affine algebra, and carry the block-end gates
with brainstate.transform.scan. Identity maps pad only the final block and their
outputs are discarded. This retains every original gate update and observation;
it changes execution batching, not physiology, scoring or the data split. Check
it against a sequential compiled oracle across multiple block boundaries and the
analytic constant-command solution before rerunning under a new evidence prefix.
