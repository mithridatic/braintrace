# Bounded family search for the layer-2 excitatory candidate

## Behavior

Under the recorded 43 pA input the layer-2 candidate shows excess
depolarization and the wrong post-pulse return. Under the 250 pA input it
fires the recorded five spikes with every interval too long, recovery minima
too negative, rises too short, and returns too long. Eight single-parameter
levers are eliminated and eight conditional effects are recorded. No split
says which parameter family carries the dominant variable.

## Gate before launch

The [acceptance table](../evidence/h01-l2-acceptance-table.md) fixes the
conventions, datums, candidate residuals, and measured numerical decision
limits for every required observation. No physiological allowance is agreed,
so this campaign issues no pass and no fail. Its output is a family ranking
and a search tree. "Steep X not separated" stays on the causal page until the
campaign records a family whose swap reverses the source-to-candidate change.

## Design: Search Dissection over families

Frozen controls: source model S and current candidate C
(`h01-l2-kv3-ninety-ca133`). C differs from S in five families:

| Family | Members in C |
| --- | --- |
| F1 sodium gate law | NaTs opening factor 2.0 (recovery 1.0) |
| F2 Kv3 gate law | Kv3 closing factor 0.9 |
| F3 calcium handling | calcium decay factor 1.33 |
| F4 regional densities | somatic NaTs density factor 1.3 |
| F5 passive and Ih | distributed Ih, factor 75; leak reversal shift -4 mV |

Every candidate runs both calibration inputs: sweep 50 with recorded bias and
sweep 43. Sweep 53 stays closed. Cap: 24 new full-cell evaluations, at most
two concurrent simulations.

- Stage 0, coarse preservation (6 evaluations). S, C, and one far corner (S
  with F1 and F4 at C values), each at nseg 3 and at nseg 9, CVode 1e-10.
  No existing S or C run has both sweeps with the recorded bias at one
  setting, so all six are new; the C fine sweep 50 repeats the frozen run
  and doubles as a repeatability check of the container build. Preserved means: every residual sign and
  every pairwise ordering among S, C, and the corner is unchanged between
  settings, and each coarse-to-fine change is below one fifth of the smallest
  contrast it must rank. If not preserved, Stages A and B run at fine settings
  and Stage B is dropped to stay under the cap.
- Stage A, family dissection (10 evaluations, coarse). Each family alone
  swapped into S (5) and each family alone reverted in C (5). Rank families by
  the root sum of squares of residual changes normalized by the measured
  numerical decision limit for each observation. Raw residuals are retained;
  normalized values guide only. If no single family reverses the S-to-C change
  on the ranked observations, swap the two largest together (spare budget).
- Stage B, within the dominant family (at most 6 evaluations, coarse).
  Half-splits of that family's members, chosen after Stage A.
- Stage C, finalists (at most 2 evaluations, nseg 9, CVode 1e-10). Recheck the
  two best Stage B candidates at fine settings.

Total at most 24: 6 + 10 + 6 + 2. The interdependency swap, if needed, takes one Stage B slot.

## Predictions stated before running

- Sparsity: one family's swap moves the ranked observations by more than the
  others combined in quadrature. Its name is the family-level Steep X.
- If F1 or F2 dominates, the rising and falling phase residuals move first; if
  F3 dominates, later intervals move while first onset and the first minimum
  stay within their numerical limits; if F4 dominates, peaks and first onset
  move; if F5 dominates, subthreshold samples and baseline move.
- If no family reverses the change alone and the paired swap does, the causal
  page records an interdependency, not a Steep X.

## Measured cost

From recorded `integration_seconds` at stop 2100 ms: coarse (nseg 3, 1e-10)
sweep 50 200-211 s and sweep 43 37-131 s, at most 5.7 min per candidate; fine
(nseg 9, 1e-10) sweep 50 592-672 s and sweep 43 361-434 s, about 17 min per
candidate. Serial container time: Stage 0 about 68 min, A about 57 min,
B about 34 min, C about 34 min; about 3.2 h total, no single run over 12 min.
Two-concurrent speedup is unmeasured; the runner records wall time per
candidate and the first timed pair sets the figure.

## Code

- `docs/evidence/h01_l2_regional_density.py`: scale one mechanism density in
  named regions (`soma`, `axon`, `dend`, `apic`, or `all`).
- `docs/evidence/h01_l2_neuron_reference.py`: `--candidate-json` (flag values
  from a file, recorded with its hash; explicit CLI flags still override) and
  repeatable `--regional-density MECHANISM:REGION:FACTOR`.
- `docs/evidence/h01_l2_campaign.py`: manifest-driven runner. Each candidate
  runs two `docker run braintrace-h01-neuron:9.0.2` calls with at most two
  concurrent, from `.cache/human-pyramidal-l2/kv3-closing-source`, into
  `docs/evidence/h01-l2-campaign/`. It skips finished candidates whose flag
  hash matches, enforces the cap, refuses Stage A without a recorded
  `preserved: true` Stage 0 decision, and records wall time.
- `docs/evidence/h01_l2_campaign_score.py`: per-candidate raw residual vectors
  (events, phases, intervals, recovery minima on the human convention,
  subthreshold samples, missing and extra events), family contrasts, RSS
  ranking, and the mermaid family tree.
- Manifest `docs/evidence/h01-l2-campaign-manifest.json`.

## Decision rule

The family ranking uses only observations whose contrast between S and C
exceeds five times its numerical decision limit. A family is the Steep X when
its RSS exceeds the RSS of all other families combined in quadrature. Any
finalist claim is conditional on the fine-setting recheck. No claim about the
human cell's channel densities follows from a family ranking.
