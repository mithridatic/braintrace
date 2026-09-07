# I cell, SP4 reserve evaluation: result

Spec [2026-09-07-h01-i-reserve.md](../specs/2026-09-07-h01-i-reserve.md), manifest
`h01-i-reserve-manifest.json`, decision `h01-i-reserve/stage-1-decision.json` (this page
is derived from it). Candidate `g-natg-axon15` (finalist flags plus `--scale NaTg:axon:1.5`,
somatic NaTg x1.1 retained), sha256 `f4a2aa6c24d887343e7003a0a226d6f6d702dfd707e31ad05d44773ea9c5dcbe`.

**Verdict: NO RESULT at cap.** Both containers exited with code 139 (SIGSEGV) after 10.9 s
(0.19 nA) and 8.3 s (0.27 nA) without writing a trace or report. The post-trough drive
prediction is therefore untested: no band was held and none was missed, and the rejection
clause could not be applied. Cap 1 is spent by the launch; no second evaluation and no
parameter change were made. The I cell stays "experimental, borrowed, labelled" and the
Noise1 sweep 48 seal stays closed.

## Wall clocks (measured)

| Event | Value |
| --- | --- |
| Wait for the transfer gate started | 2026-09-07 22:27 UTC |
| Gate file `h01-i-transfer/step1-decision.json` present | 22:38:30 UTC |
| Runner launched (`Start-Process`, pid 40720) | 22:38:45 UTC |
| Container `h01-i-reserve-g-natg-axon15-019` | 10.95 s, rc 139, not aborted |
| Container `h01-i-reserve-g-natg-axon15-027` | 8.31 s, rc 139, not aborted |
| Runner total | 19.26 s |

The finalist's retained runs took 12 to 170 s, so the crash falls at the end of the model
build or the start of integration. Container stderr was not retained: `h01_campaign.py`
runs docker with `capture_output=True` and discards it, and `--rm` removed the containers.
Docker's die events confirm exit 139 for both names. Another agent's container
(`h01-e-gain-g0-b3-sweep43`) was running throughout.

## Preparation checks (recorded before launch)

- `docker run --rm -v <worktree>/docs/evidence:/evidence braintrace-h01-neuron:9.0.2 python
  /evidence/h01_pv_neuron_reference.py --help` printed the usage with `--scale MECH:REGION:FACTOR`.
- `h01_campaign.py --stage 1 --dry-run`: gate open, one candidate at 019 and 027.
- This sparse worktree has no `.cache`; a junction `.cache -> .worktrees/h01-braincell/.cache`
  makes the manifest's relative `cache_dir` resolve (`/work/source/NeuronTemplate.hoc` found).
- Driver-API toy check (not a model run): one bare `h.Section` with the library's NaTg;
  `psection()` and `gbar_NaTg` setattr x1.5, the two calls the new scale loop makes, succeed
  in the image. The crash is not in those calls in isolation; its cause is undetermined.

## Band table

| Input | Row | Predicted | Observed | Status |
| --- | --- | --- | --- | --- |
| 0.27 nA | cycle 2 | <= 12 ms (from 16.37) | none | untested |
| 0.27 nA | count | 40 to 43 (from 37) | none | untested |
| 0.27 nA | width, cycle 1 | 0.221 +/- 0.006 ms | none | untested |
| 0.27 nA | peak | <= +22 mV | none | untested |
| 0.27 nA | troughs 1-3 | within 1 mV of -78.99 / -79.30 / -79.61 | none | untested |
| 0.19 nA | cycle 2 | <= 22 ms (from 34.76) | none | untested |
| 0.19 nA | count | 15 to 17 (from 14) | none | untested |
| 0.19 nA | width, cycle 1 | 0.221 +/- 0.006 ms | none | untested |
| 0.19 nA | peak | <= +22 mV | none | untested |
| 0.19 nA | troughs 1-3 | within 1 mV of -80.13 / -80.54 / -80.65 | none | untested |

Rejection clause (cycle 2 at 0.27 nA moving less than 12.8 ms, or a loop break): not
applicable. Usable tier, 1 mV contract and initiation (axon-first) per input: untested.
`h01_i_reserve_bands.py` (tested, `h01_i_reserve_bands_test.py`) is the scorer that would
have produced the held/missed columns from the usable-tier tables.

## Reassessment

- The two unexplained I rows stand as before: the post-trough inward drive (human refires
  within 6 to 10 ms of a -79 mV trough at every input) and the accommodation along the
  train (threshold -61 to -55 mV, late fall -294 V/s); with them the count (14 vs 12,
  37 vs 43) and the cycle-1 width (0.221 vs 0.267-0.273 ms).
- Stage 0's reading (soma h 0.98 at trough + 6 ms; axial current outward at every trough)
  still narrows the drive row to the axon-first pathway; whether axonal NaTg density
  carries it is unknown.
- Returned to the user: whether a container that crashed before integrating counts as the
  spent reserve; if not, a crash-diagnosis run with stderr retained (the runner should
  record it) before any re-evaluation. Nothing here was re-run.
