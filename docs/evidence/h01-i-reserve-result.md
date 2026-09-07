# I cell, SP4 reserve evaluation: result

Spec [2026-09-07-h01-i-reserve.md](../specs/2026-09-07-h01-i-reserve.md), manifest
`h01-i-reserve-manifest.json`, decision `h01-i-reserve/stage-1-decision.json` (this page
is derived from it). Candidate `g-natg-axon15` (finalist flags plus `--scale NaTg:axon:1.5`,
somatic NaTg x1.1 retained), sha256 `f4a2aa6c24d887343e7003a0a226d6f6d702dfd707e31ad05d44773ea9c5dcbe`.

**Verdict: FAIL at cap.** The rejection clause is met. 10 of 14 registered bands held; the four that missed are cycle 2 and count at
both inputs. Cycle 2 at 0.27 nA moved by -0.13 ms against a 12.8 ms decision limit, so the pre-registered
rejection clause is met. No loop break (width and peak held). Axon-first initiation holds at both
inputs. Cap 1 is spent by this run; no second evaluation and no parameter change. The I cell stays
"experimental, borrowed, labelled" and the Noise1 sweep 48 seal stays closed.

## The first launch was invalid, not a result

The launch at 22:38:45 UTC (pid 40720) exited 139 at both inputs after 10.9 s and 8.3 s before
`h.finitialize`, with stderr discarded by the runner. Reproduced at 20 ms with stderr captured and
isolated one variable at a time ([crash-isolation.json](h01-i-reserve/crash-isolation.json)):

| Split | Variable | rc | Outcome |
| --- | --- | --- | --- |
| A-candidate | registered candidate (finalist flags + scale NaTg:axon:1.5), HEAD driver | 139 | NEURON: section in the object was deleted |
| B-finalist-no-scale | control: finalist flags, no --scale, HEAD driver | 139 | NEURON: section in the object was deleted |
| C-scale-axon-1.0 | identity scale NaTg:axon:1.0, HEAD driver | 139 | NEURON: section in the object was deleted |
| D-scale-soma-1.5 | scale NaTg:soma:1.5, HEAD driver | 139 | NEURON: section in the object was deleted |
| E-driver-ac5ed12 | driver at commit ac5ed12 (--scale commit, before donor import), control candidate B | 0 | trace written |
| E-driver-c2ce97a | driver at commit c2ce97a (SP6c donor-import commit; identical to HEAD before the fix), control candidate B | 139 | NEURON: section in the object was deleted |
| F-probe-faulthandler | probe_myelin.py: load HL5BN1, iterate cell.myelin under python -X faulthandler | 139 |   File "/evidence/h01-i-reserve/crash-isolation/probe_myelin.py", line 21 in <module>; faulthandler: probe_myelin.py line 21: for m in cell.myelin |
| G-probe-section-exists | h.section_exists on the built HL5BN1 cell | 0 | myelin[0] 0, axon[0..1] 1, soma[0] 1 |
| H-fixed-driver-smoke | registered candidate A, fixed driver (_existing_sections guard) | 0 | trace written |

Cause: docs/evidence/h01_pv_neuron_reference.py commit c2ce97a (SP6c donor import) added `[m for m in cell.myelin if m.parentseg() is not None]` to the geometry record. NeuronTemplate.hoc declares myelin[1], deletes every section in init (forall delete_section()), and recreates myelin only in delete_axon() for HL23PN1/HL23MN1/HL5MN1; HL5BN1 takes delete_axon_BPO(), so cell.myelin is a reference to a deleted section. Iterating it from Python throws hoc_execerror 'section in the object was deleted' as an unconverted C++ exception; the interpreter aborts (terminate called), exit 139. The new --scale code is not involved: the control without --scale crashes identically and the driver at ac5ed12 (which has --scale) runs.

Fix: _existing_sections(cell, name) walks indices with h.section_exists before reading the array; the geometry line uses it for myelin. Test: h01_pv_neuron_reference_test.py::test_myelin_geometry_skips_a_template_array_of_deleted_sections. HL5MN1 keeps its myelin geometry (section_exists is 1 there).

Runner: h01_campaign.py now writes <stem>.stderr.log (stderr + stdout tail) for every real run and records stderr_tail (last 20 lines) and stderr_log per row; tests in h01_campaign_test.py.

The `--scale` code, the library (12/12 mechanism hashes and the three donor hashes match the finalist's
report), the 1500 ms duration and the concurrent container were each excluded by a split. The crash
tested nothing about the candidate, so the reserve was not spent by it.

## Wall clocks (measured)

| Event | Value |
| --- | --- |
| Runner launched (`Start-Process`, pid 6720) | 2026-09-07T22:51:42Z |
| Container `h01-i-reserve-g-natg-axon15-019` | 36.0 s, rc 0, report written 2026-09-07T22:52:18Z |
| Container `h01-i-reserve-g-natg-axon15-027` | 74.8 s, rc 0, report written 2026-09-07T22:53:33Z |
| Runner finished | 2026-09-07T22:53:34Z (112 s total; abort 900 s never reached) |
| Concurrent | h01-e-gain-g0-b3-sweep50 (another agent) was running at launch and through both runs |

## Preparation checks (recorded before the valid launch)

- image_help: docker run --rm -v <worktree>/docs/evidence:/evidence braintrace-h01-neuron:9.0.2 python /evidence/h01_pv_neuron_reference.py --help: rc 0, usage lists --scale MECH:REGION:FACTOR (fixed driver)
- smoke_20ms: registered candidate JSON, 0.27 nA, --duration-ms 20, fixed driver: rc 0 in 5 s, 418 finite samples, regional_scales recorded, candidate sha256 f4a2aa6c... (crash-isolation.json split H)
- dry_run: h01_campaign.py --stage 1 --dry-run: gate open, g-natg-axon15 selected at 019 and 027
- mounts: .cache junction -> .worktrees/h01-braincell/.cache; library hashes (12/12 .mod and libnrnmech.so) and donor file hashes match the finalist's retained report
- tests: h01_pv_neuron_reference_test.py, h01_campaign_test.py, h01_i_reserve_bands_test.py: 96 passed

## Band table

| Input | Row | Predicted | Observed | Status |
| --- | --- | --- | --- | --- |
| 0.19 nA | cycle2_ms | <= 22.0 (from 34.76) | 35.068 | MISSED |
| 0.19 nA | count | 15 to 17 (from 14) | 14 | MISSED |
| 0.19 nA | width1_ms | 0.221 +/- 0.006 | 0.220 | held |
| 0.19 nA | peak_mv | <= 22.0 | 18.319 | held |
| 0.19 nA | trough1_mv | within 1 of -80.13 | -80.235 | held |
| 0.19 nA | trough2_mv | within 1 of -80.54 | -80.574 | held |
| 0.19 nA | trough3_mv | within 1 of -80.65 | -80.690 | held |
| 0.27 nA | cycle2_ms | <= 12.0 (from 16.37) | 16.496 | MISSED |
| 0.27 nA | count | 40 to 43 (from 37) | 34 | MISSED |
| 0.27 nA | width1_ms | 0.221 +/- 0.006 | 0.220 | held |
| 0.27 nA | peak_mv | <= 22.0 | 18.435 | held |
| 0.27 nA | trough1_mv | within 1 of -78.99 | -79.086 | held |
| 0.27 nA | trough2_mv | within 1 of -79.3 | -79.338 | held |
| 0.27 nA | trough3_mv | within 1 of -79.61 | -79.659 | held |

Rejection clause: cycle 2 at 0.27 nA changed by -0.126 ms (limit 12.8 ms): below the limit; loop breaks: none; met: True.

## Tiers, contract and initiation

| Input | Human count | Model count | Usable tier | 1 mV contract | Contract scorecard (pass/fail/unavailable) | Axon-first | Axon lead (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.19 nA | 12 | 14 | fail (fails: adaptation_ratio, width_ms10, width_ms11...) | FAIL (count FAIL, 24 rows) | 28/90/2 | True | 0.331 |
| 0.27 nA | 43 | 34 | fail (fails: adaptation_ratio, rate_hz, width_ms23...) | FAIL (count FAIL, 36 rows) | 132/207/90 | True | 0.224 |

Usable tier: the rate row passes at 0.19 nA and fails at 0.27 nA (34 vs 43); adaptation fails at both;
widths fail from cycle 2 at 0.19 nA and from cycle 3 at 0.27 nA (0.220 vs 0.267-0.283 ms); every trough
passes the usable limit. Contract: count fails at both inputs; at 0.27 nA the first event is inside 1 mV
and 1 ms of the human (rise residual -0.80 ms) and the second is +9.4 ms late; at 0.19 nA the first event is already +6.9 ms late.

## Against the finalist

| Quantity | 0.19 nA finalist | 0.19 nA reserve | 0.27 nA finalist | 0.27 nA reserve |
| --- | --- | --- | --- | --- |
| count | 14 | 14 | 37 | 34 |
| cycle2_ms | 34.76 | 35.07 | 16.37 | 16.5 |
| axon_lead_ms | 0.029 to 0.045 (initiation audit) | 0.331 | 0.029 to 0.045 (initiation audit) | 0.224 |

## Reassessment

- FAIL. Cycle 2 at 0.27 nA moved -0.13 ms (16.37 -> 16.50 ms), far inside the 12.8 ms decision limit; the count fell 37 -> 34 at 0.27 nA and stayed 14 at 0.19 nA; cycle 2 at 0.19 nA stayed 35.1 ms. Width, peak and the six troughs held, so the loop did not break and the change was a null, not an instability.
- Axonal NaTg x1.5 did move the axon: the axon-first lead grew from 0.03-0.05 ms to 0.22-0.33 ms and the cycle-1 troughs are within 0.1 mV of the finalist's, yet the soma refired no sooner. The post-trough drive is therefore not carried by axonal NaTg density on this geometry (the pre-registered reassessment): the somatic and axonal sodium boundaries are both excluded, and what remains is a slow state the model lacks.
- Closed: axonal NaTg density as the post-trough drive (this evaluation); somatic NaTg availability (stage-0 audit: h 0.98 at trough + 6 ms); somatic Ca_LVA (energetic search).
- Open: post-trough inward drive: refire within 6-10 ms of a -79 mV trough at every input (both sodium boundaries now excluded); accommodation along the train (threshold climb -61 to -55 mV, late fall -294 V/s); count 14 vs 12 (0.19 nA) and 34 vs 43 (0.27 nA); cycle-1 width 0.220 vs 0.267-0.273 ms.
- experimental, borrowed, labelled (unchanged); Noise1 sweep 48 stays sealed. cap 1 is spent by this valid run; no re-run and no parameter change.
