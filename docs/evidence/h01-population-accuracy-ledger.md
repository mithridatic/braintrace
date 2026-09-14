# The population accuracy ledger

Generated 2026-09-13 from [h01-population-accuracy-ledger.json](h01-population-accuracy-ledger.json)
by [h01_population_ledger.py](h01_population_ledger.py) (8 tests). Every number below is read from
a committed decision JSON; nothing here is a new measurement.

This page exists because the campaign was about to spend a stage fixing a blocker that was already
fixed, and because the figure it would have produced hides three terms that are zero.

## The correction

The last accuracy report multiplied `71.8 percent x 55/104 x 12/104` and called the product about
4 percent. **The `12/104` was stale.** The one-point SWC branch that stopped 7 of the 104 largest
components from loading was repaired on 2026-09-08, six days before the figure was quoted:

| Gate | Reading | Source |
| --- | --- | --- |
| Import | 104 of 104 components, 2,804,445 source segments, 0 missing, 0 extra | [`h01-population-import-104.json`](h01-population-import-104.json) `passed` |
| Construction | 104 cells, 808,495 compartments, 0 failures | [`h01-ready-104-implicit-build-decision.json`](h01-ready-104-implicit-build-decision.json) `status` |
| Initialisation | 104 cells, 925.8 s, exact construction-metadata match | [`h01-ready-104-implicit-init-decision.json`](h01-ready-104-implicit-init-decision.json) `status` |
| Forward pass | all six phases pass at dt 0.005 ms, peak RSS 14.0 GiB | [`h01-arc-probe-104.json`](h01-arc-probe-104.json) `status` |

So the build term is 104/104, not 12/104. The engineering blocker I proposed to attack does not
exist any more, and there is no work to do there.

## What the correction does not buy

Raising one term from 0.115 to 1.0 does not raise the product, because correcting the ledger also
exposes three terms that were never in it and are zero. Substituting the fixed term into the old
three-factor formula gives 38 percent; that figure is only reachable by assuming all three of the
following away.

### 1. A driven window at a physiological duration: no reading

Construction is not simulation. The forward pass that passes drives an **all-ones synthetic
441-feature probe**, not an encoded episode, and records `physiology: unqualified`. The two
attempts at a real driven window both died:

- 10 ms explicit, dt 0.005: failed after 2,705 s on `Nonfinite trace: cell_7196644737/output_voltage`
  ([`h01-ready-104-ei-10ms-decision.json`](h01-ready-104-ei-10ms-decision.json)).
- 10 ms implicit retry, dt 0.000625: aborted at the wall cap after 21,286 s, exit code 15, no
  result ([`h01-ready-104-implicit-ei-10ms-r2-launch.json`](h01-ready-104-implicit-ei-10ms-r2-launch.json)).

The second attempt spent 5.9 hours and produced nothing, which is also a policy violation
(no hour-plus runs) and is the reason this ledger is being written from committed evidence rather
than from a third attempt.

### 2. The timestep: qualified at dt 0.000625 ms

*This section previously read "not qualified ... the traces were deleted and `.cache/h01` no
longer exists". Both claims were wrong.* All five rungs survived in the 2026-09-09
worktree-recovery stash, which was restored while cleaning up the stale worktrees. The ladder
closes as arithmetic, with no new simulation:

| pair | max \|ΔV\| | at | gate |
| --- | ---: | ---: | --- |
| dt 0.005 → 0.0025 | 6.365 mV | 5.0150 ms | FAIL |
| dt 0.0025 → 0.00125 | 3.471 mV | 5.0150 ms | FAIL |
| dt 0.00125 → 0.000625 | 1.820 mV | 5.0150 ms | FAIL |
| **dt 0.000625 → 0.0003125** | **0.933 mV** | 5.0144 ms | **PASS** |

Error ratios 1.83 / 1.91 / 1.95 — first order in dt, matching what the I-cell transfer study found
independently. **dt 0.000625 ms is qualified**, and it is exactly the step the failed r2 run was
already using: that run died on cost, not on accuracy.

One limit on the reading, stated because the ledger's whole purpose is not to let a term carry more
than its evidence: the gate is met by the **finest pair on the ladder**, so no rung below
0.0003125 ms confirms that convergence continues. The first-order ratios say the next halving
should land near 0.48 mV, but that is an extrapolation, not a measurement.

The earlier FAIL verdict in
[`h01-ready-cell7196644737-implicit-decision.json`](h01-ready-cell7196644737-implicit-decision.json)
compared only the two coarsest rungs. Reading:
[`h01-timestep-ladder.json`](h01-timestep-ladder.json) via
[`h01_timestep_ladder.py`](h01_timestep_ladder.py) (7 tests). The qualification is numerical
convergence on one isolated cell, not a claim about the other 103.

### 3. Anatomy transfer: measured once, and it fails

This is the term the campaign never named, and it is the binding one. The donor fits are made on
the **donor's own reconstruction**. Moved onto H01 anatomy, B3's parameters do not spike at all:

| | Donor reconstruction | H01 skeleton |
| --- | ---: | ---: |
| Spikes at 200 pA | 4 | **0** |
| Rest | −84.0 mV | −85.8 mV |
| Response at 200 pA | −64 to −65 mV | −78.2 mV plateau |
| Onset capacitance | 125 pF | **785 pF** |
| Input resistance | ~98 MΩ | ~38 MΩ |

Source: [`h01-e-morphology/stage-1-decision.json`](h01-e-morphology/stage-1-decision.json). The
converted H01 skeleton (2,460 µm of cable, 9,975 nodes, 8,551 sections) is a six-fold larger
electrical load than the Allen reconstruction the fit was made on, so at the donor's own rheobase
the cell sits on a plateau. Whether that load is physical or an artefact of the conversion
(skeleton radii used as cable radii) was explicitly left unestablished by that stage.

Every other term in this ledger is conditional on this one. Donor accuracy, type coverage and a
working build all describe a model that, on the anatomy it is meant to run on, is silent.

## The ledger

| Term | Value | Measured | What it is |
| --- | ---: | --- | --- |
| `donor_accuracy` | 0.718 | yes | B3 on ten elements at 310 pA, against Allen 541563728's own recording. Scoped to the **28** cells B3 is the type match for. |
| `type_coverage` | 0.529 | yes | 55 of 104 cells matched in layer **and** class. |
| `construction` | 1.000 | yes | import, construction, initialisation and a **synthetic** forward pass, all 104. Named `construction`, not `build`: the forward pass it credits drives an all-ones probe, and whether the population *runs* is the `driven_window` term below. |
| `driven_window` | 0.000 | no | no qualified driven window at 104 cells. |
| `timestep` | 1.000 | yes | qualified at dt 0.000625 ms (0.933 mV to the next halving), on the ladder's finest pair; first order in dt. |
| `anatomy_transfer` | 0.000 | **yes** | the donor's fit does not spike on H01 anatomy. |

**Product as measured: 0 percent.** Product if construction is counted as a working simulation and the
two unqualified terms are set to one: **38 percent** — and that figure additionally assumes the
three donors whose published fits were *reproduced and rejected* against their own recordings
score like the one donor that was scored.

## The defensible statement

> 71.8 percent on the one human L2/3 cell where the accuracy is measured, which is the type donor
> for 28 of the 104 H01 cells. Construction is qualified 104 of 104. A driven physiological window,
> and the transfer of any donor fit onto H01 anatomy are unqualified, and the one
> anatomy-transfer reading that exists is negative. The integration step is qualified at
> dt 0.000625 ms.

Anything shorter than that hides a zero.

## What this changes about what to do next

The ranking of remaining work inverts. It was: fix the build, then close the climb. It is now:

1. **Anatomy transfer** — the only term with a negative reading, and the one that makes every
   other term moot. The stage-1 record already names the next split: check the H01 conversion's
   membrane area against the H01 surface mesh before that skeleton is used again, and separately
   scale the donor's dendritic load by 1.5 and 2 to see whether the response degrades gradually or
   falls off a cliff. Neither needs the population; both are single-cell runs on the donor.
2. ~~The timestep ladder~~ **DONE.** dt 0.000625 ms is qualified.
3. **The driven window** — no longer blocked on the step, only on cost. The r2 attempt was already
   running at the qualified dt and burned 5.9 h for no result, so what is needed is a cheaper
   window (fewer cells, shorter duration, or a coarser step justified by a per-cell ladder), not a
   fourth full-population attempt.
4. The single-cell climb and rise, which carry 19.4 and 6.4 points of a score that applies to 28
   cells conditional on a transfer that currently fails.
