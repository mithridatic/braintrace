# H01 next conversation: one session, keep what works, drop what does not, start learning

Rule from J (2026-09-16): simulate the human cells and let them learn. Cells that
work are kept. Cells that do not work are dropped. This is the last session on
the cells themselves; no new fitting campaign, no new donors, no anatomy tuning.

Branch `campaign/h01-driven-window-20260914` (local worktree
`.worktrees/h01-driven-window`; Vast 50616476 `/workspace/braintrace` is on the
same branch). Stage record: `docs/evidence/h01-driven-window-20260915/README.md`.

## What is known going in

All 104 cells build, step a 50 ms window and converge numerically on the Vast
4090 (runtime gate 4/4, refinement 103/104). Whether a cell behaves like a human
cell was tested on one cell per donor type under the donor recording's own
current step (`docs/evidence/h01_anatomy_transfer_run.py`):

| Donor | Cells using it | Test cell | Result |
| --- | ---: | --- | --- |
| L4 pyramidal, Allen 527952884 | 22 | 3761379470 | 13 spikes vs human 12, clean train, returns to rest → works |
| L2 pyramidal, Allen 541563728 | 57 | 955432427 | 2 spikes then stuck at -27 mV → fails |
| L5 PV basket, HL5BN1 | 18 | 4853956860 | plateau at -37 mV, no spike → fails |
| L3 SST, HL5MN1 | 7 | 4420044370 | fires with no input, then blocks → fails |

Expectation, not a measurement: roughly 20 of 104 keep, 80 drop. Every cell gets
its own test before that is written down.

## Step 1. Test all 104 cells (about 2 h of box time, runs in the background)

One run per cell with its already-assigned donor, under that donor's primary
protocol, exactly as the four test runs were made:

```
# on the box, from /workspace/braintrace; labels transfer-all-<cell>
var/h01-driven/run_transfer.sh transfer-all-<cell> 1800 -- --cell <cell> --donor <donor_key> --polarity <E|I> \
  --pulse-on-ms <1020|270> --pulse-ms 1000 --duration-ms <2100|1500> --current-na <I> \
  --registered-count <human> --donor-model-count <fit>
```

Per donor: E L2 → 1020/2100 ms, 0.31 nA, human 10; E L4 → 1020/2100, 0.09 nA,
human 12; I PV → 270/1500, 0.19 nA, human 12; I SST → 270/1500, 0.10 nA, human 14
(repeats 14,14,13,12). Cell → donor/polarity comes from
`docs/evidence/h01-population-types.json` (`rows[*].donor_key`, `polarity`).
Write the 104 commands into a chain script like `var/h01-driven/transfer-chain.sh`
and run four chains concurrently pinned with `taskset` (single-cell processes
hold ~250 threads; the container limit is 7,680; do not run population builds at
the same time). Each run is 3-4 min on the GPU.

Keep rule, registered before the runs (write it into
`docs/specs/2026-09-16-h01-keep-drop.md` first):
- finite everywhere;
- rests within 10 mV of the donor recording's rest before the pulse and returns
  there within 200 ms after it;
- no spike before the pulse;
- count inside the human count ± max(2, 30 percent) at the primary input;
- the dt-half repeat (0.0025 ms) reproduces the count.
`docs/evidence/h01_anatomy_transfer_decision.py` already scores counts; extend it
with the rest/return/pre-pulse checks and a `keep` flag per cell (test first),
and write `docs/evidence/h01-keep-drop/decision.json` listing kept and dropped
cells with the reason for each drop.

## Step 2. Build the kept population and run its driven window (about 1 h)

- `plan_h01_cells` / `make_h01_network` take the cell list from the topology
  nodes plus a components inventory; make a filtered topology JSON
  (`docs/evidence/h01-kept-network.json`) with only kept nodes and only the
  contacts whose both ends are kept, and a filtered components inventory. Test.
- Probe: each kept cell gets its donor's primary current (0.31 / 0.09 / 0.19 /
  0.10 nA) from 2 to 40 ms instead of the 1 nA 2-5 ms probe. That is a one-line
  change in `make_h01_network` (currents dict) plus `delay_ms`/`duration_ms`
  arguments on `examples/h01_verified_network.py`. Register it in the spec.
- Run `ei` and `disconnected` for 50 ms at dt 0.000625 through
  `var/h01-driven/run_h01.sh` (two at a time via `queue.sh`), then
  `h01_driven_window_gates.py`. Change `h01_population_control_gate.py` from
  `array_equal` to `allclose(atol=1e-5)` first, with a test, because GPU
  `atomic_add` in the DHS kernel leaves 9e-7 mV run-to-run differences. If no
  kept contact has a firing presynaptic cell, say so and record the window as
  delivery-unobservable, not failed.
- Rebuild the construction reference for the kept set with
  `h01_population_build_gate`, and regenerate the ledger
  (`docs.evidence.h01_population_ledger`; add a `kept_cells` field to the report).

## Step 3. Let them learn; Example 21 grows the good ones and prunes the rest

J's rule: what is good propagates through Example 21's structural adaptation.
One verified good human cell is enough to start (L4 3761379470 already is one);
more kept cells only add diversity. Nothing biological blocks this step.

- Substrate: the kept network from Step 2 through the H01 adapter
  (`examples/pp_prop/example21_arc_adapter.py`; contract in
  `docs/specs/2026-09-08-h01-example21-adapter-contract.md`).
- Growth and pruning are already implemented and tested: `neuron-grow` twins a
  cell (`structural.grow_adam_for_twins`, adapter line ~1900) and `neuron-prune`
  removes one (`_neuron_prune`, ~1816); edge grow/prune likewise. Contract and
  bounds: `docs/specs/2026-08-25-example21-bounded-structural-adaptation.md`
  (Muon, deterministic selection, optimizer-state remapping on rebuild).
- Run the default Example 21 evolution on the kept manifest
  (`docs/specs/2026-09-13-h01-arc-evolution-machinery-run-design.md`,
  `braintrace/datasets/h01_arc_manifest.py`) with structural adaptation on, a
  60 min cap, checkpoints kept. With ~20 cells the per-event cost is ~5x below
  the 104-cell figure that never produced a candidate in 2 h 46 min. Report the
  pipeline's own pass@1 (every previous run recorded 0) and the grow/prune
  ledger: which cells were twinned, which pruned. Twinned cells are the ones
  that earned their place; that ledger is the "good cells" list going forward.
- Do not spend the session on the score. The deliverable is a kept population
  that runs, learns, grows and prunes, with its numbers written down.

## Step 4. Write it down and stop

`docs/evidence/h01-keep-drop/README.md`: kept list, dropped list with reasons,
driven-window gate verdicts, the learning run's checkpoint and score, wall time
and box hours. Update `docs/h01-causal-model.md` Y7 with the per-cell counts in
the same commit as the decision JSON. Regenerate the ledger. Push the branch and
sync the box. Then stop; no follow-up campaign is authorised.

## What is deliberately not done

- No per-cell density fits (proposal kept for the record in
  `docs/specs/2026-09-15-h01-per-cell-human-fit.md`; not for this session).
- No new donors (`h01-driven-window-20260915/allen-human-perisomatic-models.json`
  is the inventory, nothing more).
- No anatomy or channel tuning to rescue a dropped cell.
- No 104-cell refinement rerun for the one cell at 1.009 mV; if that cell is
  dropped by Step 1 the point is moot, otherwise record 1.009 mV as its reading.

## Operating rules that held

- Register the rule before the runs; report every cell against it.
- Runs on Vast; every run has launch/terminal JSON under `var/h01-driven/`.
- Two population processes at a time; single-cell chains four at a time; check
  `ps -eo nlwp | awk '{s+=$1}'` stays under ~5,000.
- Gate against the Vast build; the output-site tie-break fix (`_nearest_cv`) is
  on this branch and any rebuild re-pins 17 cells' sites.
- Update the causal model in the same commit as every decision JSON.
