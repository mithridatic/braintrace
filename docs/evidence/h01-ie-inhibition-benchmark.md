# SP5 100 ms benchmark: untested (killed at the 900 s abort)

Record: `h01-ie-inhibition/benchmark.json` (2026-09-07). Spec:
[functional inhibition](../specs/2026-09-07-h01-functional-inhibition.md); manifest
`h01-ie-inhibition-manifest.json`. Run 1 of the cap of 6, spent.

## What ran

`python -m docs.evidence.h01_ie_functional_inhibition benchmark` from the `h01-pair` root
(`PYTHONPATH=.`, validation interpreter from the sibling `h01-braincell` cache;
`braintrace.__file__` resolved to `h01-pair`). Arm `disconnected`, 100 ms, E 0.6 nA constant,
I pulses at 20, 45, 70, 95 ms, dt 0.005 ms, launched detached at 15:54:14.

## Measured (under load)

| Quantity | Value |
| --- | --- |
| Wall clock | not measured: killed at 913 s (16:09:50) with no trace and no run-log entry |
| Wrapper output | `Circuit constructed: disconnected` at about 190 s, nothing after |
| Load | CPU at 100 percent throughout; about 12 other H01 python jobs running (C3 edge-list scan, PV braincell reference, E gain split, I energetic, verified network) |
| Predicted (anchor 4.25 s per simulated ms) | 425 s; the run exceeded it by more than a factor of two |
| E spikes, E spike times, I spike times, finiteness | unobserved |

All timing is **measured under load**; the anchor was taken on a quieter machine.

## Drive rule

The registered rule (>= 1 E spike in 100 ms: 0.6 nA stands; none: report) could not be
evaluated, because no E trace exists. The drive was not changed. Whether E fires at 0.6 nA in
this circuit remains unobserved; the single permitted change to 0.8 nA stays a coordinator
decision and is not triggered by this result.

## Derived cost (labelled derived, lower bounds only)

From 913 s for less than 100 simulated ms: at least 9.13 s per simulated ms under load.

| Arm | Derived lower bound |
| --- | --- |
| each 300 ms arm (disconnected, measured, soma) | >= 2740 s (>= 45.7 min) |
| measured-halved (300 ms, dt 0.0025) | >= 5480 s (>= 91.3 min) |

These are floors, not estimates: the benchmark did not finish. Every 300 ms arm exceeds the
15 min approval threshold under the current load, so no 300 ms arm may be launched on this
record (SP0). Options for the coordinator: rerun the benchmark when the machine is quieter, or
raise the abort for the benchmark alone with a measured estimate in hand from SP1 (11.5 s per
simulated ms, 4 cells, which predicts about 19 min for 100 ms and 58 min per 300 ms arm).
