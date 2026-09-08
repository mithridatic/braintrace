# SP5 100 ms benchmark: measured, machine quiet except the C3 rescan

Record: `h01-ie-inhibition/benchmark.json` (2026-09-07 19:28 to 19:34). Spec:
[functional inhibition](../specs/2026-09-07-h01-functional-inhibition.md) with the 2026-09-07
addendum (1500 s abort); manifest `h01-ie-inhibition-manifest.json`. Run 1 of the cap of 6.

## What ran

`python -m docs.evidence.h01_ie_functional_inhibition benchmark` from the `h01-pair` root
(`PYTHONPATH=.`, validation interpreter from the sibling `h01-braincell` cache;
`braintrace.__file__` resolved to `h01-pair`), on commit `c56616a` (programme tip merged with
the BrainCell construction/init/DHS performance commits). Arm `disconnected`, 100 ms, E 0.6 nA
constant, I pulses at 20, 45, 70, 95 ms, dt 0.005 ms, launched detached at 19:28:40 after the
wait condition held (E-gain evaluation 1 inputs all completed, `docker ps` only `synapse`, no
python job over 300 MB, CPU 29 percent). Abort registered at 1500 s; not reached.

## Measured

| Quantity | Value |
| --- | --- |
| Wall clock | 347.0 s (3.47 s per simulated ms, construction included) |
| Predicted (anchor 4.25 s per simulated ms) | 425 s; measured is 0.82 of the anchor |
| E spikes | 2, at 21.775 and 38.31 ms |
| I spikes | 1, at 27.125 ms (event time; voltage peak 32.5 mV at 23.585 ms, from the 20 ms pulse) |
| I at the 45, 70, 95 ms pulses | depolarises to about -60 mV at pulse end, no spike |
| E after 40 ms | settles near -69 mV under the constant drive, no further spike to 100 ms |
| Finiteness | every saved array finite |

Predictions: E fires >= 1 spike by 100 ms (held); first spike between 20 and 60 ms (held,
21.775 ms); I fires once per pulse (not held: 1 of the 3 evaluable pulses).

## Drive rule

E fired 2 spikes at 0.6 nA, so the registered drive stands; it was not changed. Flag for the
coordinator: the 300 ms control rule (>= 3 E spikes with two full cycles) is undecided by this
window, since E fired twice before 40 ms and then accommodated for 60 ms; whether a third spike
appears within 300 ms is unobserved. The single permitted change to 0.8 nA remains the
coordinator's decision.

## Derived cost (labelled derived)

From the measured 3.47 s per simulated ms, scaled linearly:

| Arm | Derived cost |
| --- | --- |
| each 300 ms arm (disconnected, measured, soma) | 1041 s (17.4 min) |
| measured-halved (300 ms, dt 0.0025) | 2082 s (34.7 min) |

If about 120 s of the 347 s is construction (the `Circuit constructed` line appeared between
12 s and 133 s; the log was polled every 2 min), the simulation rate is about 2.27 s per
simulated ms and a 300 ms arm is about 800 s (13.3 min), the halved arm about 1480 s (24.7 min);
that split is derived, not measured. On the linear estimate every 300 ms arm is above the 15 min
approval threshold, so approval is needed before any 300 ms arm. No 300 ms arm was launched.

## Historical: first attempt, killed (untested)

Record: `h01-ie-inhibition/benchmark-2026-09-07-killed.json`. Launched 15:54:14 under load (CPU
100 percent, about 12 other H01 jobs), killed at the 900 s abort at 16:09:50 with no trace and
no run-log entry (`Circuit constructed` at about 190 s, nothing after). By the programme rule a
killed run is untested and does not spend the cap (`prior_evaluations` reverted to 0 before the
rerun).
