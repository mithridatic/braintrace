# SP12. E cell: a spike-triggered, long-lasting outward shift

Status: APPROVED 2026-09-11 (user: "go for it"), registered before any run. Drafted the
same day from the SP11 stage-0 observations. Manifest
[h01-e-sahp-manifest.json](../evidence/h01-e-sahp-manifest.json); mechanism
[KsAHP.mod](../evidence/h01-l2-mechanisms/KsAHP.mod); scorer and dose calculator
`docs/evidence/h01_e_sahp.py`.

## Observations that motivate it (retained data plus SP11 stage 0)

Sweep 56 (200 pA): the human ramps from -70 mV to -61 mV (p50 of 1100-1200 ms), fires
once at 1225.4 ms, then sits at -67.5 (mid pulse, 1300-1500 ms) and -67.1 mV (late pulse,
1800-2000 ms) for the remaining 800 ms: a 6 mV shift after one spike that does not decay
within the pulse. Unchanged B3 fires at 1147, 1216, 1476 and 1772 ms and its interspike
level is -62 to -65 mV. At 110 pA (sweep 43, no spike) the same model matches the
human's onset rows within 0.4-0.9 mV (`h01-e-gain/stage-g0-decision.json`). Thus the
difference at 200 pA appears only after a spike.

Sweep 50 (250 pA): human spikes at 58, 92, 312, 620 and 893 ms (a doublet, then about
300 ms cycles); B3 at 55, 76, 123, 307, 478, 643, 804 and 961 ms (a triplet, then about
165 ms). Sweep 53 (310 pA): late cycles match (about 120 ms). Sweep 55 (350 pA): the
human's late cycle is about 88 ms, the model's about 103 ms.

Human threshold climbs along the train (-56.4 to -52.8 mV at 310 pA; -55.8 to -54.3 mV
at 250 pA); the model's stays at -57.2 mV at every input.

SP11 stage 0, corrected (see the erratum in
[stage-close-decision.json](../evidence/h01-e-currents/stage-close-decision.json)): the
current tables report one soma segment of nine (65.8 of 592 um2). Whole-soma interspike
currents at 200 pA are leak -0.058, Kv3 -0.017, SK -0.016, NaTs +0.010, Nap +0.012 and
Im -0.0002 nA, summing to -0.069 nA (35 percent of the applied 0.196 nA). Removing all
of Nap therefore reaches 59 percent of the 0.020 nA late-pulse p50 offset and 20 percent
of the 0.059 nA mid-pulse mean offset; Im still needs a 100x dose that acts at the
310 pA threshold. Nap removal is not a spike-triggered lever (Nap is open at every
plateau), so it is spent here as the comparison arm rather than as a repair candidate.

## Hypothesis family

A spike-triggered outward current with a decay of the order of one second, absent from
the Allen genome: a slow calcium-activated potassium current (sAHP) or a sodium-activated
potassium current (KNa). Its signature is (1) a post-spike level shift that persists over
the whole 1 s pulse at rheobase, (2) an accumulating brake at 250 pA that spreads the
cycles to about 300 ms, (3) little effect on the late 310 pA train, where the existing
SK/calcium setting already lands the rate, and (4) a rising threshold along the train
only if paired with sodium slow inactivation, which this family does not supply.

The carrier cannot be the fit's calcium: the recorded soma calcium at 200 pA rises from
1.0e-4 to 2.8e-4 mM with the depolarisation and is not spike-shaped (Ca_LVA at the
plateau, 80 ms removal), so a calcium-gated gate would open at 110 pA as well. The
mechanism is therefore a phenomenological gate `z` that opens only above -20 mV (crossed
during a spike) with a 1 ms time constant and closes with a 1000 ms time constant; below
-20 mV its steady state is zero. This isolates the family's signature from its carrier:
a pass says a spike-triggered 1 s outward shift describes the response, not which channel
carries it.

Excluded by construction: a uniform outward conductance active from -72 mV (it would move
the 110 pA rows past 1 mV), and a faster SK dose (the 310 pA late rate is already landed
and SK 0.5 to 0.35 did not remove low-drive spikes).

## Implementation (done before the runs)

`KsAHP.mod` in the library (`USEION k`, `RANGE gbar, vhalf, slope, tau_on, tau_off`),
inserted through the existing `--insert-density KsAHP:soma:VALUE` flag (the soma reversal
row already exists, so `insert_density_fit` adds only the density row). The driver
records `soma_KsAHP_ma_cm2` and `soma_KsAHP_z` whenever KsAHP is inserted on the soma.
On the box the mod file is copied to the `kv3-closing-source` library root and the
library recompiled with `nrnivmodl`; the driver's `mechanism_library_sha256` records the
change. Existing candidate identities (flag hashes) do not change. The pre-spike band
below doubles as the recompiled library's reproduction check.

## Registered doses

`h01_e_sahp.py register` integrates the gate along the retained `m0-b3-currents-sweep56`
trace with only its first spike kept (1147.2 ms; 0.99 ms above -20 mV; gate 0.51 after
the spike; mean gate 0.242 over 1800-2000 ms) and sizes the density at the human's late
level (-67.06 mV, driving force 40 mV to ek -107 mV, soma 592 um2):

| Dose | Target offset | gbar (S/cm2) |
| --- | ---: | ---: |
| A (`s0-ksahp-a`) | 0.020 nA (SP11 late p50 offset) | 3.4968e-4 |
| B (`s0-ksahp-b`) | 0.059 nA (SP11 mid-pulse mean offset, 5.8 mV at 98 MOhm) | 1.0316e-3 |

Comparison arm `s0-nap-zero`: B3 with soma Nap density 0 (whole-soma +0.012 nA removed,
open at every plateau and at 110 pA at one eighth of its 200 pA value).

## Registered predictions

Stage 0 (sweep 56 only, three evaluations): a KsAHP dose holds count 1-2, late- and
mid-pulse p50 within 1 mV of the human, the trace before B3's first spike within 0.1 mV
and the first spike within 1 ms of B3's, and axon-first initiation. The comparison arm
holds count 3 or more and a late p50 shift between 0 and -1.5 mV against B3. The survivor
is the dose with count 1 and the smaller late-level error.

Stage 1 (the survivor at sweeps 50, 53, 43; one evaluation): 250 pA count 5-7 with cycles
4-5 of 250-350 ms; 310 pA count 9-10 with late cycles within 12.8 ms of 120 ms; every
sweep-43 onset row within 1 mV of the human and within 0.1 mV of g0-b3.

Rejection: no dose holds every stage-0 band (the family with a 1 s tail is closed at this
gate shape); the 310 pA count falls below 9; a sweep-43 row moves by 0.1 mV or more; the
250 pA count leaves 5-7. If the comparison arm lands count 1 within 1 mV of the human,
the spike-triggered reading is not isolated and that is recorded before any pass.

Both tiers reported at every stage. A pass is a gain-split result for B3, not a
promotion: the contract tier's first-spike time (human 1225 ms, model 1147 ms) is not
addressed by a mechanism that is silent before a spike, and sweep 54 stays sealed.
