# SP12 (proposed). E cell: a spike-triggered, long-lasting outward shift

Status: DRAFT written 2026-09-11 from the SP11 stage-0 observations. Not approved,
not run. It needs a new mechanism in the `kv3-closing-source` library, so it is an
implementation spec for the user to approve before code.

## Observations that motivate it (retained data plus SP11 stage 0)

Sweep 56 (200 pA): the human ramps from -70 mV to its threshold (-54.7 mV) over
206 ms, fires once, then sits at -67.1 to -67.7 mV (p50 of 1300-1500 and 1800-2000 ms)
for the remaining 800 ms. Unchanged B3 fires at 127, 196, 456 and 752 ms, and its
interspike level is -62 to -63 mV. At 110 pA (sweep 43, no spike) the same model
matches the human's onset rows within 0.4-0.9 mV (`h01-e-gain/stage-g0-decision.json`).
Thus the 5 mV difference at 200 pA appears only after a spike.

Sweep 50 (250 pA): human spikes at 58, 92, 312, 620 and 893 ms (a doublet, then about
300 ms cycles); B3 at 55, 76, 123, 307, 478, 643, 804 and 961 ms (a triplet, then about
165 ms). Sweep 53 (310 pA): late cycles match (about 120 ms). Sweep 55 (350 pA): the
human's late cycle is about 88 ms, the model's about 103 ms.

Human threshold climbs along the train (-56.4 to -52.8 mV at 310 pA; -55.8 to -54.3 mV
at 250 pA); the model's stays at -57.2 mV at every input.

At 200 pA the model soma passes 0.188 of the applied 0.196 nA into the cable; its soma
ionic terms sum to about -0.009 nA (leak -0.0066, Kv3 -0.003, SK -0.0017, NaTs +0.0015,
Nap +0.0015, Im -0.0001). The fit places every active mechanism in the soma (Nap
2.8e-4, Im 3.0e-4 S/cm2); the dendrites are passive. Neither registered SP11 lever can
carry a 5 mV plateau offset (about 0.05 nA at the model's input resistance): Nap's
whole soma current is 0.0015 nA, and Im at -62 mV has m_inf 0.0045 (a 500x dose would
also act at -55 mV and remove the 310 pA train).

## Hypothesis family

A spike-triggered outward current with a decay of the order of one second, absent from
the Allen genome: a slow calcium-activated potassium current (sAHP), or a sodium-
activated potassium current (KNa). Its signature is (1) a post-spike level shift that
persists over the whole 1 s pulse at rheobase, (2) an accumulating brake at 250 pA that
spreads the cycles to about 300 ms, (3) little effect on the late 310 pA train, where the
existing SK/calcium setting already lands the rate, and (4) a rising threshold along the
train only if it is paired with sodium slow inactivation, which this family does not
supply by itself.

Excluded by construction: a uniform outward conductance active from -72 mV (it would move
the 110 pA rows past 1 mV), and a faster SK dose (the 310 pA late rate is already landed
and SK 0.5 to 0.35 did not remove low-drive spikes).

## Implementation needed

One mod file `KsAHP.mod` (or `KNa.mod`) in the library, a `--insert-density` entry in
`h01_l2_neuron_reference.py` for it, and a soma-current probe for its current and gate.
The driver hashes the candidate flags, so the new mechanism changes no existing
candidate identity.

## Registered predictions (to be fixed with the dose before any run)

Stage 0 (dose scan on sweep 56 only, up to two evaluations): the post-spike late-pulse
p50 lands within 1 mV of -67.4 mV with count 1 (band 1-2). Stage 1 (the surviving dose at
sweeps 50, 53, 43): 250 pA count 5-7 with cycles 4-5 of 250-350 ms; 310 pA count 9-10
with late cycles within 12.8 ms of 120 ms; sweep-43 onset rows within 1 mV of g0-b3 (the
mechanism is silent without a spike). Rejection: the 310 pA count falls below 9, or the
sweep-43 rows move by 1 mV or more, or the 200 pA post-spike level needs a dose that
silences 250 pA.

Both tiers reported. A pass is a gain-split result for B3, not a promotion; sweep 54
stays sealed until a registered prediction exists.
