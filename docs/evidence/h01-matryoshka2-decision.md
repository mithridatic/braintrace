# Stage 2 decision: Matryoshka characterisation on the phase plane

Method: Hartshorne 2020 small multiples (p037, p038) with the family order
elemental, cyclical, structural, temporal (p034, p131, p135). Figures:
[I](h01-multivari-i.png), [E](h01-multivari-e.png). Family ranges in units of the
combined decision limit: [I](h01-matryoshka2-i.md), [E](h01-matryoshka2-e.md).
Only rows limited by both the human and the model repeats are ranked.

| Cell, model | Elemental | Cyclical | Structural | Temporal | Named |
| --- | --- | --- | --- | --- | --- |
| I, source | 26.3 | 3.9 | 0.0 | 12.9 | elemental |
| I, candidate | 6.8 | 3.8 | 2.1 | 6.7 | elemental (temporal equal) |
| E, source | 120.0 | 15.0 | 0.0 | 5.8 | elemental |
| E, candidate | 54.1 | 11.7 | 0.0 | 3.9 | elemental |

Prediction check (registered in the Stage 1 decision): I elemental plus temporal,
E elemental at every spike and input. Both confirmed. The rejection clause
(elemental family differing beyond the limit in the first cycle at both inputs)
is the case for E and for the I source, so Stage 3 starts at the first spike.

## What the loops show

**I cell.** The source loop is a single large ellipse: peak +44 mV, rise 1388 V/s,
fall −141 V/s. The candidate loop lies on the human loop in rise, peak and fall at
both inputs, and departs on the return branch: it closes at −73 mV where the human
closes at −79 mV, at every cycle and both inputs (elemental, minimum_mv 5.0 units).
The human rising branch carries a shoulder between −60 and −45 mV (dV/dt 100 to 250
V/s before the main upstroke) that neither model reproduces. Along the train the
human threshold climbs 8 mV while the model's stays flat (temporal, 3.6 units).
At 0.27 nA the human fires an initial burst of three 6 ms cycles and settles to
22 to 25 ms; the candidate's burst runs nine 5 ms cycles before settling to 26 ms.
At 0.19 nA the source reaches the human's late cycle length (120 vs 100 to 125 ms)
with the wrong loop, the candidate the right loop with the wrong late cycle (88 ms):
the loop and the late cycle are coupled through the sub-systems that were swapped
(the G1a and G3 interdependency of the last campaign, seen now as two families).

**E cell.** Both models rise two to three times faster than the human (351 V/s
against 641 and 1019) at every spike, with a smooth parabolic upstroke where the
human has a two-stage upstroke (shoulder near −45 mV). Threshold is 3 mV high.
The candidate's cycle lengths follow the human's (272/307/294 against
220/307/273 ms) while the source stops at 190 ms; the temporal family is small.
The late plateau and return at sweep 43 are within two limits.

## Decision

Stage 3 opens for both cells on the function branch, at the first spike:

- I: which boundary's charge sets the depth of the post-spike minimum, and which
  boundary's slow state carries the threshold climb along the train.
- E: which boundary carries the upstroke charge that makes dV/dt two to three
  times too fast at the same peak voltage: sodium charge in, or capacitive and
  axial load out. A peak at the right voltage with a rise twice as fast points at
  the load (capacitance and axial current), not the sodium boundary alone.

Registered predictions for Stage 3: I minimum depth follows the potassium and
axial charge in the falling phase; E rise rate follows the capacitive plus axial
charge fraction during the upstroke, and zeroing the somatic sodium density moves
the peak, not the ratio.
