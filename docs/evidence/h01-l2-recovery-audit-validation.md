# Recovery prediction audit

The audit checks that the only genome change is a doubled somatic
calcium-removal time. Geometry, solver, initial-state rule, and other
parameters must match. The time and applied-current arrays must match
exactly over their common prefix. A longer control tail is permitted.

The result retains all complete events, the first three individual
interval errors, the first peak change, and the first interspike minimum
change. All three interval errors must shrink. The minimum and peak
constraints must also pass. A smaller count alone cannot pass the test.

Sixteen tests pass with 100% statement coverage. They include one failed
interval among two improved intervals, an excessively deep minimum,
a changed peak, missing events, changed input, changed setup, and an
incorrect genome intervention. Two regression tests first reproduced
incorrect classification of invalid evidence: a negative-peak excursion
was accepted as a required spike, and a nonfinite human interval was
treated as a rejected prediction. Both now invalidate the comparison.
Future direct-event audits must validate event identity and reference
values before applying acceptance conditions.

The audit is implementation evidence. The completed
[calcium experiment](h01-l2-calcium-removal-result.md) supplies the
separate mechanism result; it does not establish human cell validation.
