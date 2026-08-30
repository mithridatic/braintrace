# Example 21 public API validation

## Scope

The Example 21 BrainCell and structural adaptation modules expose public
functions, classes, and properties. Every public API must use a NumPy-style
docstring with a summary, parameter contracts, and a return contract when it
returns a value. Private helpers are outside this requirement.

User-facing validation errors must use sentence case and state the corrective
value or action. A message that only reports the failed condition is not
enough.

Tracked shell wrappers used by validation tooling must be invoked through
`bash` and remain non-executable in a review checkout. The Git tree and the
materialized file must both preserve mode `0644`.

## Verification

Co-located tests must inspect every public API in both modules. A public
callable with arguments must declare every argument under `Parameters`. A
value-returning callable must declare `Returns`; a generator must declare
`Yields`; a public class must declare constructor `Parameters` or public
`Attributes`. The audit must reject summary-only docstrings.

The audit must inspect every return statement in each public callable,
including returns nested in conditionals and other control-flow blocks, while
not treating returns in nested callable definitions as returns of the outer
callable. Every literal or formatted message passed to a raised user-facing
exception must start with an uppercase letter, contain one sentence-case
failure statement, and name a corrective value or action after a semicolon.
Representative validation errors must also be exercised at runtime.
