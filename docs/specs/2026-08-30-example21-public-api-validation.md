# Example 21 public API validation

## Scope

The Example 21 BrainCell and structural adaptation modules expose public
functions, classes, and properties. Every public API must use a NumPy-style
docstring with a summary, parameter contracts, and a return contract when it
returns a value. Private helpers are outside this requirement.

User-facing validation errors must use sentence case and state the corrective
value or action. A message that only reports the failed condition is not
enough.

## Verification

Co-located tests must inspect every public API in both modules and reject
summary-only docstrings. They must also exercise representative validation
errors and reject messages without corrective guidance.
