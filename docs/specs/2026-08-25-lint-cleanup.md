# Lint cleanup

## Scope

The shipped `braintrace` package is the lint target. Tests, examples, and
notebooks are checked separately because they intentionally contain executable
snippets and import-path fixtures.

## Requirements

- Ruff reports no findings for shipped package modules.
- basedpyright reports no new findings for changed modules.
- Existing suppression comments are removed when the underlying code or type
  can be expressed without a suppression.
- Public behavior and the legacy lazy-import API remain unchanged.

## Verification

Run `ruff check braintrace --exclude '*_test.py'` and basedpyright on changed
modules. Run focused tests for any behavior changed during cleanup.
