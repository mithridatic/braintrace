# Simplified Technical English language pass

## Purpose

Make natural-language comments, docstrings, and user-facing error messages
clear, concise, and consistent with sentence case and ASD-STE100 principles.

## Scope

- Rewrite natural-language Python comments and docstrings.
- Rewrite user-facing exception, assertion, warning, and diagnostic text.
- Use sentence case for prose and headings.
- Use short, direct sentences and active voice.
- Add a specific remediation sentence to each user-facing error message.
- Preserve code, identifiers, public API names, legal notices, protocol text,
  quoted external text, and machine-readable values.

## Acceptance criteria

- Python files compile without syntax errors.
- Existing behavior and public interfaces do not change.
- User-facing errors state the cause and the action that resolves it.
- Focused tests for diagnostics, compilation, examples, and repository
  conventions pass.
- The final diff contains only the language pass and this specification.
