# Repository JSON implementation: `msgspec`

## Decision

All executable Python JSON serialization and deserialization in this
repository uses `msgspec.json`.  The migration includes BrainTrace-adjacent
examples, diagnostics, benchmark runners, and their co-located tests.  JSON
file names and wire formats remain unchanged.

The shared `examples/pp_prop/msgspec_json.py` adapter retains the text-oriented
surface used by the older tooling while delegating encoding and decoding to
`msgspec`.  It also retains deterministic key ordering, pretty-printing, and
the strict `allow_nan=False` contract.  Strict validation is explicit because
`msgspec` otherwise maps non-finite floats to JSON `null`.

The Gate C controls writer continues to enforce its maximum artifact size and
atomic replacement behavior.  Its encoder is now msgspec-backed; the size
check includes the required terminal newline.

## Validation requirements

- No tracked Python file imports the stdlib `json` module.
- Canonical output remains compact, UTF-8 JSON with sorted keys.
- Human-readable artifacts retain their indentation and final newline.
- Malformed input raises `msgspec.DecodeError` through the compatibility
  adapter.
- Strict artifact writers reject NaN and infinity before replacement.
