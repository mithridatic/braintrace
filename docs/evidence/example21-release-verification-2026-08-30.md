# Example 21 release verification

## Environment

The release gates ran on branch `paperclip/BRA-12-retire-the-obsolete-example-21-path`
at commit `4113016`. CPU JAX used the official ARC-AGI-1 JSON files from the
`fchollet/ARC-AGI` repository. The repository-local equivalent root was
`var/release-arc-data`, which is ignored and was passed with `--arc-root`.

The proof task `d631b094` has SHA-256
`8ac92452dbb30cd080144dfe375ecfaa5a1e262c552ebbb27f0b4a88e3792c7f`.
The validation task `46f33fce` has SHA-256
`5e43f62b2cb40f9c954bbab06f7f1a7014a3fb522f16a63a07cb11455a5b518b` in both
the loader-required practice path and the requested evaluation path.

## Release gates

- Real proof: passed in 16.30 seconds; eight updates; finite optimizer state;
  changed recurrent weights; changed direct prediction; unchanged validation
  parameter state; deadline not exceeded.
- Warmed decoder: passed for four executed validation requests; five calls per
  request; every call was below the 100 ms limit; request medians were
  0.007815, 0.007175, 0.007387, and 0.007220 ms.
- Ordinary run: passed with exactly 64 updates; finite optimizer state; and
  unchanged validation parameter state.
- Focused tests: 77 passed on four workers in 40.55 seconds.
- Active BrainCell coverage: 91% branch coverage from 51 co-located tests in
  52.53 seconds.
- Public API tests: 57 passed in 5.28 seconds.
- OpenSpec strict validation: change validation and all validation passed.
- Whitespace and active retired-reference scans passed.

The detailed JSON reports remain in the ignored release-data workspace paths
`var/release-proof-20260830-fixed/example21-proof.json` and
`var/release-run-20260830/example21-run.json`.
