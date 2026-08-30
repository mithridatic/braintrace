# Example 21 retirement

## Scope

Example 21 has one active implementation:
`examples/pp_prop/21-braincell-arc.py` and its co-located test.

The active documentation and image command SHALL use that entry point. The
image SHALL copy raw ARC files to `/datasets/arc/raw` and SHALL NOT build an
index, load a source manifest, or import a retired module.

The old entry points, latent-workspace modules and tests, diagnostic scripts,
and ARC index builder are retired. Historical Git, OpenSpec, and evidence
records MAY mention them, but active commands and imports SHALL NOT.

## Verification

- README commands resolve to `21-braincell-arc.py`.
- The image command resolves to `21-braincell-arc.py` and retains raw ARC
  files.
- Active Python and container files contain no retired import or index,
  manifest, synthetic-task, BPTT, copy, rule, candidate, forest, reranker,
  partial-score, average-score, or large-result execution path.
- The public `braintrace` export surface is unchanged.
- Focused Example 21 tests, repository scans, OpenSpec validation, and
  whitespace checks pass.
