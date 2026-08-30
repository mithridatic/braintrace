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

The active entry point SHALL expose a real command line interface. `--help`
SHALL print usage and exit without running a fixture. `--smoke` SHALL run the
bounded BrainCell compatibility checks and report a successful smoke result.
The `proof` subcommand SHALL run the bounded eight-update proof schedule and
report its result. Both modes SHALL accept the documented `--device` and
`--output-dir` options, while refusing unsupported device values and proof
schedule changes before model execution.

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
