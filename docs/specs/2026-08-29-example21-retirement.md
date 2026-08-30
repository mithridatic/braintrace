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
The `proof` subcommand SHALL load the raw practice task `d631b094`, encode its
supervised query, construct `BrainCellArcModel` and `PPPropEpisodeTrainer`, and
apply exactly eight real PP-Prop updates through `run_fixed_schedule`. It SHALL
also run a forward-only validation episode for `46f33fce` after training, and
report finite optimizer state, changed recurrent weights, changed direct
prediction, and validation parameter isolation. A proof SHALL fail when the
raw task files are unavailable, when a synthetic schedule or probe replaces
the model/trainer, or when any required observation is false.

The `run` subcommand SHALL load the eight declared practice training tasks and
apply exactly 64 real PP-Prop updates in the fixed order. It SHALL report the
direct strict result for the training and validation tasks without reading
target values as event inputs. Both modes SHALL accept `--arc-root`,
`--device`, and `--output-dir` options, while refusing unsupported device
values and proof or run schedule changes before model execution.

The proof training objective SHALL remain outside the inference event vectors.
Each counted episode SHALL carry fixed-size target supervision metadata with a
request kind and a one-row validity mask. The compiled learner step SHALL
produce a scalar shape loss at the shape request and a scalar row loss at each
valid row request. The proof SHALL pass only the 31 shape and row request
positions to that objective. Direct gradients for the readout weights and bias
SHALL be computed from the same target-aware objective and included in every
counted update. Mutating a held-out target SHALL change the supervised loss or
its direct readout gradient when the corresponding logits differ.

Each counted training episode SHALL also carry its Boolean `advance_mask` into
the trainer call. The trainer SHALL reset biological and eligibility state once
before each counted episode, while retaining trainable parameters and optimizer
state. The compiled step SHALL receive the Boolean advance value and SHALL use
the false branch to return zero loss and zero readout features without calling
the learner. False-advance positions SHALL contribute neither loss nor
gradient, including when a malformed request mask marks one as valid.

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
