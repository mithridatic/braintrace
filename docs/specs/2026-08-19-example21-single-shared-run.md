# Example 21 — single shared online run

Status: approved
Date: 2026-08-19
Branch: `feat/example21-row-refinement`

## Goal

Run Example 21 as one shared pp-prop learning process over the training-task
stream, followed by one frozen evaluation of the learned model.

## Contract

- There is no separate pretraining phase or task-local adaptation phase.
- One model parameter set and one optimizer state persist across training
  episodes and task boundaries.
- Only dynamic model state and eligibility-trace state reset at an independent
  episode boundary.
- Leave-one-demonstration-out adaptation is disabled by default.
- The repeat, no-context, shuffled-demonstration, and slot-ablation arms are
  disabled by default; they remain opt-in diagnostics.
- The primary evaluation is the intact frozen model on the evaluation split.
- Scoring checkpoints are every 30 latent ticks through ``latent_steps``;
  ``latent_steps`` is the primary submission checkpoint. A 300-tick run scores
  ``0, 30, 60, ..., 300`` and trains across ``30, 60, ..., 300``.
- Progress is emitted to stderr at stage boundaries and after every training
  chunk, including completed/total work, elapsed seconds, and an ETA when one
  can be calculated.

## Acceptance

- The default configuration reports the shared frozen evaluation mode.
- A default run does not call the task-local adaptation runner.
- A default run executes one intact evaluation arm.
- A disabled-controls run does not execute or time ``repeat_intact``.
- ``--training-chunk-size 1`` reports each optimizer update independently.
- Existing diagnostic arms remain available when explicitly enabled.

## Canonical GPU run

Run the source-mounted CUDA image with ``PYTHONPATH=/work``, the ARC corpus
mounted read-only at ``/datasets/arc``, and these experiment arguments:

```text
--device gpu --source-manifest var/example21-arc-v1.0.2-sources.json
--neurons 1024 --recurrent-edges 1024 --max-demonstrations 10
--latent-steps 300 --training-updates 13 --training-batch-size 32
--training-chunk-size 1
```

Do not pass ``--task-local-adaptation`` or ``--evaluation-controls`` for the
single shared run.

The tracked GPU image installs ``msgspec`` directly and records the source
revision used for the build. Build it from the repository root.
