# Example 21 sparse checkpoint evaluation

Status: approved
Date: 2026-08-20

## Goal

Complete long-horizon shared-model evaluation without retaining one dense model
output and physical-state snapshot for every latent tick.

## Contract

- With evaluation controls disabled, the device arm gathers only the configured
  scoring checkpoints.
- A 390-tick run gathers exactly ``0, 30, 60, ..., 390``: 14 snapshots rather
  than 391.
- Exact ARC scoring and the primary submission checkpoint remain unchanged.
- The lean result reports checkpoint-level physical diagnostics; it does not
  claim dense per-tick trajectories.
- Opt-in evaluation controls retain the existing dense trajectory contract.
- Training settings, data, model dimensions, and optimizer schedule are not
  changed.

## Acceptance

- A regression proves the lean selected-index array equals the configured
  checkpoint schedule.
- Existing 60-tick scoring remains compatible.
- The 1024-neuron, 1024-edge, batch-32, 13-update, 390-tick GPU run completes
  and emits scores for all 14 checkpoints.
