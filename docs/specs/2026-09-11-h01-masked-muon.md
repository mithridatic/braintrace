# H01 masked Muon

Status: implemented and locally verified 2026-09-11; isolated worktree branch.

H01 input and recurrent edge vectors must receive actual Muon updates. Keep the
forward model sparse. Scatter encoder gradients to (1,441,N); recurrent contacts
to (K,N,N), with independent Muon per plane. Stable synapse identities determine
initial parallel-contact slots in sorted order per cell pair. Persist slots;
survivors retain them, new identities take the lowest free slot per pair. Never
sum contacts. Gather only existing edges after orthogonalization. This is masked
Muon, an explicit modeling convention, not a validated sparse convergence claim.

Use Optax scale_by_muon with momentum .95, Nesterov, five steps, epsilon 1e-8,
coefficients (3.4445,-4.7750,2.0315), Frobenius preconditioning, no adaptive scaling,
and default width scaling sqrt(max(1,outputs/inputs)). Apply decoupled decay .1
and retain rates input .001, recurrent .0003, readout .003. Readout matrix uses
ordinary Muon; bias retains Nesterov AdamW (.9,.999,1e-8,decay .1). Persist edge-shaped
momentum and counters, not dense buffers. Reconstruct zero momentum off topology.

An explicit trainer adapter is required for H01 probe and sessions. Default
non-H01 trainers retain existing behavior. Version settings, mapping, optimizer
configuration and implementation hashes. Reject legacy optimizer continuation.
Mutation transports only survivor momentum by identity; clone momentum is zero.
Check mappings, state/parameter shapes and optimizer version before execution.

Conservative workspace estimate: sum over groups of dtype bytes times
(12 * dense elements + 6 * plane_count * min(rows,cols)^2), plus dense readout
workspace. Reject estimates over 512 MiB before learner compilation. This is an
estimate, not a peak RSS guarantee; existing process limits still apply.

Verification: first reproduce current vector AdamW fallback; numerical multi-step
dense Optax Muon oracle, dense degeneration, AdamW distinction without decay,
36/240 contacts on 12 cells, duplicate contacts, empty/single groups, zero gradients,
reordering, mapping/resource failures, restore next-update equality and mutation
moment transport. Exercise small synthetic physical cable fixtures. New optimizer
module coverage must exceed 90 percent. No real 12-cell probe or training campaign.

## Verification record

- Regression reproduced: the previous sparse vector update matched Optax's
  Nesterov-enabled AdamW fallback exactly. The replacement matches dense Muon
  with topology masking instead.
- Final combined CPU run: 194 passed, 12 readout-state compiler warnings,
  188.12 seconds. No real-data 12-cell probe or training campaign was run.
- New optimizer statement coverage: 121/121 statements (100%). Coverage required
  `COVERAGE_CORE=pytrace` and an explicit file include after scientific imports;
  the initial pytest-cov module-target run collected no data and was not counted.
- Included h01_muon, h01_arc_probe, h01_session, h01_remap, h01_checkpoint,
  h01_arc_adapter, h01_episode_recovery, 21-braincell-arc, example21_arc_adapter,
  and repository-conventions test modules.
- Verified exact next-update equality after real synthetic-cable checkpoint
  restore; survivor momentum transport; distinct 240-contact parallel edges;
  invalid checkpoint layout rejection before learner compilation.
- Removing unused AdamW state exposed a trainer dependency on its group keys.
  A failing regression was added for compiled gradient mapping and parameter
  write-back; H01 now derives both from explicit parameter groups. The default
  Example 21 path retains its existing behavior.

The code remains a masked-Muon variant. Numerical equivalence to its dense
reference does not establish a training-quality improvement or human physiology.
Legacy H01 optimizer checkpoints require a fresh run, not silent state conversion.
The subsequently authorized Vast 12-cell probe passed, as did 20 GPU optimizer
tests. See ../evidence/h01-12-masked-muon-b2f0fd0.md for exact acceptance boundaries.
