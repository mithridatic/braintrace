# Gate 4 rerun under the strict-aligned step loss — 2026-08-29

Spec: `docs/specs/2026-08-29-example21-strict-aligned-step-loss.md`.

Image: `braintrace-example21:b75b834` (runtime `pip install braincell==0.1.0`);
a layer with braincell baked in was built as `braintrace-example21:braincell-0.1.0`.

Command: `docker run --rm --gpus all -v ${PWD}:/opt/braintrace -w /opt/braintrace -e PYTHONPATH=/opt/braintrace braintrace-example21:b75b834 python examples/pp_prop/example21_gate4.py --data-root /datasets/arc/raw/data/training --output /opt/braintrace/var/gate4-rerun/reviewer-gate4-result.json`

## What changed versus the 2026-08-25/27 evidence

- PP-Prop now differentiates `request_step_loss(readout(voltage_t), target_t)`
  at the 31 request events with the query target as the second `etrace_grad`
  sequence. Previously the objective was `sum(voltage)` with no target.
- `readout_weight` and `readout_bias` are in the differentiated weight set.
  Previously they were never updated.
- Predictions are decoded from the 31 request readouts (events 674–704).
  Previously the final state's single `(360,)` readout was replicated across
  rows, so every decoded grid had constant-colour rows. The decoder-timing
  helper also read event 673 (input end) instead of 674 (shape request).

## Measured result (all Gate 4 checks pass)

- Total proof time 118.5 s (< 180 s). GPU selected: GPU median 3604.6 ms vs
  CPU median 6007.0 ms per gradient episode; prediction bytes identical.
- Decoder: 31 requests × 5 warmed calls, max 0.04 ms (< 100 ms).
- Eight updates on `d631b094`; `46f33fce` forward-only, state unchanged.
- Masked training loss per update: 8.789, 8.634, 8.631, 8.635, 8.630, 8.636,
  8.630, 8.636 (shape CE + one valid row CE; chance is 6.80 + 2.30 = 9.10).
- Reported components: shape 6.785 → 6.695, rows 2.189 → 2.183.
- Readout weight movement 0.00641; recurrent weight movement 0.01664.
- Direct accuracy, target 1×5: pre shape wrong (16×16), 0/5 cells; post shape
  wrong (16×16), 0/5 cells. Strict 0/1. The loss falls on the first update
  and then stalls; eight updates at the declared rates do not move the shape
  argmax.
- All six interventions recorded; null unchanged, five state interventions
  change the prediction.

Structured evidence: `2026-08-29-arc-loss-gate4-evidence.json`, SHA-256 in
`2026-08-29-arc-loss-gate4-evidence.sha256`.
