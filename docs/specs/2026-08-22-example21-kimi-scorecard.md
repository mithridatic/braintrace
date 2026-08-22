# Example 21 Kimi transfer scorecard

## Result

No tested mechanism beat the accepted baseline under the preregistered
promotion gates. The accepted defaults are unchanged. Exact-task accuracy was
zero in every completed ARC pilot and full reduced-topology run.

This record is an experimental scorecard, not a claim of paper reproduction.
Scores are from protocol v2, 4,096 neurons, 4,096 recurrent edges, memory width
32, 60 latent ticks, 260 updates, batch 32, and seed 2108 unless noted.
The 100-task pilots evaluated 104 queries. Full runs evaluated 419 queries.

## Accepted linear baseline

Configuration: Muon, weight decay 0.1, cosine schedule, no warmup, softcaps
4/25, learned additive memory update, linear read, interval 1.

| run | R0 pixel | R30 pixel | R60 pixel | exact | runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100-task pilot, seed 2108 | 0.004712 | 0.004712 | 0.004456 | 0 | 196.34 |
| full, seed 2108 | 0.012294 | 0.005609 | 0.005097 | 0 | 286.26 |
| full, seed 31337 | 0.014029 | 0.005969 | 0.004911 | 0 | 222.09 |
| full, seed 7777 | 0.013963 | 0.006877 | 0.005877 | 0 | 226.45 |

The baseline itself has negative effort scaling: R60 is below R30 in all
three full runs and in the pilot.

## Stage 0: optimizer, schedule, and softcap pilots

All entries below are 100-task pilots at seed 2108. `Effort` is R60 minus R30
pixel accuracy. `Pairing` is intact minus shuffled-demonstration pixel
accuracy where retained in the pilot diagnostics.

| arm | R60 pixel | effort | pairing | runtime (s) | result |
| --- | ---: | ---: | ---: | ---: | --- |
| Muon decay 0.01 | 0.004003 | -0.000433 | +0.000028 | 184.88 | rejected |
| AdamW decay 0.01 | 0.018652 | -0.006198 | 0.000000 | 192.34 | rejected; pixel-only gain |
| cosine + 1% warmup | 0.004456 | -0.000256 | +0.000267 | 184.94 | rejected |
| constant rate | 0.006316 | -0.003967 | +0.000122 | 205.70 | rejected |
| softcap 1/25 | 0.004295 | -0.000325 | -0.000176 | 196.36 | rejected |
| softcap 4/1 | 0.004347 | -0.000395 | 0.000000 | 235.94 | rejected |
| softcap 1/1 | 0.004556 | -0.001140 | +0.000035 | 300.45 | rejected; runtime gate |

AdamW produced the largest isolated pixel increase (+0.014196 versus the
baseline pilot), but it did not improve pairing or effort scaling and was not
promoted.

## Stages 1–4 mechanism pilots

| stage / arm | R0 pixel | R30 pixel | R60 pixel | exact | result |
| --- | ---: | ---: | ---: | ---: | --- |
| 1, gated read | 0.008786 | 0.004068 | 0.003600 | 0 | rejected |
| 1, gated RMS read | 0.016956 | 0.015654 | 0.012451 | 0 | rejected; shuffled R60 0.015000, negative pairing |
| 2, interval 4 | 0.006318 | 0.007576 | 0.006617 | 0 | rejected; R60 below R30 |
| 2, interval 8 | 0.006185 | 0.011102 | 0.007048 | 0 | rejected; R60 below R30 |
| 3, latent Attention Residual | 0.009319 | 0.004712 | 0.004189 | 0 | rejected |
| 4, progressive effort | 0.009560 | 0.003459 | 0.003326 | 0 | rejected |

The progressive schedule metadata was correct: 160 updates sampled effort 0,
72 sampled effort 0/30, and 28 sampled effort 0/30/60. It still did not fix
R30-over-R60 behavior.

## Stage 5 and Stage 6 status

- `delta_write`: no score. The 100-task evaluation failed closed because the
  decoder received non-finite height logits. The retained traceback is
  `ValueError: height logits contains non-finite values`. This arm is not a
  promotion candidate.
- `situ_glu_update`: not run after the delta-write non-finite failure and the
  lack of a positive preceding pilot.
- effort self-distillation: skipped. The teacher gate was not satisfied because
  deeper effort was not better than R30 in the baseline evidence.

## Promotion decision

Promotion result: **none**. No defaults changed. No exact ARC gain was measured.
The implementation, operator tests, configuration round-trips, checkpoint
compatibility, and regression suites remain in the feature branch, but the
experimental gain from these stages is currently zero.

The full-run logs also contain a repeatable CUDA allocator warning during
evaluation. The completed baseline artifacts remain finite; the warning is
recorded as infrastructure evidence and is not counted as a model gain.
