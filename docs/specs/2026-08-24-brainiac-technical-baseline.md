# Brainiac technical baseline and risk register

Status: complete development baseline; review required before the proposed experiment

Issue: BRA-4

Date: 2026-08-24

Branch: `paperclip/bra-4-baseline`

Source revision: `0a101278c7839cd25f59c303d1510bc75fdf3f79` (clean at measurement start)

## Success metrics and decision rule

This baseline separates three questions:

1. **Learning quality:** validation ensemble accuracy and negative log likelihood (NLL) after 200 optimizer updates.
2. **Adaptation speed:** updates and sample-ticks needed to retain 0.80 validation accuracy. A run ending below 0.80 is right-censored as `>200 updates` and `>96,000 sample-ticks`; it is not assigned a fabricated time-to-threshold.
3. **Engineering performance:** algorithm elapsed time, end-to-end process wall time, logged cold-process compilation time, and sampled peak process working set.

The online-learning result is interpretable only if the matched full-window BPTT arm learns the task. BPTT is the task-validity/correctness oracle here, while a frozen-random-recurrence arm tests whether recurrent plasticity adds value. This end-to-end experiment does not establish element-wise gradient correctness.

## Representative task

The task is Example 17's balanced delayed-cue recall at the medium horizon. Each trial presents one of two disjoint eight-channel cues for four 1 ms steps, then 22 silent steps, then four response steps containing only a label-independent go cue. The 17-input sparse recurrent spiking network must retain the cue and classify it after the delay. Only the response window is supervised.

This is a compact rapid-adaptation test because the model must improve from streaming labeled episodes under a fixed 200-update budget while solving temporal credit assignment across a cue-free delay. It is more diagnostic of recurrent learning than a static mapping, but it is not an ARC, continual-learning, or lifelike-behavior benchmark.

Profile: 24 LIF neurons, fixed out-degree 4 sparse recurrence, batch 16, 30 steps per trial, 200 updates, 3,200 training trials, 96,000 sample-ticks, 1,024/256 train/validation trials, eight deterministic validation encodings, CPU, no curriculum, no sealed test, and no gradient-evidence add-on.

## Results

| Arm | Initial accuracy | Final accuracy | Accuracy delta | Initial NLL | Final NLL | Retained 0.80 speed | Algorithm elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online PP-prop | 0.5000 | 0.4961 | -0.0039 | 0.694327 | 0.696480 | >200 updates / >96,000 ticks | 16.857 s |
| Full-window BPTT oracle | 0.5000 | 0.4883 | -0.0117 | 0.694327 | 0.693144 | >200 updates / >96,000 ticks | 14.014 s |
| Frozen random recurrence | 0.5000 | 0.4961 | -0.0039 | 0.694327 | 0.696215 | >200 updates / >96,000 ticks | 12.261 s |

The PP-prop arm did not improve validation quality and did not separate from the frozen recurrence control. The BPTT arm also remained at chance, so this reduced profile fails its prerequisite as an algorithm evaluation. It establishes a reproducible engineering/measurement baseline and an evaluation gap, not a negative claim about PP-prop.

The runs remained dynamically finite: PP-prop mean firing was 0.0414 spikes per neuron-step, with zero silent-neuron and saturated-neuron fractions. That rules out gross silent/saturated collapse in this one run but does not rescue the failed learning gate.

### Engineering measurements

The externally monitored fresh-process repeat measured:

| Arm | Process wall time | Sampled peak working set |
|---|---:|---:|
| Online PP-prop | 18.529 s | 1,087,328,256 bytes (1,036.96 MiB) |
| Full-window BPTT | 18.127 s | 1,078,513,664 bytes (1,028.55 MiB) |

For PP-prop, a cold-process JAX compile log contained 38 trace events and 34 MLIR/XLA events. Summed logged durations were 1.140 s tracing, 0.570 s jaxpr-to-MLIR, and 2.781 s XLA compilation: 4.490 s total logged pipeline time. This is a sum of individual compiler events, not one fused-program compile latency.

On this small CPU profile PP-prop showed no efficiency advantage over BPTT: its algorithm elapsed time was 20.3% longer, and its sampled process peak was 0.8% higher. Peak working set includes Python, JAX, compilation/runtime state, and evaluation arrays; it is not persistent model-state memory and must not be extrapolated to brain scale.

Evidence:

- [Online PP-prop raw result](../evidence/bra-4/pp-prop.json)
- [BPTT raw result](../evidence/bra-4/bptt.json)
- [Frozen-recurrence raw result](../evidence/bra-4/frozen-random-recurrence.json)
- [Externally monitored PP-prop repeat](../evidence/bra-4/pp-prop-profile.json)
- [Externally monitored BPTT repeat](../evidence/bra-4/bptt-profile.json)
- [Compile log](../evidence/bra-4/pp-prop-compile.log)
- [External measurement summary and limits](../evidence/bra-4/measurement.json)
- [Task and acceptance specification](2026-08-10-temporal-credit-benchmark.md)

## Environment and reproducibility

- Windows 11 `10.0.26200`, Python 3.14.6.
- JAX/JAXlib 0.11.1, CPU backend, `CpuDevice(id=0)`.
- BrainState 0.5.4, BrainEvent 0.2.1, BrainTrace 0.2.5.
- No CUDA, NVIDIA driver, or container image applies to this CPU development run.
- Manifest master seed: `20260810`.
- Bundle: `split0-topology0-weight0`.
- Split seed `4240395627`; topology seed `475518980`; weight seed `2803593026`; training-order seed `3244608299`; training-encoding seed `1534222160`.
- Evaluation-encoding seeds: `404957872`, `3631603620`, `3194688560`, `575489032`, `2077102468`, `1904045692`, `750849854`, `4197398880`.
- Test commitment: `ba996aa8b641eb68339898bcd8936624d69d55a3eac7fad7d4611fcbe9b73cde`; sealed test data were not opened.

From the repository root, set `PYTHONPATH=.` and run each arm by substituting `ARM` and `FILE`:

```powershell
$env:PYTHONPATH='.'
python examples/pp_prop/17-temporal-credit-benchmark.py `
  --arm ARM --horizon medium --neurons 24 --degree 4 `
  --batch-size 16 --updates 200 --device cpu `
  --no-curriculum --no-gradient-evidence --no-sealed-test `
  --json-output FILE
```

The three pairs are `all_pp_prop` / `pp-prop.json`, `all_bptt` / `bptt.json`, and `frozen_random_recurrence` / `frozen-random-recurrence.json`. Set `JAX_LOG_COMPILES=1` for the monitored PP-prop repeat. Sample `WorkingSet64` every 100 ms around a hidden `Start-Process`; the exact resulting measurements and limitations are recorded in `measurement.json`.

## Prioritized mission risks

1. **Rapid learning / evaluation validity — critical.** No arm, including BPTT, learned the reduced task within 200 updates. Until the oracle learns, the profile cannot distinguish an online-rule defect from an underpowered or mis-tuned task configuration. Single-seed, endpoint-only evaluation also cannot estimate variance, threshold crossing, or forgetting.
2. **Efficiency — high.** A 24-neuron CPU run consumed about 1.04 GiB peak process working set, and PP-prop was neither faster nor smaller than BPTT here. The measurement includes fixed runtime overhead, but the current baseline offers no observed efficiency win to extrapolate.
3. **Lifelike behavior — high.** The task is balanced, episodic, binary, and resets state between trials. It exercises spiking temporal memory but not continual adaptation, nonstationarity, embodied sensory structure, multi-timescale memory, or retention under interference.

The failed BPTT task-validity gate is an invalid-evaluation risk and must be escalated before architecture conclusions or mission claims are made.

## Smallest next experiment — proposed, not run

Run only the matched BPTT oracle on the same bundle, model, medium horizon, CPU, and learning rates with the specification's 800-update budget. Add validation checkpoints every 50 updates and retain the existing 0.80 gate. This changes one variable (budget), costs one arm, and answers whether the current profile can become a valid learning benchmark.

- If BPTT reaches and retains 0.80, review the trace and then run matched PP-prop and frozen-recurrence arms on the three declared development bundles.
- If BPTT remains below 0.80, stop: the reduced profile is invalid for comparing learning rules, and any tuning or task change requires a separately reviewed proposal.

No product code, model architecture, test, or autonomous run was added in BRA-4. The next experiment has not been implemented or executed.
