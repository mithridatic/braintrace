# BRA-9 Gate 5 remediation result

## Disposition

Commit `9617a0a2c571cace2c82e20cf9ae549f220c64a7` repairs the reviewer-identified
implementation and evidence defects. The measured Gate 5 result is still not
eligible for review approval. The accepted parent and the repaired neuron-add
candidate both pass zero of 12 strict checks. The candidate therefore has no
strict gain and is not promoted. Pruning remains disabled because every parent
validation strict Boolean is false.

The compact measured record is
[example21-structural-remediation.json](example21-structural-remediation.json).
The full 4.0 MiB diagnostic artifact has SHA-256
`065647708d58a9d79dc1a32ca1b43796e3a95b2db56a7622373acab8c01c5aa3`.
It is not installed as the Gate 5 artifact because `promoted` is false.

## Reproducible baseline

- Model/task: `BrainCellArcModel` on the fixed Example 21 ARC screen: eight
  training tasks and four validation tasks.
- Learning-quality metric: strict task pass at 1. Before: 0/12. After: 0/12.
- Adaptation-speed metric: one 64-update neuron-add arm completed in
  72.876 seconds, below the 300-second bound.
- Structural action: 103 neurons added, from 2,048 to 2,151; recurrent sparse
  items became 18,032. No dense neuron-pair array was created.
- Optimizer: parent file SHA-256
  `f5a7f40ec99784a1f7b76e1ab748e4110ad388c9e1eb344493b14d2eecf33e9d`;
  nonzero input, recurrent, and readout state at step 64; surviving values and
  step counts preserved; new values zero; active Muon state loaded and remapped.
- Runtime: `braintrace-example21:b75b834`, Python 3.14.0, JAX CPU,
  Linux/WSL2 x86-64. Seeds: 21, 22, 23.
- Peak memory: 1,024,770,048 bytes from the container cgroup-v2
  `memory.peak` counter. This is an upper bound that includes process startup
  and the in-container dependency check.

Command, with `$scratch_dir` set to the run-owned evidence directory:

```sh
docker run --rm -e JAX_PLATFORMS=cpu -e XDG_CACHE_HOME=/work/.cache \
  -e PYTHONPATH=/work -v "$PWD:/work" -w /work \
  braintrace-example21:b75b834 sh -lc \
  "python -m pip install -q --no-cache-dir braincell==0.1.0 && \
   python examples/pp_prop/example21_structural.py neuron-add \
     --data-root /datasets/arc/raw \
     --parent-checkpoint /work/$scratch_dir/parent.npz \
     --output /work/$scratch_dir/neuron-add-9617a0a.json; \
   cat /sys/fs/cgroup/memory.peak"
```

## Verification

- `89 passed, 10 warnings in 99.78s`: structural, ARC contract, and compiled
  Example 21 suites together.
- `55 passed in 15.05s`: final structural and ARC contract suites after the
  checkpoint-file digest correction.
- Focused coverage: 982 statements, 65 missed; 346 branches, 47 partial;
  91% combined statement-plus-branch coverage.

Coverage command:

```sh
coverage run --rcfile=/dev/null --branch \
  --include='examples/pp_prop/example21_structural.py' \
  -m pytest -q examples/pp_prop/example21_structural_test.py \
  examples/pp_prop/arc_contracts_test.py
coverage report --rcfile=/dev/null --show-missing
```

## Prioritized risk register

1. Critical: the parent has no validation strict pass. OpenSpec correctly
   blocks both pruning arms, and the measured addition does not create a strict
   gain. Gate 5 cannot pass without a higher-quality accepted parent or an
   approved change to the promotion contract.
2. High: the compiler reports that direct readout weight and bias are outside
   the ETP-compiled model. The implementation supplies explicit target-dependent
   gradients, but this split remains an integration risk.
3. Medium: the peak-memory figure is a container upper bound, not isolated
   model allocations. Use an image with GNU `time` or an external cgroup sampler
   if isolated process peak RSS becomes a release criterion.

Suggested next tests after the product decision are a neuron-add arm from a
parent with at least one validation strict pass, followed by each pruning arm
and connection-add from the newly accepted checkpoint. The merged artifact
must still reject every no-gain candidate.
