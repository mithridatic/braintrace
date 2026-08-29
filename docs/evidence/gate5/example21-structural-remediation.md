# BRA-9 Gate 5 bounded structural evidence

## Disposition

Implementation commit `67e440306d7f1e0329305e01b02302bc60860e90`
completes OpenSpec tasks 7.1 through 7.6 under the board clarification that a
strict false-to-true change is the promotion condition, not the candidate
completion condition. The merged artifact is
`docs/evidence/gate5/example21-structural-arm.json`, SHA-256
`a17e196c44842da4117323b95c6e043a691ba4a9b4013ee7da391e764ba31b77`.

The parent and every candidate have the direct strict vector 0/12. Both
addition candidates completed and caused no strict regression, but neither is
promoted. Both pruning arms are blocked by design because all four parent
validation Booleans are false. Every non-promoted run left parent checkpoint
`87bbf8bb901cce36dcb10a98e12007cc9c105660510c8fa8afd24058ecf5f27e`
byte-identical.

This rerun closes the Gate 5 review findings. Spike activity is the direct
per-event `model.previous_spikes` state, never a voltage threshold. Both
addition paths reject a candidate above the `1,024 * neuron_count` biological-
connection ceiling before candidate construction or compilation. The merge
gate validates each arm's finite complete-process time, not only its internal
operation timer.

## Reproducible baseline

- Model and task: `BrainCellArcModel`, 2,048 Hodgkin-Huxley neurons, 14,112
  sparse input edges, 16,384 sparse recurrent edges, and a 737,280-value
  voltage readout on eight fixed training and four fixed validation ARC
  practice tasks.
- Learning-quality metric: ordered direct `strict_task_pass_at_1` vector for
  all 12 tasks. Parent and all candidate vectors are 0/12.
- Adaptation-speed metric: each addition performs exactly 64 compiled PP-Prop
  updates. No false-to-true transition occurred.
- Environment: `braintrace-example21:b75b834`, Python 3.14.0, JAX CPU,
  Linux/WSL2 x86-64, seeds 21, 22, and 23.
- Parent: 64 optimizer steps for input, recurrent, and readout groups, with
  nonzero state. Regeneration from the implementation produced exact SHA-256
  `87bbf8bb…` and 0/12 strict before and after.

## Measured arms

| Arm | Outcome | Change | Updates | Complete time | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| neuron-prune | blocked by design | 0 | 0 | 22.849 s | 1,314,627,584 B |
| connection-prune | blocked by design | 0 | 0 | 137.568 s | 1,317,982,208 B |
| neuron-add | non-promoted | 103 neurons | 64 | 90.104 s | 1,464,385,536 B |
| connection-add | non-promoted | 820 edges | 64 | 93.716 s | 1,374,351,360 B |

The addition arms preserve all surviving optimizer values and step counts,
initialize new values to zero, and preserve the nonzero active Muon state. The
connection selector scanned 2 of 64 tiles, held at most 65,536 pairs, stopped
on its proven bound, and did not create a dense neuron-pair array. Each arm has
a distinct recorded process identity comprising PID namespace, PID, and process
start ticks.

Compaction identity cannot execute on this parent because pruning is fail-closed.
The focused suite verifies byte-identical mask-versus-compaction predictions,
strict-vector identity, preserved Adam/Muon leaves, reset eligibility, and the
exact sparse endpoint remap on hand-calculated fixtures.

## Verification

Focused command:

```sh
docker run --rm -e JAX_PLATFORMS=cpu -e XDG_CACHE_HOME=/work/.cache \
  -e PYTHONPATH=/work -v "$PWD:/work" -w /work \
  braintrace-example21:b75b834 sh -lc \
  'python -m pip install -q --no-cache-dir braincell==0.1.0 && \
   python -m coverage erase && \
   python -m coverage run --rcfile=/dev/null --branch \
     --include=examples/pp_prop/example21_structural.py -m pytest \
     examples/pp_prop/example21_structural_test.py \
     examples/pp_prop/arc_contracts_test.py -q && \
   python -m coverage report --rcfile=/dev/null -m \
     examples/pp_prop/example21_structural.py'
```

Result: 59 passed in 15.77 seconds; 90.8322% combined line-plus-branch
coverage. The merged-artifact command then passed its strict schema and sparse,
complete-process time, optimizer, process-isolation, and promotion-consistency
gates. Focused regressions separately prove that both growth paths enforce the
biological-connection ceiling before allocating candidate arrays.

## Prioritized risk register

1. **High — parent learning quality:** the only parent is 0/12 strict. The
   structural mechanisms execute correctly, but this evidence does not show a
   learning-quality improvement and promotes no candidate.
2. **High — readout integration boundary:** compilation warns that readout
   weight and bias are outside the ETP-compiled model. Direct target-dependent
   gradients cover this boundary, but it remains an integration risk.
3. **Medium — memory headroom:** the largest measured peak process RSS is
   1.464 GB on CPU. GPU and concurrent-run headroom are not established.
4. **Medium — compilation variance:** connection-prune took 137.568 seconds
   despite performing no mutation. It remains below 300 seconds but has less
   margin than the other arms.
5. **Low — process namespace reuse:** sequential containers can reuse PIDs and
   namespace identifiers. The artifact also records process-start ticks and the
   merge gate requires each full identity tuple to be distinct.
