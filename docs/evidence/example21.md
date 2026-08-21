# Example 21 evidence index

The six `20260821` bundles under `var/` are immutable historical schema-1
evidence from revision `353bd46`. They used the legacy effort/confounded
protocol and are replay evidence, not protocol-v2 qualification artifacts.
Their exact bytes are pinned by `example21_evidence_index_test.py`.

Protocol v2 is implemented on branch `fix/example21-paper-audit` with efforts
0/30/60, a fixed 30-row decoder sweep at every effort, learned-update memory,
explicit phase gates, enabled state-hold and recurrent-lesion controls, global
factorized top-two selection, and schema-2 provenance. The implementation
revision is recorded from the live Git checkout in each artifact.

The preregistered reduced-edge diagnostic is generated only from a clean
implementation commit with:

```text
python /opt/braintrace/examples/pp_prop/21-arc-agi-latent-reasoning.py --recurrent-edges 4096
```

It is deliberately nonqualifying for `actual_full_scale`, because the source
default remains 4,194,304 recurrent edges. Its exact, shape, pixel, binding,
control, and resource results are evidence even when the exact score is zero.
