# Example 21 evidence archive

The six `20260821` bundles under `var/` are immutable historical schema-1
evidence from revision `353bd46`. They used the legacy effort/confounded
protocol and are replay evidence, not protocol-v2 qualification artifacts.
Their exact bytes are historical replay evidence. They are not active Example
21 inputs or qualification results.

The active implementation is
`examples/pp_prop/21-braincell-arc.py`. It reads named raw ARC files directly
and uses the compiled BrainCell and BrainTrace path described in
`docs/specs/2026-08-29-example21-retirement.md`.

The bounded smoke check uses:

```text
python /opt/braintrace/examples/pp_prop/21-braincell-arc.py --smoke --device cpu
```

The image keeps raw ARC files at `/datasets/arc/raw`. No generated index or
source manifest is required. Historical bundles remain immutable archive data;
they do not define the active command or result contract.
