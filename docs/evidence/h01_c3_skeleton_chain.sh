#!/usr/bin/env bash
# C3-B chain on the Vast box (worktree /workspace/braintrace-c3-skel). Receipts under var/c3-skel/.
# 1. export the 40 candidates (venv-c3) -> .cache/h01/c3-candidates-20260916.zip + provenance
# 2. components inventory, import audit, construct/initialise check (venv314) on the candidates
# 3. the same three on the control archive (kept cell 1684504313 from C3) + the SWC comparison
set -euo pipefail
cd "$(dirname "$0")/../.."
V3=${V3:-/workspace/venv-c3/bin/python}
V14=${V14:-/workspace/venv314/bin/python}
OUT=docs/evidence
CACHE=.cache/h01
R=var/c3-skel
CAND=$OUT/h01-c3-candidates.json
ZIP=$CACHE/c3-candidates-20260916.zip
CTRL=$CACHE/c3-control-1684504313.zip
LABEL=c3_candidates_20260916
mkdir -p "$R"
rm -f "$R/chain.done" "$R/chain.failed"
trap 'echo "$(date -u +%FT%TZ) failed at line $LINENO" > "$R/chain.failed"' ERR
echo "$(date -u +%FT%TZ) start" > "$R/chain.log"

SUPERSEDED_ARGS=()
if [ -n "${SUPERSEDED_SHA:-}" ]; then SUPERSEDED_ARGS=(--superseded-sha256 "$SUPERSEDED_SHA" --superseded-note "${SUPERSEDED_NOTE:-}"); fi
"$V3" $OUT/h01_c3_skeleton_export.py --candidates "$CAND" --output "$ZIP" "${SUPERSEDED_ARGS[@]}" \
  --provenance $OUT/h01-c3-candidates-archive.json --stage "$CACHE/c3-candidates-20260916.stage" > "$R/export.log" 2>&1
SHA=$(python3 -c "import json;print(json.load(open('$OUT/h01-c3-candidates-archive.json'))['archive_sha256'])")
echo "$(date -u +%FT%TZ) exported $SHA" >> "$R/chain.log"

PYTHONPATH=. "$V14" $OUT/h01_population_components.py --archive "$ZIP" --no-network \
  --output $OUT/h01-c3-candidates-components > "$R/components.log" 2>&1
PYTHONPATH=. "$V14" $OUT/h01_population_import_audit.py --archive "$ZIP" --components $OUT/h01-c3-candidates-components.json \
  --output $OUT/h01-c3-candidates-import.json --expected-sha256 "$SHA" --source $LABEL --soma-bearing > "$R/import.log" 2>&1
echo "$(date -u +%FT%TZ) import audit done" >> "$R/chain.log"
rm -f $OUT/h01-c3-candidates-construct.json
PYTHONPATH=. "$V14" $OUT/h01_c3_construct_check.py --archive "$ZIP" --expected-sha256 "$SHA" --source $LABEL \
  --components $OUT/h01-c3-candidates-components.json --candidates "$CAND" \
  --cell-table "$CACHE/c3-segment-properties.json" --output $OUT/h01-c3-candidates-construct.json > "$R/construct.log" 2>&1
echo "$(date -u +%FT%TZ) construct check done" >> "$R/chain.log"

if [ -n "${SKIP_CONTROL:-}" ]; then touch "$R/chain.done"; exit 0; fi
"$V3" $OUT/h01_c3_skeleton_export.py --ids 1684504313 --output "$CTRL" --provenance "$R/c3-control-archive.json" \
  --stage "$CACHE/c3-candidates-20260916.stage" > "$R/control-export.log" 2>&1
CSHA=$(python3 -c "import json;print(json.load(open('$R/c3-control-archive.json'))['archive_sha256'])")
PYTHONPATH=. "$V14" $OUT/h01_population_components.py --archive "$CTRL" --no-network --output "$R/c3-control-components" > "$R/control-components.log" 2>&1
PYTHONPATH=. "$V14" $OUT/h01_population_import_audit.py --archive "$CTRL" --components "$R/c3-control-components.json" \
  --output "$R/c3-control-import.json" --expected-sha256 "$CSHA" --source c3_control_1684504313 --soma-bearing > "$R/control-import.log" 2>&1
PYTHONPATH=. "$V14" $OUT/h01_c3_construct_check.py --archive "$CTRL" --expected-sha256 "$CSHA" --source c3_control_1684504313 \
  --components "$R/c3-control-components.json" --cell-table "$CACHE/c3-segment-properties.json" \
  --polarity E --donor l2-pyramidal-allen-541563728 --output "$R/c3-control-construct.json" > "$R/control-construct.log" 2>&1
PYTHONPATH=. "$V14" $OUT/h01_c3_skeleton_control.py --c3-archive "$CTRL" --output $OUT/h01-c3-control-1684504313.json > "$R/control.log" 2>&1
python3 - "$OUT/h01-c3-control-1684504313.json" "$R" "$CSHA" <<'EOF'
import json, sys
path, receipts, digest = sys.argv[1], sys.argv[2], sys.argv[3]
record = json.load(open(path))
imported = json.load(open(f"{receipts}/c3-control-import.json"))["cells"][0]
built = json.load(open(f"{receipts}/c3-control-construct.json"))["cells"][0]
record["pipeline"] = {"archive": "c3-control-1684504313.zip", "archive_sha256": digest,
                      "import": {k: imported.get(k) for k in ("component", "source_segments", "imported_segments", "missing_segments", "extra_segments", "passed", "seconds")},
                      "construct": {k: built.get(k) for k in ("component", "component_nodes", "n_compartments", "construction_seconds", "init_seconds", "passed", "error")}}
json.dump(record, open(path, "w"), indent=1)
EOF
echo "$(date -u +%FT%TZ) control done" >> "$R/chain.log"
touch "$R/chain.done"
