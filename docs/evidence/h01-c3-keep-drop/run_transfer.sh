#!/usr/bin/env bash
# Receipted launcher for docs/evidence/h01_anatomy_transfer_run.py in the C3 keep-test worktree on the Vast box.
# Same receipts as var/h01-driven/run_transfer.sh (launch.json, run.log, run.err, time.txt, terminal.json), but the
# checkout is the campaign worktree and the receipts live under var/c3-keep/, never under /workspace/braintrace.
# usage: run_transfer.sh LABEL WALL_CAP_S -- <runner args...>   (output is fixed to $OUT/run)
set -u
LABEL="$1"; CAP="$2"; shift 2; [ "$1" = "--" ] && shift
ROOT=${ROOT:-/workspace/braintrace-c3-keep}
RUNS=${RUNS:-var/c3-keep}
case "$ROOT" in /workspace/braintrace|/workspace/braintrace/) echo "refusing to run in the main checkout $ROOT" >&2; exit 2;; esac
OUT=$ROOT/$RUNS/$LABEL
mkdir -p "$OUT"
cd "$ROOT"
PY=${PY:-/workspace/venv314/bin/python3}
COMMIT=$(git rev-parse HEAD)
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
printf '{"label":"%s","status":"running","started_utc":"%s","source_commit":"%s","checkout":"%s","wall_cap_seconds":%s,"runner_sha256":"%s","cpuset":"%s","memfrac":"%s","arguments":%s}\n' \
  "$LABEL" "$START" "$COMMIT" "$ROOT" "$CAP" "$(sha256sum docs/evidence/h01_anatomy_transfer_run.py | cut -c1-64)" \
  "${CPUSET:-200-231}" "${MEMFRAC:-.3}" \
  "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")" > "$OUT/launch.json"
timeout -k 30 "$CAP" /usr/bin/time -v -o "$OUT/time.txt" \
  env PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-.3} XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8 \
  taskset -c ${CPUSET:-200-231} $PY -u docs/evidence/h01_anatomy_transfer_run.py "$@" --output "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"
RC=$?
T1=$(date +%s.%N)
SECS=$($PY -c "print($T1-$T0)")
PEAK=$(grep 'Maximum resident' "$OUT/time.txt" | awk '{print $NF}')
printf '{"label":"%s","exit_code":%s,"wall_seconds":%s,"peak_rss_kib":%s,"finished_utc":"%s","timed_out":%s}\n' \
  "$LABEL" "$RC" "$SECS" "${PEAK:-null}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$([ $RC -eq 124 ] && echo true || echo false)" > "$OUT/terminal.json"
echo "DONE $LABEL rc=$RC secs=$SECS" >> "$OUT/run.log"
