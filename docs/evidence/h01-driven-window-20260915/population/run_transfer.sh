#!/usr/bin/env bash
# Receipted launcher for docs/evidence/h01_anatomy_transfer_run.py on the Vast box.
# usage: run_transfer.sh LABEL WALL_CAP_S -- <runner args...>   (output is fixed to $OUT/run)
set -u
LABEL="$1"; CAP="$2"; shift 2; [ "$1" = "--" ] && shift
ROOT=/workspace/braintrace
OUT=$ROOT/var/h01-driven/$LABEL
mkdir -p "$OUT"
cd "$ROOT"
PY=/workspace/venv314/bin/python3
COMMIT=$(git rev-parse HEAD)
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
printf '{"label":"%s","status":"running","started_utc":"%s","source_commit":"%s","wall_cap_seconds":%s,"runner_sha256":"%s","arguments":%s}\n' \
  "$LABEL" "$START" "$COMMIT" "$CAP" "$(sha256sum docs/evidence/h01_anatomy_transfer_run.py | cut -c1-64)" \
  "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")" > "$OUT/launch.json"
timeout -k 30 "$CAP" /usr/bin/time -v -o "$OUT/time.txt" \
  env PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_MEM_FRACTION=.3 XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8 \
  taskset -c ${CPUSET:-200-231} $PY -u docs/evidence/h01_anatomy_transfer_run.py "$@" --output "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"
RC=$?
T1=$(date +%s.%N)
SECS=$($PY -c "print($T1-$T0)")
PEAK=$(grep 'Maximum resident' "$OUT/time.txt" | awk '{print $NF}')
printf '{"label":"%s","exit_code":%s,"wall_seconds":%s,"peak_rss_kib":%s,"finished_utc":"%s","timed_out":%s}\n' \
  "$LABEL" "$RC" "$SECS" "${PEAK:-null}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$([ $RC -eq 124 ] && echo true || echo false)" > "$OUT/terminal.json"
echo "DONE $LABEL rc=$RC secs=$SECS" >> "$OUT/run.log"
