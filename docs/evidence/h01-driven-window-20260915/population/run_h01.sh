#!/usr/bin/env bash
# Bounded, receipted launcher for examples.h01_verified_network on the Vast box.
# usage: run_h01.sh LABEL WALL_CAP_S -- <runner args...>
set -u
LABEL="$1"; CAP="$2"; shift 2; [ "$1" = "--" ] && shift
ROOT=/workspace/braintrace
OUT=$ROOT/var/h01-driven/$LABEL
mkdir -p "$OUT"
cd "$ROOT"
PY=/workspace/venv314/bin/python3
COMMIT=$(git rev-parse HEAD)
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
printf '{"label":"%s","status":"running","started_utc":"%s","source_commit":"%s","wall_cap_seconds":%s,"arguments":%s}\n' \
  "$LABEL" "$START" "$COMMIT" "$CAP" "$($PY -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")" > "$OUT/launch.json"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$OUT/gpu-before.txt"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 30; done ) > "$OUT/gpu-samples.txt" 2>/dev/null &
SAMPLER=$!
timeout -k 30 "$CAP" /usr/bin/time -v -o "$OUT/time.txt" \
  env PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_MEM_FRACTION=.92 XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8 \
  taskset -c ${CPUSET:-0-254} $PY -u -m examples.h01_verified_network "$@" --output "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"
RC=$?
kill $SAMPLER 2>/dev/null
T1=$(date +%s.%N)
SECS=$($PY -c "print($T1-$T0)")
PEAK=$(grep 'Maximum resident' "$OUT/time.txt" | awk '{print $NF}')
GPUMAX=$(sort -n "$OUT/gpu-samples.txt" | tail -1)
printf '{"label":"%s","exit_code":%s,"wall_seconds":%s,"peak_rss_kib":%s,"gpu_mem_max_mib":%s,"finished_utc":"%s","timed_out":%s}\n' \
  "$LABEL" "$RC" "$SECS" "${PEAK:-null}" "${GPUMAX:-null}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$([ $RC -eq 124 ] && echo true || echo false)" > "$OUT/terminal.json"
echo "DONE $LABEL rc=$RC secs=$SECS" >> "$OUT/run.log"
