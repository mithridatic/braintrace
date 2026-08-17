#!/usr/bin/env bash
# Launch a braintrace-gpu container with the shared XLA compilation cache mounted.
#
# The image sets JAX_COMPILATION_CACHE_DIR=/cache/jax, but that path is
# container-local unless a host directory is bind-mounted over it. Without the
# mount every run recompiles and re-autotunes every XLA kernel from scratch.
# This wrapper always supplies the mount so compiled kernels and per-fusion
# autotune results are reused across runs and shared between concurrent
# containers.
#
# Usage:
#   ./run-gpu-container.sh [--cache-dir DIR] [--image IMAGE] [--mount HOST:CONTAINER]... -- CMD...
set -euo pipefail

cache_dir="${BRAINTRACE_JAX_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/braintrace/jax-cache}"
image='braintrace-gpu:0.11.0-py314'
mounts=()
envs=()
workdir=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir) cache_dir="$2"; shift 2 ;;
    --image)     image="$2";     shift 2 ;;
    --mount)     mounts+=(-v "$2"); shift 2 ;;
    --env)       envs+=(--env "$2"); shift 2 ;;
    --workdir)   workdir=(--workdir "$2"); shift 2 ;;
    --)          shift; break ;;
    *)           break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command supplied" >&2
  exit 2
fi

mkdir -p "$cache_dir"

exec docker run --rm --gpus all \
  -v "${cache_dir}:/cache/jax" \
  ${mounts[@]+"${mounts[@]}"} \
  ${envs[@]+"${envs[@]}"} \
  ${workdir[@]+"${workdir[@]}"} \
  "$image" \
  "$@"
