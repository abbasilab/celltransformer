#!/usr/bin/env bash
set -euo pipefail
ALLEN_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_VISIBLE_DEVICES="${ALLEN_GPU:-0}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
ulimit -n 65536 2>/dev/null || true
exec python3 -u "$ALLEN_PACKAGE/pipeline.py" "$@"
