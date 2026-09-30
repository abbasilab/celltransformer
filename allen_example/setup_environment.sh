#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Run this launcher on a Linux x86_64 GPU workstation."; exit 1; }
ALLEN_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ALLEN_ROOT:?Set by pipeline}"
: "${ALLEN_PYTHON:?Set by pipeline}"
: "${ALLEN_RAPIDS_PYTHON:?Set by pipeline}"
ALLEN_MAMBA="$ALLEN_ROOT/software/bin/micromamba"
export MAMBA_ROOT_PREFIX="$ALLEN_ROOT/mamba-root" PYTHONNOUSERSITE=1 WANDB_MODE=disabled
unset PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# Verify actual driver support before overriding conda virtual-package detection.
python3 - <<'PY'
import re, subprocess
smi = subprocess.check_output(["nvidia-smi"], text=True)
match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
if not match or tuple(map(int, match.groups())) < (12, 2):
    raise SystemExit("This example requires an NVIDIA driver advertising CUDA >=12.2")
print("Driver CUDA capability:", match.group(0))
PY
export CONDA_OVERRIDE_CUDA=12.2
if [[ "${ALLEN_REUSE_PYTHON:-0}" == 0 || "${ALLEN_REUSE_RAPIDS:-0}" == 0 ]]; then
if [[ ! -x "$ALLEN_MAMBA" ]]; then
  mkdir -p "$ALLEN_ROOT/software/bin"
  curl --fail --location --retry 5 --output "$ALLEN_MAMBA.partial" \
    https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
  printf '%s  %s\n' '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3' "$ALLEN_MAMBA.partial" | sha256sum --check
  chmod +x "$ALLEN_MAMBA.partial"
  mv "$ALLEN_MAMBA.partial" "$ALLEN_MAMBA"
fi
fi
if [[ "${ALLEN_REUSE_PYTHON:-0}" == 0 ]]; then
ALLEN_ENV="$(dirname "$(dirname "$ALLEN_PYTHON")")"
if [[ ! -x "$ALLEN_PYTHON" ]]; then
  "$ALLEN_MAMBA" create -y -p "$ALLEN_ENV" --override-channels -c conda-forge 'python=3.11' pip
fi
ALLEN_REQ_SHA="$(cat "$ALLEN_PACKAGE/requirements.txt" "$ALLEN_PACKAGE/environment/ct2-pip-freeze.txt" | sha256sum | cut -d ' ' -f 1)"
if [[ ! -f "$ALLEN_ENV/ALLEN_REQUIREMENTS_SHA" ]] || [[ "$(cat "$ALLEN_ENV/ALLEN_REQUIREMENTS_SHA")" != "$ALLEN_REQ_SHA" ]]; then
  "$ALLEN_PYTHON" -m pip install --no-cache-dir 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
  "$ALLEN_PYTHON" -m pip install --no-cache-dir -r "$ALLEN_PACKAGE/requirements.txt" -c "$ALLEN_PACKAGE/environment/ct2-pip-freeze.txt"
  "$ALLEN_PYTHON" -m pip check
  printf '%s\n' "$ALLEN_REQ_SHA" > "$ALLEN_ENV/ALLEN_REQUIREMENTS_SHA"
fi
fi
"$ALLEN_PYTHON" -m pip check
"$ALLEN_PYTHON" - <<'PY'
import torch, lightning, torchdata
assert torch.__version__.split('+')[0]=='2.5.1'
assert lightning.__version__=='2.4.0'
assert torchdata.__version__.split('+')[0]=='0.11.0'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('CT2 environment:',torch.__version__,lightning.__version__,torch.cuda.get_device_name())
PY
if [[ "${ALLEN_REUSE_RAPIDS:-0}" == 0 ]]; then
ALLEN_RAPIDS_ENV="$(dirname "$(dirname "$ALLEN_RAPIDS_PYTHON")")"
ALLEN_RAPIDS_SPEC='python=3.11 cuml=24.12 cudf=24.12 cuda-version=12.2 numpy=1.26.4 scipy=1.14.1 numba=0.60.0 numba-cuda=0.0.17.1 pynvjitlink=0.4.0'
if [[ ! -f "$ALLEN_RAPIDS_ENV/ALLEN_RAPIDS_SPEC" ]] || [[ "$(cat "$ALLEN_RAPIDS_ENV/ALLEN_RAPIDS_SPEC")" != "$ALLEN_RAPIDS_SPEC" ]]; then
  ALLEN_MAMBA_ACTION=create
  if [[ -d "$ALLEN_RAPIDS_ENV/conda-meta" ]]; then ALLEN_MAMBA_ACTION=install; fi
  "$ALLEN_MAMBA" "$ALLEN_MAMBA_ACTION" -y -p "$ALLEN_RAPIDS_ENV" --override-channels -c rapidsai -c conda-forge -c nvidia \
    'python=3.11' 'cuml=24.12' 'cudf=24.12' 'cuda-version=12.2' 'numpy=1.26.4' 'scipy=1.14.1' \
    'numba=0.60.0' 'numba-cuda=0.0.17.1' 'pynvjitlink=0.4.0' pip
  printf '%s\n' "$ALLEN_RAPIDS_SPEC" > "$ALLEN_RAPIDS_ENV/ALLEN_RAPIDS_SPEC"
fi
# Plotting imports are required by the repository's native clustering entry point.
if ! "$ALLEN_RAPIDS_PYTHON" -c 'import colorcet,seaborn,matplotlib,typer'; then
  "$ALLEN_RAPIDS_PYTHON" -m pip install --no-cache-dir 'matplotlib==3.9.2' 'seaborn==0.13.2' 'colorcet==3.1.0' 'typer==0.15.2'
fi
fi
export ALLEN_PACKAGE
"$ALLEN_RAPIDS_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0,os.environ["ALLEN_PACKAGE"])
from source_layout import source_root
sys.path.insert(0,str(source_root()/"src"))
import numpy as np, cupy as cp
from celltransformer.cluster.main import _fit_single_gpu
from cuml.cluster import KMeans
import cuml
assert cuml.__version__.startswith('24.12')
x=np.random.default_rng(42).normal(size=(256,16)).astype('float32')
y=_fit_single_gpu(x,4,42)
assert len(np.unique(np.asarray(y)))==4
print('RAPIDS GPU KMEANS PREFLIGHT: PASSED')
PY
nvidia-smi > "$ALLEN_ROOT/logs/nvidia_smi.txt"
if command -v lscpu >/dev/null 2>&1; then lscpu > "$ALLEN_ROOT/logs/lscpu.txt"; fi
uname -a > "$ALLEN_ROOT/logs/uname.txt"
if [[ -r /etc/os-release ]]; then cat /etc/os-release > "$ALLEN_ROOT/logs/os_release.txt"; fi
cat /proc/meminfo > "$ALLEN_ROOT/logs/meminfo.txt"
"$ALLEN_PYTHON" -m pip freeze > "$ALLEN_ROOT/logs/ct2_environment.txt"
"$ALLEN_RAPIDS_PYTHON" -m pip freeze > "$ALLEN_ROOT/logs/rapids_environment.txt"

if [[ -x "$ALLEN_MAMBA" ]]; then
  if [[ "${ALLEN_REUSE_PYTHON:-0}" == 0 ]]; then
    "$ALLEN_MAMBA" list -p "$ALLEN_ENV" --explicit > "$ALLEN_ROOT/logs/ct2_conda_explicit.txt"
  fi
  if [[ "${ALLEN_REUSE_RAPIDS:-0}" == 0 ]]; then
    "$ALLEN_MAMBA" list -p "$ALLEN_RAPIDS_ENV" --explicit > "$ALLEN_ROOT/logs/rapids_conda_explicit.txt"
    "$ALLEN_MAMBA" list -p "$ALLEN_RAPIDS_ENV" --json > "$ALLEN_ROOT/logs/rapids_conda_packages.json"
  fi
  "$ALLEN_MAMBA" clean --tarballs --index-cache --yes
fi
