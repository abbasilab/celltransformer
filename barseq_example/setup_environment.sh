#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Run this launcher on cactus (Linux x86_64)."; exit 1; }
BARSEQ_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${BARSEQ_ROOT:?Set by pipeline}"
: "${BARSEQ_PYTHON:?Set by pipeline}"
: "${BARSEQ_RAPIDS_PYTHON:?Set by pipeline}"
BARSEQ_MAMBA="$BARSEQ_ROOT/software/bin/micromamba"
export MAMBA_ROOT_PREFIX="$BARSEQ_ROOT/mamba-root" PYTHONNOUSERSITE=1 WANDB_MODE=disabled
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
if [[ ! -x "$BARSEQ_MAMBA" ]]; then
  mkdir -p "$BARSEQ_ROOT/software/bin"
  curl --fail --location --retry 5 --output "$BARSEQ_MAMBA.partial" \
    https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
  printf '%s  %s\n' '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3' "$BARSEQ_MAMBA.partial" | sha256sum --check
  chmod +x "$BARSEQ_MAMBA.partial"
  mv "$BARSEQ_MAMBA.partial" "$BARSEQ_MAMBA"
fi
BARSEQ_ENV="$(dirname "$(dirname "$BARSEQ_PYTHON")")"
if [[ ! -x "$BARSEQ_PYTHON" ]]; then
  "$BARSEQ_MAMBA" create -y -p "$BARSEQ_ENV" --override-channels -c conda-forge 'python=3.11' pip
fi
BARSEQ_REQ_SHA="$(cat "$BARSEQ_PACKAGE/requirements.txt" "$BARSEQ_PACKAGE/environment/ct2-pip-freeze.txt" | sha256sum | cut -d ' ' -f 1)"
if [[ ! -f "$BARSEQ_ENV/BARSEQ_REQUIREMENTS_SHA" ]] || [[ "$(cat "$BARSEQ_ENV/BARSEQ_REQUIREMENTS_SHA")" != "$BARSEQ_REQ_SHA" ]]; then
  "$BARSEQ_PYTHON" -m pip install --no-cache-dir 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
  "$BARSEQ_PYTHON" -m pip install --no-cache-dir -r "$BARSEQ_PACKAGE/requirements.txt" -c "$BARSEQ_PACKAGE/environment/ct2-pip-freeze.txt"
  "$BARSEQ_PYTHON" -m pip check
  printf '%s\n' "$BARSEQ_REQ_SHA" > "$BARSEQ_ENV/BARSEQ_REQUIREMENTS_SHA"
fi
"$BARSEQ_PYTHON" -m pip check
"$BARSEQ_PYTHON" - <<'PY'
import torch, lightning, torchdata
assert torch.__version__.split('+')[0]=='2.5.1'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('CT2 environment:',torch.__version__,lightning.__version__,torch.cuda.get_device_name())
PY
BARSEQ_RAPIDS_ENV="$(dirname "$(dirname "$BARSEQ_RAPIDS_PYTHON")")"
BARSEQ_RAPIDS_SPEC='python=3.11 cuml=24.12 cudf=24.12 cuda-version=12.2 numpy=1.26.4 scipy=1.14.1 numba=0.60.0 numba-cuda=0.0.17.1 pynvjitlink=0.4.0'
if [[ ! -f "$BARSEQ_RAPIDS_ENV/BARSEQ_RAPIDS_SPEC" ]] || [[ "$(cat "$BARSEQ_RAPIDS_ENV/BARSEQ_RAPIDS_SPEC")" != "$BARSEQ_RAPIDS_SPEC" ]]; then
  BARSEQ_MAMBA_ACTION=create
  if [[ -d "$BARSEQ_RAPIDS_ENV/conda-meta" ]]; then BARSEQ_MAMBA_ACTION=install; fi
  "$BARSEQ_MAMBA" "$BARSEQ_MAMBA_ACTION" -y -p "$BARSEQ_RAPIDS_ENV" --override-channels -c rapidsai -c conda-forge -c nvidia \
    'python=3.11' 'cuml=24.12' 'cudf=24.12' 'cuda-version=12.2' 'numpy=1.26.4' 'scipy=1.14.1' \
    'numba=0.60.0' 'numba-cuda=0.0.17.1' 'pynvjitlink=0.4.0' pip
  printf '%s\n' "$BARSEQ_RAPIDS_SPEC" > "$BARSEQ_RAPIDS_ENV/BARSEQ_RAPIDS_SPEC"
fi
# Plotting imports are required by the repository's native clustering entry point.
if ! "$BARSEQ_RAPIDS_PYTHON" -c 'import colorcet,seaborn,matplotlib,typer'; then
  "$BARSEQ_RAPIDS_PYTHON" -m pip install --no-cache-dir 'matplotlib==3.9.2' 'seaborn==0.13.2' 'colorcet==3.1.0' 'typer==0.15.2'
fi
export BARSEQ_PACKAGE
"$BARSEQ_RAPIDS_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0,os.environ["BARSEQ_PACKAGE"])
from source_layout import source_root
sys.path.insert(0,str(source_root()/"src"))
import numpy as np, cupy as cp
from celltransformer.cluster.main import _fit_single_gpu
from cuml.cluster import KMeans
x=np.random.default_rng(42).normal(size=(256,16)).astype('float32')
y=_fit_single_gpu(x,4,42)
assert len(np.unique(np.asarray(y)))==4
print('RAPIDS GPU KMEANS PREFLIGHT: PASSED')
PY
nvidia-smi > "$BARSEQ_ROOT/logs/nvidia_smi.txt"
if command -v lscpu >/dev/null 2>&1; then lscpu > "$BARSEQ_ROOT/logs/lscpu.txt"; fi
uname -a > "$BARSEQ_ROOT/logs/uname.txt"
cat /proc/meminfo > "$BARSEQ_ROOT/logs/meminfo.txt"
"$BARSEQ_PYTHON" -m pip freeze > "$BARSEQ_ROOT/logs/ct2_environment.txt"
"$BARSEQ_RAPIDS_PYTHON" -m pip freeze > "$BARSEQ_ROOT/logs/rapids_environment.txt"

"$BARSEQ_MAMBA" list -p "$BARSEQ_ENV" --explicit > "$BARSEQ_ROOT/logs/ct2_conda_explicit.txt"
"$BARSEQ_MAMBA" list -p "$BARSEQ_RAPIDS_ENV" --explicit > "$BARSEQ_ROOT/logs/rapids_conda_explicit.txt"
"$BARSEQ_MAMBA" list -p "$BARSEQ_RAPIDS_ENV" --json > "$BARSEQ_ROOT/logs/rapids_conda_packages.json"

# Only archives and index caches in this run root; installed environments remain.
"$BARSEQ_MAMBA" clean --tarballs --index-cache --yes
