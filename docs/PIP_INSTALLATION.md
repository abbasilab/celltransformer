# Pip installation and package builds

The distribution is named **`celltransformer2-test`**, version **0.2.0**. It
installs the native `celltransformer` Python package, six native console commands,
and the `celltransformer-allen` and `celltransformer-barseq` workflow commands. Python **3.11** is required for
this distribution's pinned dependencies. The original source version marker is
preserved; use `celltransformer-allen --version` or
`importlib.metadata.version('celltransformer2-test')` for the distribution version.

This package can be installed from a checkout, GitHub, a wheel or a source
archive. It has **not been published to PyPI**; `pip install celltransformer2-test`
alone is not the installation command for this repository.

## Install on a Linux GPU workstation

Use a fresh environment to avoid overlapping installations of another package
that also provides the `celltransformer` import namespace:

```bash
python3.11 -m venv "$HOME/.venvs/celltransformer2"
source "$HOME/.venvs/celltransformer2/bin/activate"
python -m pip install --upgrade pip

# Select the CUDA wheel used in the demonstrated experiment before installing CT2.
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121

# From the root of your clone:
python -m pip install .

celltransformer-allen --version
celltransformer-allen audit
celltransformer-barseq audit
```

Pip installs the Python dependencies declared in
[Allen](../allen_example/requirements.txt) and [BARseq](../barseq_example/requirements.txt) requirements, plus Click 8.1.8 from
[`packaging_requirements.txt`](../packaging_requirements.txt) for compatibility
with the native Typer 0.15.2 console help. The historical run used a newer Click
through direct function calls; its recorded environment remains unchanged.
Pip does not install or change the
NVIDIA driver. Running either complete workflow still requires the hardware, driver, network
and storage described in the [Allen](../allen_example/README.md) or
[BARseq](../barseq_example/README.md) guide.

Both run managers create pinned training and RAPIDS environments under `--root`. This preserves the tested CUDA runtime and conda
compatibility pins. The pip environment supplies the importable model and
installed commands; `pip install` alone does not create the GPU KMeans environment,
download the dataset or start training.

### Install directly from GitHub

While the repository is private, use an SSH identity that already has access:

```bash
python -m pip install 'git+ssh://git@github.com/ads303/celltransformer2-test.git@codex/barseq-example'
```

For an accessible repository over HTTPS, including after it becomes public:

```bash
python -m pip install 'git+https://github.com/ads303/celltransformer2-test.git@codex/barseq-example'
```

These commands require Git. For a publication, replace the branch ref after `@`
with the exact commit used. Authentication is handled by Git; do not embed a
token or password in a command, README or saved requirements file.

## Run the primary Allen 1 example

The Allen 1 MERFISH workflow uses the original CellTransformer paper's primary
dataset and trains CellTransformer2 from scratch for ten epochs by default:

```bash
celltransformer-allen run \
    --root "$HOME/Allen1_CellTransformer2" --epochs 10 --gpu 0 --background

celltransformer-allen status --root "$HOME/Allen1_CellTransformer2"

celltransformer-allen resume --root "$HOME/Allen1_CellTransformer2" --gpu 0 --background
```

To reuse compatible existing environments, pass their Python executables when
starting a new Allen run. The workflow validates them without modifying them:

```bash
celltransformer-allen run --root "$HOME/Allen1_CellTransformer2" \
    --python /absolute/path/to/ct2-environment/bin/python \
    --rapids-python /absolute/path/to/rapids-environment/bin/python \
    --epochs 10 --gpu 0 --background
```

`--prepare-only` stops after download and preparation. Resume that run to train
and finish the analyses. The run keeps its own configuration and source snapshot.

## Run the secondary BARseq example

These commands work from any directory after installation; no source checkout is
needed for a wheel installation:

```bash
celltransformer-barseq run \
    --root "$HOME/BARseq_CT2_example" --epochs 10 --gpu 0 --background

celltransformer-barseq status --root "$HOME/BARseq_CT2_example"

# Resume the latest run under the same storage root:
celltransformer-barseq resume --root "$HOME/BARseq_CT2_example" --gpu 0 --background
```

Omit `--background` for foreground execution or an allocated scheduler job. To
resume a particular run, supply `--run /absolute/path/to/runs/RUN_ID`; its saved
configuration determines the storage root. The original Bash helper filenames
are retained internally, and the stage logic, training settings and result layout
are the same as the checkout workflow.

`celltransformer-allen audit` and `celltransformer-barseq audit` each check both the normally imported native package
and the original source bundled for run snapshots. The wheel includes the
scripts for both examples, dependency definitions, reference metadata, source
manifests and native configurations. Downloaded measurements, model checkpoints and large
report images are not wheel contents.

## Native Python API and command-line tools

```python
from celltransformer.model import SpatialCellTransformer
from importlib.metadata import version

print(version('celltransformer2-test'))
```

The six native console tools are also available:

```bash
celltransformer-train --help
celltransformer-extract --help
celltransformer-preprocess --help
celltransformer-cluster --help
celltransformer-smooth --help
celltransformer-profile --help
```

The installed `celltransformer-train` command resolves the bundled Hydra
configuration directory before invoking the unchanged native training function.
Use this console entry point for installed native training; it supplies the
configuration path that the source-oriented `python -m celltransformer.train`
entry point does not resolve in a wheel. An explicit `--config-path` overrides
the bundled path. Native multi-GPU, data-path and logging defaults still require
configuration; use `celltransformer-allen run` or `celltransformer-barseq run` for the examples.

GPU operations require the corresponding CUDA/RAPIDS environment even when a
command's help or the package import works on a CPU. CPU software tests and
documentation inspection can use a CPU PyTorch wheel. A full training run is not
supported on macOS or a CPU-only machine.

## Editable installation for development

From a clone with the prerequisites above installed:

```bash
python -m pip install -e .
celltransformer-allen audit
celltransformer-barseq audit
```

An editable install reads the example resources from the checkout. A regular
wheel install uses its own bundled resources. Both create source snapshots for
new runs. After intentional adapter edits, update the example manifest and run
the relevant tests before starting a new experiment; existing run snapshots are
not overwritten. Use `python scripts/update_example_manifest.py --example allen`
or `--example barseq` after a reviewed adapter change; the default audits and
refreshes both examples. Native source changes fail this audit.

The original `pyproject.toml` is preserved in
[`provenance/upstream/pyproject.toml`](../provenance/upstream/pyproject.toml).
The root metadata describes this distribution. All 66 original files, including
the original version marker, retain their source-manifest checksums.

## Build and distribute package files

From the repository root, in a Python 3.11 build environment:

```bash
python -m pip install 'setuptools==80.9.0' 'wheel==0.45.1' 'build==1.2.2.post1' 'twine==6.1.0'
python -m build --no-isolation
python -m twine check dist/*
```

The build first creates a source distribution, then builds the wheel from that
archive. It does not require Git history, so an extracted source archive also
builds successfully. Both archives carry package metadata and licensing records.

```bash
python -m pip install /path/to/celltransformer2_test-0.2.0-py3-none-any.whl
# Alternatively:
python -m pip install /path/to/celltransformer2_test-0.2.0.tar.gz
```

Wheels bundle source and small resources, not their third-party dependencies;
pip normally retrieves those dependencies from package indexes. The automated
workflow uploads the wheel and source archive as the
`celltransformer2-pip-distributions` build artifact. This is separate from any
future PyPI publication.

Packaging follows the official [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/),
[setuptools build extension interface](https://setuptools.pypa.io/en/latest/userguide/extension.html)
and [pip VCS installation interface](https://pip.pypa.io/en/stable/topics/vcs-support/).
