# CellTransformer2

CellTransformer2 finds spatial domains in spatial transcriptomics data without predefined region labels.
It is a self-supervised encoder–decoder transformer: from the cells in a square neighborhood, it predicts a
center cell's masked gene expression, conditioned on the center cell's type. The trained encoder turns each
cell's neighborhood into an embedding, and k-means on those embeddings gives spatial domains at any number
of clusters.

CellTransformer2 keeps the objective of [CellTransformer](https://github.com/alexj-lee/spatialcelltransformer)
(Lee et al., 2025) and adds configurable model settings, sequence packing for neighborhoods of variable size,
and optimized spatial data loading. It scales to millions of cells on one workstation GPU.

The protocol is run from three Jupyter notebooks. Start with notebook 03 on the five example sections
(~13 min on one GPU), then point it at your own data.

## Requirements

- Linux x86_64 and an NVIDIA GPU with bfloat16 support.
- An NVIDIA driver that reports CUDA 12.2 or newer in `nvidia-smi`.
- Conda (we recommend [Miniforge](https://github.com/conda-forge/miniforge)).

## Installation

The notebooks use two conda environments:

- `celltransformer2-dev`: the notebook kernel, used for data preparation, training and embedding extraction.
- `celltransformer2-rapids`: GPU nearest-neighbor search and k-means (cuML) for smoothing and clustering.

Two environments are needed because cuDF 24.12 requires an older pyarrow than CellTransformer2.
The tested package versions are recorded in [`environment/`](environment/README.md).

Clone the repository and run all commands from its top-level folder:

> **TODO before publishing:** confirm the repository URL and folder name below once the repository is created
> (assumed: `abbasilab/celltransformer2`).

```bash
git clone https://github.com/abbasilab/celltransformer2.git
cd celltransformer2
```

Create the training environment and register it as a Jupyter kernel:

```bash
conda create -y -n celltransformer2-dev -c conda-forge python=3.11 pip
conda activate celltransformer2-dev
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install ipykernel
python -m ipykernel install --user --name celltransformer2-dev
```

Check it with `python -c "import torch, torchdata; print(torch.__version__, torchdata.__version__, torch.cuda.is_available())"`.
The expected output is `2.5.1+cu121 0.11.0 True`.

Create the RAPIDS environment:

```bash
conda create -y -n celltransformer2-rapids --override-channels \
    -c rapidsai -c conda-forge -c nvidia \
    python=3.11 cuml=24.12 cudf=24.12 cuda-version=12.2 numpy=1.26.4 \
    scipy=1.14.1 numba=0.60.0 numba-cuda=0.0.17.1 pynvjitlink=0.4.0 pip
conda activate celltransformer2-rapids
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install $(grep -v '^pyarrow==' requirements.txt)
pip install --no-deps -e .
```

Do not omit `--no-deps` from the last command: without it, pip upgrades pyarrow to 19.0.1 and cuDF stops working.
Check it with `python -c "import cuml, pyarrow; print(cuml.__version__, pyarrow.__version__)"` and
`celltransformer-smooth --help`. cuML should report 24.12 and pyarrow a version older than 19.

Then open a notebook and select the `celltransformer2-dev` kernel. The location of the RAPIDS environment
(`RAPIDS_BIN`) is set in Step 1 of each notebook.

## Notebooks

All three notebooks follow the same 13 steps: set paths, check the environment, write the standard input,
preprocess, check the files, choose the design, train, follow training, extract embeddings, smooth,
cluster, view the domains and (optionally) write an HTML report.

| Notebook | Data | What it shows |
| --- | --- | --- |
| [`01_allen_dataset.ipynb`](notebooks/01_allen_dataset.ipynb) | Allen Brain Cell Atlas MERFISH (MERFISH-C57BL6J-638850), all sections, ~9.3 GB download | Full whole-brain run with the original CellTransformer neighborhood and architecture settings |
| [`02_barseq_dataset.ipynb`](notebooks/02_barseq_dataset.ipynb) | BARseq whole-cortex pilot brain (Chen et al.), 40 sections, ~0.83 GB download | The same workflow on in situ sequencing data |
| [`03_custom_dataset.ipynb`](notebooks/03_custom_dataset.ipynb) | Five Allen MERFISH sections as a worked example | Template for your own data; settings to change are marked **REPLACE WITH YOUR DATA** |

The notebooks download and checksum the example data themselves. Every step after Step 3 reads one pair of
files per section: an `.h5ad` with raw counts (cells × genes) in `layers['raw']`, and a `.parquet` with the
same cells in the same order and columns `section`, `x`, `y` (µm) and `cell_type`. For your own data, write
these files in Step 3 of notebook 03. The optional HTML report is described in
[`docs/notebook_html_report.md`](docs/notebook_html_report.md).

## Commands

The notebooks run five command-line tools, which can also be run from a terminal (`<command> --help`):

| Command | Environment | What it does |
| --- | --- | --- |
| `celltransformer-preprocess` | training | Writes one `.h5ad`/`.parquet` pair per section, with log2(1 + x) of the raw counts in `.X` |
| `celltransformer-train` | training | Trains the model; configured with Hydra from [`config/`](config/), with any value overridable on the command line |
| `celltransformer-extract` | training | Writes one `*_embeddings.npz` per section from a trained checkpoint |
| `celltransformer-smooth` | RAPIDS | Gaussian-weighted average of each cell's embedding over its nearest neighbors, per section |
| `celltransformer-cluster` | RAPIDS | GPU k-means on the embeddings of all sections for each k; saves labels and per-section domain plots |

## Data and licenses

The code is MIT licensed ([`LICENSE`](LICENSE)). The example data are not redistributed here; the notebooks
download them at runtime from the URLs, with the sizes and checksums, listed in
[`allen_example/DOWNLOAD_MANIFEST.json`](allen_example/DOWNLOAD_MANIFEST.json) and
[`barseq_example/DOWNLOAD_MANIFEST.json`](barseq_example/DOWNLOAD_MANIFEST.json).

- Allen MERFISH data: [Allen Brain Cell Atlas](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html),
  CC BY-NC 4.0 (Yao et al., 2023).
- BARseq pilot brain: [Mendeley Data](https://doi.org/10.17632/8bhhk7c5n9.1), CC BY 4.0 (Chen et al., 2025).

The MIT license does not change the data's terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## Citation

> **TODO on publication:** add the protocol paper's reference (authors, title, journal, DOI) here.

If you use CellTransformer2, please cite the protocol and the original CellTransformer paper:

> Lee, A. J. et al. Data-driven fine-grained region discovery in the mouse brain with transformers.
> *Nat. Commun.* **16**, 8536 (2025). https://doi.org/10.1038/s41467-025-64259-4

If you use the example data, please also cite:

> Yao, Z. et al. A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain.
> *Nature* **624**, 317–332 (2023). https://doi.org/10.1038/s41586-023-06812-z

> Chen, X. et al. Whole-cortex in situ sequencing reveals input-dependent area identity.
> *Nature* **647**, 203–212 (2025). https://doi.org/10.1038/s41586-024-07221-6

## Credits

CellTransformer was written by Alex Lee; the original repository is
[alexj-lee/spatialcelltransformer](https://github.com/alexj-lee/spatialcelltransformer).
Alex also wrote much of CellTransformer2.
