# Allen 1 mouse MERFISH · CellTransformer2

A complete Linux GPU workflow for training the supplied CellTransformer2 from
random initialization on the primary Allen dataset from the original
CellTransformer paper, followed by anatomical comparisons and an offline HTML
report. The default budget is **10 full epochs**. No pretrained weights are
downloaded, and all model parameters are trained.

## Contents

- [Inputs and cohort](#inputs-and-cohort)
- [Requirements](#requirements)
- [Launch](#launch)
- [Monitor, stop and resume](#monitor-stop-and-resume)
- [Outputs](#outputs)
- [Training and comparison settings](#training-and-comparison-settings)
- [Reproducibility and validation](#reproducibility-and-validation)

## Inputs and cohort

The source is **MERFISH-C57BL6J-638850**, the Allen Institute animal called
**Allen 1** in the paper. This is not the four Zhuang animals or BARseq.

Downloads are pinned to the [20231215 Allen release manifest](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/releases/20231215/manifest.json).
The manifest selects the 20230830 measured raw matrix, the 20231215 cell-type and
CCF annotations, and the associated reference dictionaries and atlas volumes.
Every file is checked against its published byte count and MD5. Subsequent
launches reuse completed, verified files; interrupted downloads resume.

| Item | Selection |
|---|---|
| Raw matrix | 4,334,174 cells × 550 channels |
| Annotated cell/count intersection | 3,739,961 cells from 53 coronal sections |
| Eligible centers at the specified 85-µm window | 3,737,501; 2,460 isolated cells excluded as centers |
| Measured model genes | 500 biological genes; 50 Blank controls excluded |
| Cell-type inputs | 338 released subclasses, ordered and saved explicitly |
| Cell-type embedding capacity | 384, the supplied CT2 model configuration; 338 indices used |
| Anatomical reference | Released division / structure / substructure labels |
| Clustering resolutions | 25 / 354 / 670 |
| Source downloads | 9.27 GB total, including measured counts and atlas volumes |

**Cohort accounting:** the paper reports 3,737,550 cells. The verified public
annotation table contains 3,739,961 rows, all found in the raw count file. We do
not remove 2,411 arbitrary cells to force agreement. All those released rows
enter preparation, and cells without neighbors are flagged and excluded as
centers. The input and eligible-center counts are recorded separately. This is
CT2 on the primary public dataset, not a claim to recover the exact unpublished
analysis cell list or the original CT1 numerical results.
The full-coordinate audit found 3,737,501 eligible centers under this workflow's
neighborhood definition, 49 fewer than the paper's reported cell total.

All joins use `cell_label`. Counts retain their original HDF5 row order; both
source row indices and cell IDs are saved. Neighborhood coordinates use the
released `x_reconstructed` and `y_reconstructed`, converted from millimeters to
micrometers with a factor of **1,000**. CCF-warped positions are not used to form
model neighborhoods. Anatomical labels define the comparison resolutions and
evaluation reference; they are not model inputs or training targets.
Zero-expression cells, if present after selecting biological genes, remain in
the dataset and can be centers when they have neighbors.

The dictionary and two compressed annotation volumes are retained under
`data/raw`. Report maps use the released per-cell CCF assignments; the pipeline
does not re-register cells to the reference volume.

## Requirements

- Linux x86_64; reported workstation OS **Ubuntu 22.04.5 LTS**, kernel
  **5.15.0-190-generic**. A GPU with CUDA and bfloat16 support is required.
  The exact OS patch/kernel describe the workstation, not a minimum version requirement.
- The intended workstation has an RTX 6000 Ada with 48 GB GPU memory and
  approximately 128 GB system RAM. A full Allen GPU training record is not
  bundled in this repository. The GPU/model preflight is mandatory before training.
- An NVIDIA driver advertising CUDA 12.2 or newer; no separate system CUDA
  toolkit installation is required by these launchers.
- `bash`, `python3`, `curl`, `tar`, `sha256sum`, `nvidia-smi` and internet access.
- **At least 80 GiB free** at launch. This is a conservative working-space
  requirement, not the download size. Counts, embeddings, neighborhood graphs,
  checkpoints, environments and package caches also occupy space. A 20-GiB
  reserve is checked before expensive stages.

The installer creates isolated Python 3.11 environments, using the working
BARseq package pins: PyTorch 2.5.1/cu121, Lightning 2.4.0, torchdata 0.10.1 and
RAPIDS 24.12. See `environment/README.md`. A global conda installation is not
required. Existing environments can be reused read-only by specifying their
Python executables. The script validates them and stops if incompatible.

## Launch

From a repository checkout on the Linux GPU workstation:

```bash
cd /path/to/celltransformer2-test
python3 allen_example/source_layout.py
export ALLEN_ROOT="$HOME/Allen1_CellTransformer2"
bash allen_example/launch.sh
```

For installation with pip and the `celltransformer-allen` command, see the
[shared installation guide](../docs/PIP_INSTALLATION.md). Both interfaces use the
same scientific stages and output layout. The [main README](../README.md)
introduces this primary example alongside the secondary BARseq example.

To reuse the environments from the completed BARseq example:

```bash
export ALLEN_ROOT="$HOME/Allen1_CellTransformer2"
bash allen_example/launch.sh \
  --python "$HOME/BARseq_CT2_example/envs/celltransformer2-v1/bin/python" \
  --rapids-python "$HOME/BARseq_CT2_example/envs/rapids-24.12/bin/python"
```

Adjust these paths if the BARseq run used a different storage root; they must exist. Reusing environments does not load the BARseq checkpoint
or alter its data, run directories or installed packages.

The launcher uses `nohup`, disconnects standard input and prints the log path.
It continues on the remote workstation after the SSH connection closes. A launch
PID is not evidence that all stages passed. Do not launch a second copy to check
status; the root lock prevents concurrent runs in the same storage directory.

To choose another writable disk or GPU:

```bash
export ALLEN_ROOT="/path/to/writable/storage/Allen1_CellTransformer2"
export ALLEN_GPU=0
export ALLEN_EPOCHS=10
bash allen_example/launch.sh
```

For a foreground run, use `run_all.sh --root "$ALLEN_ROOT" --epochs 10`.
`--prepare-only` installs/checks dependencies, downloads and prepares the data,
then stops with status `PREPARED`; resume the same run to train. Downloading data
alone requires only the Python standard library and curl:

```bash
python3 allen_example/download.py --root "$ALLEN_ROOT"
```

## Monitor, stop and resume

```bash
bash allen_example/status.sh
ALLEN_RUN=$(cat "$ALLEN_ROOT/LATEST_ALLEN_CT2")
cat "$ALLEN_RUN/status.json"
tail -f "$ALLEN_RUN/logs/train.log"
```

`train.log` exists only when training starts. Earlier stages have separate logs
under `logs/environment.log`, `logs/download.log` and `logs/prepare.log` in the
run directory. The main log printed by the launcher announces each stage.
`Ctrl-C` stops `tail` without stopping training.

To stop the pipeline while retaining downloaded data and completed epochs:

```bash
ALLEN_RUN=$(cat "$ALLEN_ROOT/LATEST_ALLEN_CT2")
cat "$ALLEN_RUN/PID"
ps -p "$(cat "$ALLEN_RUN/PID")" -o pid,etime,args
kill -TERM "$(cat "$ALLEN_RUN/PID")"
```

Check the displayed process belongs to this run before sending the signal.
The coordinator terminates its active stage; data and completed checkpoints
remain. To resume after it has stopped:

```bash
bash allen_example/resume.sh
```

An interrupted epoch restarts from the last **completed epoch**. Export resumes
only against the same checkpoint hash. New `launch.sh` calls create new run
directories; `resume.sh` continues the existing one. Epoch budget, split and
model configuration are fixed for that run.

## Outputs

Stages run in order: **environment → download → prepare → train/export → smooth
→ cluster → metrics → report**.

Only a `SUCCESS` marker and `overall: PASSED` indicate full completion.
The marker contains the report path:

```bash
ALLEN_RUN=$(cat "$ALLEN_ROOT/LATEST_ALLEN_CT2")
cat "$ALLEN_RUN/SUCCESS"
ls -lh "$ALLEN_RUN/report/review.html" "$ALLEN_RUN/report/results.zip"
```

The standalone HTML includes:

- Training and validation NLL and the epoch table.
- All 53 section maps at each reference resolution, comparing CCF, CT2 without
  smoothing and CT2 with smoothing. Domain colors are stable across sections.
- Cortical enlargements where sufficiently represented, with every distinct
  anatomical label—including separate cortical layers—given a distinct color.
- Figure 2c–f-style spatial similarity, discreteness and composition comparisons.
- Finest-resolution region correlation and cell-cluster composition matrices.
- Exact preprocessing, metric definitions, settings and source provenance.

The approximate cortical outlines are nearest-cell raster boundaries, not
original atlas polygons. The report explains their construction. The comparison
does not include CellCharter, SPIRAL or other baseline methods.

`results.zip` includes the HTML, section PNGs, metric tables, training history,
gene/type mappings, source manifests and captured software/hardware records.
Large counts, embeddings and checkpoints remain in the run directory.

From a separate terminal on your local computer, replace `USER@GPU_HOST` and
`REMOTE_RUN` with the actual host and printed run path:

```bash
scp USER@GPU_HOST:REMOTE_RUN/report/results.zip "$HOME/Downloads/"
```

## Training and comparison settings

| Setting | This example |
|---|---|
| Model | Unchanged supplied `SpatialCellTransformer` |
| Initialization | Random; all parameters trained |
| Training budget | 10 complete epochs by default |
| Input transformation | `log2(raw measured counts + 1)` |
| Neighborhood | 85-µm-wide square, same section, center excluded |
| Encoder / decoder | 4 / 4 layers, 384 dimensions, 8 heads |
| Type input | Released subclass annotation; 338 used indices in capacity 384 |
| Objective | Native CT2 ZINB loss on molecule counts |
| Optimizer / schedule | Native CT2; LR 0.0003, weight decay 1e-8, warmup 500, timescale 5000 |
| Split | Seed 42; random 80/20 eligible centers across the dataset |
| Batch budget | Up to 256 centers and 16,384 tokens; no gradient accumulation |
| Precision | bfloat16 mixed training; float32 embedding export |
| Export | Final completed epoch checkpoint |
| Smoothing | Native Gaussian weights; sigma 12.01 µm; up to 500 same-section nearest neighbors including self |
| Clustering | Native single-GPU cuML KMeans, seed 42, `n_init=5` |
| Spatial metric | Fraction of up to 100 closest same-section evaluation neighbors sharing a label; self excluded |
| Discreteness | Fraction of model domains at or above the reference per-domain 20th-percentile spatial score |
| Composition | Subclass fractions; maximum Pearson correlation and optimal one-to-one assignment |

The reported sigma value follows the original paper's numerical **12.01 µm**.
The paper also calls this a 40-µm FWHM, which is inconsistent with the usual
Gaussian conversion; this workflow specifies sigma directly and does not equate
those two numbers.

This follows the successful finite-epoch BARseq adapter for CT2. It changes
dataset-dependent inputs, neighborhood width and reference labels, and uses an
80/20 split to follow the original Allen analysis more closely. The original
paper used **CT1**, 40 epochs, a peak LR of 0.001, weight decay 0.00005 and
different KMeans settings. These distinctions are recorded rather than hidden.
An epoch visits every eligible training center exactly once; its number of
optimizer updates depends on neighborhood sizes and the token budget.

Validation centers can appear as context for training centers. This is
exploratory anatomical discovery, not independent-mouse generalization.
Cell-type composition agreement is also not independent of the model inputs,
because the model uses expression-derived cell-type annotations.

## Reproducibility and validation

All **66 supplied CT2 source/config/test/license files are byte-identical** at
the repository root (with original README/project metadata under `provenance/upstream/`) and
checked against `CT2_SOURCE_MANIFEST.json`. The wrappers handle download checks,
cell joins, cached neighborhoods, single-GPU execution, finite epochs, logging,
resume and report generation. The mathematical model, objective, optimizer,
scheduler, Gaussian weight function and KMeans function are imported from CT2.

Each run saves a source snapshot and configuration. `celltransformer/input_schema.json`
records the gene order, subclass-to-index mapping, input transformation and
neighborhood settings alongside the trained checkpoint, so a future inference
workflow can reuse this model with verified inputs.

See [VALIDATION.md](VALIDATION.md) for completed checks and their limits.
The repository does not include Allen GPU training metrics; no synthetic CPU
output is presented as an Allen training result.

Sources: [CellTransformer paper](https://www.nature.com/articles/s41467-025-64259-4),
[Allen dataset description](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html),
and the pinned public release manifest. The dataset retains its source license;
see the Allen page. The original CT2 code license is in `../LICENSE`.
