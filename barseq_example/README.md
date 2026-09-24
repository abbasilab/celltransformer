# BARseq: download → CellTransformer2 → CCF comparison → HTML

This example packages the completed 10-epoch BARseq experiment. It trains the
supplied CellTransformer2 model from scratch, then compares **CellTransformer
(-smoothing)** and **CellTransformer (+smoothing)** with **CCF (author
registration)**. It runs no CellCharter, SPIRAL or other model baselines.

## Requirements

- Linux x86_64; system Python 3, Bash, curl, tar and standard GNU command-line tools.
- Internet access to GitHub, PyPI/PyTorch, conda channels, Mendeley and Allen downloads.
- NVIDIA GPU with bfloat16 support; an NVIDIA driver advertising CUDA ≥12.2.
- Demonstrated hardware: RTX 6000 Ada, approximately 48 GB GPU memory, 128 GB RAM,
  AMD Ryzen 9 7950X; driver 535.309.01. One selected GPU is used.
- At least **80 GiB initially free**. Initial planning allowed 50–70 GiB for
  installation and one full run, not a measured guaranteed maximum. A 20-GiB free
  reserve is checked between stages. Other filesystem users can affect this.

No administrator access, pre-existing conda installation, MATLAB, external
pretrained checkpoint or Weights & Biases login is needed. The installer creates
its own pinned micromamba and two environments inside the chosen storage root.

## Launch on a Linux GPU workstation

From the repository root:

```bash
export BARSEQ_ROOT="$HOME/BARseq_CT2_example"
export BARSEQ_GPU=0
export BARSEQ_EPOCHS=10
bash barseq_example/launch_cactus.sh
```

`launch_cactus.sh`, `status_cactus.sh` and `resume_cactus.sh` are the existing
helper filenames. They work on any Linux GPU workstation meeting the requirements
above; no particular hostname is required.

Choose a storage root **outside the source checkout**. The new default differs
from the historical `BARseq_CellTransformer` directory, preserving that experiment.
The launcher starts `nohup` in the background and prints the PID and log. If you
launch it over SSH, you can disconnect or close your local computer; the process
continues on the remote GPU workstation while that workstation remains running.
On a managed cluster with a scheduler, run `run_all.sh` inside an allocated GPU
job instead of starting a job on a login node.

For foreground execution, or use inside an allocated job:

```bash
bash barseq_example/run_all.sh --root "$BARSEQ_ROOT" --epochs 10
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set **before Python starts**, preserving
deterministic training. The installer verifies the NVIDIA driver's CUDA capability
before setting `CONDA_OVERRIDE_CUDA=12.2` for its RAPIDS dependency solve. These
settings incorporate the repairs used in the demonstrated run; do not change the
NVIDIA driver or disable deterministic training to work around those messages.

## Check progress

Keep `BARSEQ_ROOT` set to the same path in each terminal:

```bash
export BARSEQ_ROOT="$HOME/BARseq_CT2_example"
bash barseq_example/status_cactus.sh

BARSEQ_RUN=$(cat "$BARSEQ_ROOT/LATEST_BARSEQ_CT2")
tail -n 30 -F "$BARSEQ_RUN/logs/train.log"
```

The stages are `environment`, `prepare`, `train`, `smooth`, `cluster`, `metrics`
and `report`. Each has its own log in `$BARSEQ_RUN/logs/`. `Ctrl+C` stops following
the log without stopping the job. The helper checks both status and the recorded
PID; a stale `RUNNING` JSON value alone is not proof a process is alive.

**Training completion is not report completion.** The finished run has
`status.json` with `overall=PASSED`, a `SUCCESS` file, `report/review.html` and
`report/results.zip`.

## Stop or resume

To stop the whole active pipeline gracefully, first inspect its recorded PID:

```bash
BARSEQ_RUN=$(cat "$BARSEQ_ROOT/LATEST_BARSEQ_CT2")
BARSEQ_PID=$(cat "$BARSEQ_RUN/PID")
ps -p "$BARSEQ_PID" -o pid,etime,args
# If this is the pipeline process for the intended run:
kill -TERM "$BARSEQ_PID"
```

Do not delete data, checkpoints or stage manifests. Resume with:

```bash
bash barseq_example/resume_cactus.sh "$BARSEQ_RUN"
```

Environment checks rerun. Completed stages are reused. Training restores the last
completed-epoch checkpoint; an interrupted partial epoch repeats, with partial
history rolled back. The run snapshot protects the resumed computation from
later edits to the repository.

## Retrieve and open the report on your local computer

After all stages pass, run these commands **on your local computer** in a
Bash-compatible terminal, such as macOS Terminal, a Linux terminal or WSL.
Replace `your_username@your_gpu_host` with your SSH login and hostname, and set
`BARSEQ_REMOTE_ROOT` to the absolute storage path used on that remote machine:

```bash
BARSEQ_HOST="your_username@your_gpu_host"
BARSEQ_REMOTE_ROOT="/absolute/path/to/BARseq_CT2_example"
BARSEQ_REMOTE_RUN=$(ssh "$BARSEQ_HOST" "cat '$BARSEQ_REMOTE_ROOT/LATEST_BARSEQ_CT2'")
ssh "$BARSEQ_HOST" "test -f '$BARSEQ_REMOTE_RUN/SUCCESS'" && \
mkdir -p "$HOME/Downloads/BARseq_CT2_example" && \
scp "$BARSEQ_HOST:$BARSEQ_REMOTE_RUN/report/review.html" \
    "$BARSEQ_HOST:$BARSEQ_REMOTE_RUN/report/results.zip" \
    "$HOME/Downloads/BARseq_CT2_example/"
```

Open the downloaded `Downloads/BARseq_CT2_example/review.html` in your web browser.
On macOS, you can also run `open "$HOME/Downloads/BARseq_CT2_example/review.html"`;
on a Linux desktop, use `xdg-open` with the same path. If the browser runs on the
GPU workstation itself, open `$BARSEQ_RUN/report/review.html` directly.

The HTML embeds its images and works offline. It includes all 40 sections at all
three reference resolutions, cortical enlargements, training diagnostics, Figure
2c–i-style comparisons and provenance. Reference labels have unique colors,
including individual cortical layers; the finest resolution is selected initially.

The ZIP includes figures, tables, configuration, source manifests and environment
records. Counts, checkpoints and embeddings remain on the GPU machine. The full
HTML and ZIP can be hundreds of MB and are not meant to be committed to Git.

## Files and documentation

| File | Purpose |
|---|---|
| `pipeline.py` | Stage orchestration, locks, snapshots and resume |
| `setup_environment.sh` | Pinned micromamba, PyTorch and RAPIDS setup; GPU checks |
| `requirements.txt`, `environment/` | Dependency definitions and run records |
| `prepare_barseq.py` | Verified downloads, ID joins, targets, anatomy, neighborhoods |
| `data_ct2.py`, `train_ct2.py` | Native-model training with finite epochs and export |
| `analyze.py`, `cluster.py` | Section-safe smoothing, GPU KMeans, comparison metrics |
| `report.py`, `reference_colors.py` | Offline report, figures and distinct CCF palette |
| `results/` | Selected evidence from the completed real experiment |

Read [methods](docs/METHODS.md), [dependencies](environment/README.md),
[troubleshooting](docs/TROUBLESHOOTING.md) and [validation](docs/VALIDATION.md).
