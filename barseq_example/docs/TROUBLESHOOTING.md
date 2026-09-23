# Troubleshooting

## Find the failure

Run `bash barseq_example/status_cactus.sh` with `BARSEQ_ROOT` set. Inspect the
reported stage log, for example `tail -n 120 "$BARSEQ_RUN/logs/train.log"`.
Resume the same run after fixing the cause; do not delete prepared data.

## CuBLAS deterministic-operation error

The fixed launcher exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.
If calling a Python stage directly, supply that variable from the shell. Setting
it after CUDA initializes is too late. Keep deterministic training enabled.

## RAPIDS dependency solver reports a missing/conflicting `__cuda`

The installer checks `nvidia-smi` for driver support of CUDA ≥12.2, then sets
`CONDA_OVERRIDE_CUDA=12.2` for its own process. This sets solver metadata; it does
not install or upgrade the NVIDIA driver. A driver that genuinely lacks the
required support must be addressed separately. No CUDA toolkit installation via
administrator commands is required by the example.

## DataLoader has few workers

The demonstrated configuration uses `workers=0`. This advisory warning does not
invalidate training. Lightning's suggested worker count is not a measured optimum
for the dataset; keep the proven setting for the example. Benchmark a separate
configuration if loading is later shown to be a bottleneck.

## Cleaning tarballs

Micromamba cleans its own download/index cache after installation. This may take
time depending on disk activity. The installed environments remain in place.

## Out of memory or insufficient disk

The recorded run used a GPU with approximately 48 GB VRAM and 128 GB system RAM.
The startup storage check requires 80 GiB free and stages retain a 20-GiB reserve.
Select a suitable `BARSEQ_ROOT`; do not delete another user's files or silently
reduce data, genes or neighborhoods to claim the same experiment.

## Training passed but HTML is missing

Smoothing, clustering, metrics and report generation follow training. Wait for
`ALL STAGES COMPLETED` and the run's `SUCCESS` file. A PID or a training success
line alone is insufficient. Keep the launcher running rather than invoking a
second pipeline concurrently.

## A high-k matched-correlation entry is empty

In the demonstrated run, unsmoothed CT2 at k=564 had 564 populated clusters across
all eligible cells but only 561 on the common reference-labelled evaluation set.
The code therefore leaves the complete one-to-one score undefined. This is
reported explicitly; the missing value is not replaced with zero or a partial
match presented as complete.

## An integrity audit rejects edited code

Each run snapshots its source. The native source manifest must continue to match
the demonstrated CellTransformer2 version. For intentional, reviewed changes to
the example adapter, run `python3 scripts/update_example_manifest.py`, rerun the
CPU checks, commit the changes and start a fresh run. Do not rewrite a completed
run's snapshot or clear its stage manifests to bypass an audit.
