# Validation and provenance — updated 24 September 2026

## Completed real GPU experiment

The BARseq pipeline completed on a Linux GPU workstation on 21 September 2026. The
downloaded `results.zip` contains the 10-epoch history, final-checkpoint summary,
CCF comparison tables, section figures and software records. Selected records
are preserved in `../results/` with member-level provenance and hashes.

The model was initialized randomly with all weights trainable. The native GPU
preflight passed, 10 full epochs produced 49,287 optimizer updates, and 1,161,154
embeddings were exported. Smoothing, clustering, metrics and HTML/ZIP generation
completed. The subsequent unique-reference-color refresh recorded unchanged
scientific inputs and unique colors for all 564 fine reference categories.

The actual run required the CUDA solver metadata and CuBLAS reproducibility
settings now included directly in the launcher/installer. Native CT2 code was
not edited to repair those environment issues.

## Repository assembly checks

Performed locally on macOS with Python 3.11 and PyTorch 2.5.1:

- All 66 original CellTransformer2 files verified against the demonstration's
  SHA256 source manifest, including relocated documentation/tooling originals.
- Python compilation and Bash syntax checks passed.
- **41 upstream CPU tests passed** for distributions, loader and preprocessing.
- BARseq release-join, duplicate/missing-ID, panel selection, CCF coordinate-axis,
  display-boundary and completed-download-resume checks passed.
- **Three packaging tests passed**: source snapshot byte equality, unique CCF
  colors with preserved anatomical identities and consistent cross-resolution
  mapping, and CuBLAS configuration present before Python starts.
- The synthetic adapter test passed native sampler tensor equality, 12 token and
  batch-budget combinations, a completed epoch followed by an interrupted epoch
  and checkpoint recovery, correct history rollback and complete center coverage.
- Smoothing passed an independent analytical two-point calculation and no
  cross-section mixing. Composition matching passed a permutation example with
  Pearson correlation 1.
- The assembled example generated all synthetic section/resolution plots and an
  offline HTML/ZIP. Synthetic reports explicitly identify themselves as tests.

The CPU orchestration test uses a clearly labelled toy forward model together
with the native loss, optimizer and scheduler. It does not stand in for a native
CUDA model run. Production always requires the real model's GPU preflight.

## What has not been re-run

The new root-source repository layout and integrated defaults have not received a
second full Linux environment installation and 10-epoch GPU run. Core scientific
code and settings match the demonstrated run; packaging and presentation changes
are covered by the local checks above. No claim of bitwise-identical results on
different hardware or dependency builds is made.

The GitHub Actions workflow repeats source, compilation and CPU adapter checks;
it does not download the full dataset or claim a GPU performance benchmark.

The initial GitHub run passed the source audit and all 44 pytest checks, then
stopped because the 96-cell synthetic adapter fixture inherited the full-data
20-GiB free-space reserve. The fixture now explicitly uses a 1-GiB reserve and
caps reported free space at 2 GiB during graph preparation to exercise this
constraint on developer machines too. Production retains its 20-GiB reserve;
the data, model, training and analysis code are unchanged by this test fix.

The original pre-GPU preparation/validation notes are retained in
`initial_package_validation.md` as a historical record, superseded by the
completed-run evidence described here.

## Re-run CPU checks

In a suitable Python environment, from the repository root:

```bash
python3 barseq_example/source_layout.py
python -m compileall -q src barseq_example scripts
python barseq_example/test_barseq.py
python -m pytest -q barseq_example/test_packaging.py
PYTHONPATH=src python -m pytest -q tests/test_distributions.py tests/test_loader.py tests/test_preprocess.py
python barseq_example/test_adapter.py --output /tmp/barseq-synthetic-NEW
```

The synthetic output directory must not already exist. Never use a real run
directory as the test output. See the Actions workflow for CPU dependency setup.
