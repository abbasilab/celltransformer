# Source provenance

The source snapshot was supplied as the local `celltransformer2` directory. Its
`pyproject.toml` identifies Alex Lee as author and links to
https://github.com/alexj-lee/spatialcelltransformer. The supplied directory had no
Git history; no upstream commit hash or tagged version is asserted.

`allen_example/CT2_SOURCE_MANIFEST.json` and
`barseq_example/CT2_SOURCE_MANIFEST.json` contain the same SHA256 values for all
66 supplied files. The manifest's own SHA256 is
`c9b7dd51ae597793af600f7f0dddb8954a86c22a6d4385e20c7b62ffa11310ed`.

The manifest as originally supplied had SHA256
`1a18efdedd9a8f64ced795005a0b3aaf524dca35deffd24e2e7d3b6de2a60b68`. It was
updated by hand when three core files were changed to make the training GPUs
configurable: `src/celltransformer/train/main.py`,
`config/train_redone_registration.yaml` and `config/train_zhuang.yaml`. Their
original and current hashes are listed under "Configurable training devices" in
the root README. The other 63 files are unchanged.

Original paths `README.md`, `.gitignore`, `.roborev.toml` and `pyproject.toml` are stored under
`provenance/upstream/`; other original paths remain at the repository root.
Run `python3 allen_example/source_layout.py` and
`python3 barseq_example/source_layout.py` to audit the source and both examples.

The original README is retained as historical source documentation. Its local
filesystem paths, installation notes and dataset-specific defaults are not the
example instructions; use `allen_example/README.md` for the primary Allen 1
workflow or `barseq_example/README.md` for the secondary BARseq workflow.

The native model, ZINB objective, optimization code, budget batcher, Gaussian
weighting function and GPU KMeans function are used by both examples. The adapters
add dataset-specific joins and count selection, cached section-local neighborhoods,
finite epochs, a one-GPU Lightning trainer, checkpoint recovery, export, metrics,
and report generation. Neither uses the native multi-GPU data orchestrator.

At launch, each example copies the audited native source and its own scripts into
the run's `runner_snapshot/`. Existing snapshots and completed results remain
independent of later repository edits.

Packaging changes relative to the demonstrated launcher are limited to resolving
the root source tree, portable output paths, incorporating the proven CUDA launch
settings, saving additional environment records, and rendering the demonstrated
distinct-reference palette by default. Training parameters and scientific
transformations are unchanged. These packaging changes are CPU-tested; a new
10-epoch GPU run of this repository assembly has not been performed.

Pip packaging adds distribution metadata, a resource-bundling build step and the
`celltransformer-allen` and `celltransformer-barseq` console commands. The root `pyproject.toml` now describes
the `celltransformer2-test` distribution; the original metadata remains in
`provenance/upstream/pyproject.toml` with its original checksum. The installed
native training command supplies the bundled Hydra configuration path and then
calls the native function, which is unchanged except that it reads its GPU list
from the `devices` configuration key instead of hardcoding eight GPUs. No model, loss, optimizer, scheduler or
scientific stage is rewritten for pip installation.

The wheel includes both the importable native package and an audited source copy
for run snapshots. Both are checked by either example
audit command. Editable
installs use the checkout's resources. Source archives and wheel builds use an
explicit distribution version and do not require Git history or regenerate the
original native `_version.py` file.

The Allen example was integrated from the separately delivered Allen 1 runner.
Its scientific settings are retained, including 10 full epochs, an 85-µm square
neighborhood, 20% random-center validation, and reference cuts 25/354/670. It
uses CT2 on the primary dataset from the CT1 paper; it is not a CT1 checkpoint
conversion. See `allen_example/VALIDATION.md` for measured-data and software
checks. Recorded BARseq figures do not represent an Allen training run.

The operating-system banner reported for the user's workstation is Ubuntu
22.04.5 LTS, GNU/Linux 5.15.0-190-generic x86_64. It is a workstation record, not
an exact kernel requirement or evidence of another completed GPU experiment.
New environment setups capture `/etc/os-release` and `uname -a` separately.
