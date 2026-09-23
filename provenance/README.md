# Source provenance

The source snapshot was supplied as the local `celltransformer2` directory. Its
`pyproject.toml` identifies Alex Lee as author and links to
https://github.com/alexj-lee/spatialcelltransformer. The supplied directory had no
Git history; no upstream commit hash or tagged version is asserted.

`barseq_example/CT2_SOURCE_MANIFEST.json` records SHA256 values for all 66 supplied
files. The manifest's own SHA256 is
`1a18efdedd9a8f64ced795005a0b3aaf524dca35deffd24e2e7d3b6de2a60b68`.

Original paths `README.md`, `.gitignore` and `.roborev.toml` are stored under
`provenance/upstream/`; other original paths remain at the repository root.
Run `python3 barseq_example/source_layout.py` to audit the source and example.

The original README is retained as historical source documentation. Its local
filesystem paths, installation notes and dataset-specific defaults are not the
tested BARseq instructions; use `barseq_example/README.md` instead.

The native model, ZINB objective, optimization code, budget batcher, Gaussian
weighting function and GPU KMeans function are used by the example. The adapter
adds the BARseq joins and count selection, cached section-local neighborhoods,
finite epochs, a one-GPU Lightning trainer, checkpoint recovery, export, metrics,
and report generation. It does not use the native multi-GPU data orchestrator.

At launch, the example copies the audited native source and its own scripts into
the run's `runner_snapshot/`. Existing snapshots and completed results remain
independent of later repository edits.

Packaging changes relative to the demonstrated launcher are limited to resolving
the root source tree, portable output paths, incorporating the proven CUDA launch
settings, saving additional environment records, and rendering the demonstrated
distinct-reference palette by default. Training parameters and scientific
transformations are unchanged. These packaging changes are CPU-tested; a new
10-epoch GPU run of this repository assembly has not been performed.
